"""Config-driven pipeline report: run the pipeline on each (pdf, page) of a JSON config and write
everything needed to look at the run (`pipeline_report_viewer.py`) and to score it
(`pipeline_report_benchmark.py`).

    python scripts/generate_pipeline_report.py --config scripts/report_configs/example.json

Output (`outputs/pipeline_report/<ts>__<config-stem>/`):

    config_and_hyperparameters.txt     the config, its path, and the RunSettings used
    benchmark.json                     benchmark mode only: the inputs, for the benchmark script
    p2_work/                           P2 backends' own intermediate files (RunSettings.dump_dir)
    <doc>/
        manifest.json                  source PDF, pages, backends, and the layer list
        dump.json                      every page's final texts + vectors + timings
        <stage>__<layer>.pdf           one multi-page PDF per debug layer the backends emit,
                                       plus reconstructed__page.pdf (the final output)
        <folder>/                      debug images the backends offer (capped per folder)
        converted_p<N>.pdf             the converted input (vectorise / benchmark modes)
        benchmark mode only:
        ground_truth_<type>.json       the labels scored, per text type
        <type>_bbox.pdf, <type>_text.pdf    GT overlays (green found / red missed)
        benchmark__extra_*.pdf, benchmark__missed_vectors.pdf   (inputs with vector labels)

`<doc>` is the PDF stem, or in benchmark mode the input key without its `pdf:` / `labels:`
prefix. Config fields: `scripts/report_configs/README.md`; the whole workflow: `docs/SCRIPTS.md`.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gc
import json
import sys
import time
from pathlib import Path

import pymupdf as fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raster_parser.commons.logging_setup import configure_logging, get_logger  # noqa: E402
from raster_parser.commons.progress import bar, stage  # noqa: E402
from raster_parser.commons.paths import output_dir  # noqa: E402
from raster_parser.core.pipeline import run_pipeline  # noqa: E402
from raster_parser.core.settings import RunSettings  # noqa: E402
from raster_parser.evaluation import conversion, dump_io  # noqa: E402
from raster_parser.evaluation.evaluate import adapters, metrics  # noqa: E402
from raster_parser.evaluation.evaluate.metrics import TEXT_TYPES  # noqa: E402
from raster_parser.evaluation.labelling import native_label  # noqa: E402
from raster_parser.evaluation.labelling.label_schema import (  # noqa: E402
    LabelSet,
    load_labels,
    load_labels_from_master_folder,
    save_labels,
)
from raster_parser.evaluation.report.report_artifacts import (  # noqa: E402
    ImageReservoirs,
    LayerWriter,
    add_extra_prediction_layers,
    debug_image_sink,
    debug_layer_sink,
    finalize_doc_dir,
    page_geometry,
    reconstructed_layer,
    restamp_page,
    text_routed_vectors,
    write_hyperparams,
    write_label_overlays,
)
from raster_parser.evaluation.report.report_config import CONVERT, BenchInput, ReportConfig  # noqa: E402
from raster_parser.P1_Reading_Native.phase1 import read_and_extract  # noqa: E402

_LOG = get_logger("generate_pipeline_report")

OVERLAY_COLORS = {"native_to_vector": "#22c55e", "original_vector": "#2563eb"}


def filter_valid_pages(pdf_path: Path, pages: list[int], label: str) -> list[int]:
    """Drop page indices out of range for `pdf_path` (warned once per input) -- a shared `pages`
    list often doesn't fit every input."""
    with fitz.open(str(pdf_path)) as doc:
        page_count = doc.page_count
    valid = [p for p in pages if 0 <= p < page_count]
    invalid = [p for p in pages if p not in valid]
    if invalid:
        _LOG.warning("%s: skipping out-of-range page(s) %s (%s has %d pages)",
                     label, invalid, pdf_path.name, page_count)
    return valid


def bench_doc_name(bench: BenchInput) -> str:
    """Folder name for a benchmark input: its key without the `pdf:` / `labels:` prefix (a
    master_label folder's PDF is always `original.pdf`, so the PDF stem would collide)."""
    name = bench.key.split(":", 1)[-1]
    return "".join(c if c not in '<>:"/\\|?*' else "_" for c in name) or "input"


def _step_time_sink(sub_durations: dict):
    """`RunSettings.on_step_time` that sums each `(phase, name)` into `sub_durations`."""
    def sink(phase: str, name: str, seconds: float) -> None:
        per_phase = sub_durations.setdefault(phase, {})
        per_phase[name] = per_phase.get(name, 0.0) + seconds
    return sink


def _settings(config: ReportConfig, run_dir: Path, writer: LayerWriter | None,
              reservoirs: ImageReservoirs | None, page_index: int,
              sub_durations: "dict | None" = None) -> RunSettings:
    return RunSettings(
        ocr_dpi=config.ocr_dpi, use_angle_cls=config.use_angle_cls, workers=config.workers,
        dump_dir=run_dir / "p2_work",
        on_debug_layer=debug_layer_sink(writer) if writer is not None and config.debug_layers else None,
        on_debug_image=debug_image_sink(reservoirs, page_index) if reservoirs is not None else None,
        on_step_time=_step_time_sink(sub_durations) if sub_durations is not None else None,
    )


def _run_page(config: ReportConfig, run_dir: Path, run_input: str, run_page: int, page_index: int,
              writer: LayerWriter, reservoirs: ImageReservoirs | None, debug_durations: dict):
    """One pipeline run of `run_input` page `run_page` (report page `page_index`) with the
    report's callbacks, plus the reconstructed layer. Returns `(result, sub_durations)`."""
    sub_durations: dict = {}
    res = run_pipeline(
        run_input, run_page, p2=config.p2, p3=config.p3, stop_after=config.final_stage,
        settings=_settings(config, run_dir, writer, reservoirs, page_index, sub_durations),
    )
    if config.debug_layers:
        start = time.perf_counter()
        reconstructed_layer(writer, res)
        debug_durations["reconstructed"] = time.perf_counter() - start
    return res, sub_durations


def _new_page_dump(res, config: ReportConfig, debug_durations: dict,
                   sub_durations: "dict | None" = None) -> dump_io.PageDump:
    return dump_io.PageDump(
        page_meta=res.page.meta, texts=list(res.texts), vectors=list(res.vectors),
        p2=config.p2, p3=config.p3, step_durations=dict(res.step_durations),
        sub_durations=dict(sub_durations or {}), debug_durations=debug_durations,
    )


def process_pdf(pdf_path: Path, config: ReportConfig, run_dir: Path) -> None:
    doc_dir = run_dir / pdf_path.stem
    doc_dir.mkdir(parents=True, exist_ok=True)
    reservoirs = ImageReservoirs(doc_dir, config.debug_image_cap) if config.debug_images else None
    pages = filter_valid_pages(pdf_path, config.pages_for(pdf_path.stem), pdf_path.stem)
    writer = LayerWriter(doc_dir / ".layer_parts")
    dumps: list[dump_io.PageDump] = []

    for pos, page_index in enumerate(bar(pages, desc=pdf_path.stem, unit="page")):
        debug_durations: dict = {}
        run_input, run_page = str(pdf_path), page_index
        if config.vectorise:
            conv_path = doc_dir / f"converted_p{page_index}.pdf"
            start = time.perf_counter()
            with stage(f"report › convert {pdf_path.stem} p{page_index} ({config.vectorise_mode})"):
                CONVERT[config.vectorise_mode](str(pdf_path), page_index, str(conv_path))
            debug_durations["conversion"] = time.perf_counter() - start
            run_input, run_page = str(conv_path), 0
        writer.begin_page(pos, *page_geometry(run_input, run_page))

        _LOG.info("running %s page %d (p2=%s, p3=%s)", pdf_path.name, page_index, config.p2, config.p3)
        res, sub_durations = _run_page(config, run_dir, run_input, run_page, page_index, writer, reservoirs,
                                       debug_durations)
        restamp_page(res, page_index)
        dumps.append(_new_page_dump(res, config, debug_durations, sub_durations))
        if config.vectorise and not config.keep_converted_pdfs:
            conv_path.unlink(missing_ok=True)
        del res
        gc.collect()

    with stage(f"report › write {pdf_path.stem} layers + dump.json"):
        finalize_doc_dir(doc_dir, pdf_path, pages, config, writer, dumps)


def bench_ground_truth_by_type(bench: BenchInput, pages: list[int]) -> "dict[str, LabelSet]":
    """One `LabelSet` per scored text type (every `TEXT_TYPES` key present), filtered to `pages`.

    `native_to_vector` comes from a master_label folder's own `native_labels.json` when there is
    one, else it is derived from the PDF's native text (`native_label.native_label_pdf`).
    `original_vector` comes from `bench.labels_path` (a label JSON or a master_label folder)."""
    pdf_str = str(bench.pdf_path)
    labels = None
    if bench.labels_path is not None:
        labels = (load_labels_from_master_folder(bench.labels_path) if bench.labels_path.is_dir()
                  else load_labels(str(bench.labels_path)))
    buckets = adapters.entries_by_text_type(labels) if labels is not None else {t: [] for t in TEXT_TYPES}

    native = buckets["native_to_vector"]
    if not native:
        native = [e for p in pages for e in native_label.native_label_pdf(pdf_str, p).entries]
    return {
        "native_to_vector": LabelSet(pdf_path=pdf_str, entries=[e for e in native if e.page_index in pages]),
        "original_vector": LabelSet(
            pdf_path=pdf_str, entries=[e for e in buckets["original_vector"] if e.page_index in pages]),
    }


def process_pdf_benchmark(bench: BenchInput, config: ReportConfig, run_dir: Path) -> dict:
    """One benchmark input -> `run_dir/<name>/`: per page, a `convert_page_to_vector_text` copy
    (native text drawn as vector paths over the untouched drawings) goes through the pipeline,
    and its OCR text is what the benchmark scores. Writes the ground truth and its overlays.
    Returns the `benchmark.json` entry."""
    doc_name = bench_doc_name(bench)
    doc_dir = run_dir / doc_name
    doc_dir.mkdir(parents=True, exist_ok=True)
    reservoirs = ImageReservoirs(doc_dir, config.debug_image_cap) if config.debug_images else None
    pages = filter_valid_pages(bench.pdf_path, config.pages_for(doc_name), bench.key)
    cfg = metrics.MetricConfig(iou_edge_min=config.iou_edge_min)
    with stage(f"report › ground truth {bench.key}"):
        gt_by_type = bench_ground_truth_by_type(bench, pages)
    has_manual = bool(gt_by_type["original_vector"].entries)

    writer = LayerWriter(doc_dir / ".layer_parts")
    dumps: list[dump_io.PageDump] = []
    for pos, p in enumerate(bar(pages, desc=doc_name, unit="page")):
        debug_durations: dict = {}
        conv_path = doc_dir / f"converted_p{p}.pdf"
        start = time.perf_counter()
        with stage(f"report › convert {doc_name} p{p} (to_vector_text)"):
            conversion.convert_page_to_vector_text(str(bench.pdf_path), p, str(conv_path))
        debug_durations["conversion"] = time.perf_counter() - start
        writer.begin_page(pos, *page_geometry(conv_path, 0))

        _LOG.info("benchmark %s page %d (p2=%s, p3=%s)", bench.key, p, config.p2, config.p3)
        res, sub_durations = _run_page(config, run_dir, str(conv_path), 0, p, writer, reservoirs,
                                       debug_durations)
        if has_manual and config.debug_layers:
            start = time.perf_counter()
            with stage(f"report › extra-prediction layers {doc_name} p{p}"):
                add_extra_prediction_layers(
                    writer, res.page.meta, list(res.texts),
                    text_routed_vectors(read_and_extract(str(conv_path), 0).vectors, list(res.vectors)),
                    list(res.vectors),
                    gt_boxes=[tuple(e.cluster_bbox) for t in TEXT_TYPES
                              for e in gt_by_type[t].entries if e.page_index == p],
                    manual_boxes=[tuple(e.cluster_bbox) for e in gt_by_type["original_vector"].entries
                                  if e.page_index == p],
                )
            debug_durations["extra_predictions"] = time.perf_counter() - start
        restamp_page(res, p)
        dumps.append(_new_page_dump(res, config, debug_durations, sub_durations))
        if not config.keep_converted_pdfs:
            conv_path.unlink(missing_ok=True)
        del res
        gc.collect()

    sources: list[str] = []
    extra_layers: list[dict] = []
    doc_durations: dict = {}
    for text_type in TEXT_TYPES:
        gt = gt_by_type[text_type]
        if not gt.entries:
            continue
        save_labels(gt, str(doc_dir / f"ground_truth_{text_type}.json"))
        sources.append(text_type)
        if config.debug_layers:
            start = time.perf_counter()
            with stage(f"report › {text_type} label overlays {doc_name}"):
                write_label_overlays(doc_dir, text_type, gt, dumps, cfg)
            doc_durations["label_overlays"] = doc_durations.get("label_overlays", 0.0) + time.perf_counter() - start
            for kind in ("bbox", "text"):
                extra_layers.append({
                    "stage": "benchmark", "layer": f"{text_type} label {kind}",
                    "file": f"{text_type}_{kind}.pdf", "color": OVERLAY_COLORS[text_type],
                })

    with stage(f"report › write {doc_name} layers + dump.json"):
        finalize_doc_dir(doc_dir, bench.pdf_path, pages, config, writer, dumps,
                         extra_layers=tuple(extra_layers), doc_durations=doc_durations)
    return {
        "key": bench.key,
        "pdf_stem": bench.pdf_path.stem,
        "source_pdf": str(bench.pdf_path),
        "labels": str(bench.labels_path) if bench.labels_path else None,
        "pages": pages,
        "dir": doc_name,
        "sources": sources,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path, help="Path to the run config JSON.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging()
    config = ReportConfig.model_validate_json(Path(args.config).read_text(encoding="utf-8"))

    root = config.output_root or output_dir("pipeline_report")
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / f"{ts}__{Path(args.config).stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_hyperparams(run_dir / "config_and_hyperparameters.txt", config,
                      _settings(config, run_dir, None, None, 0), Path(args.config))

    if config.benchmark:
        inputs = config.benchmark_inputs()
        if not inputs:
            _LOG.error("no benchmark inputs (set input_dir and/or input_files)")
            return 1
        entries = [process_pdf_benchmark(bench, config, run_dir)
                   for bench in bar(inputs, desc="inputs", unit="input")]
        (run_dir / "benchmark.json").write_text(json.dumps({
            "benchmark": True,
            "config_stem": Path(args.config).stem,
            "config_path": str(Path(args.config).resolve()),
            "p2": config.p2,
            "p3": config.p3,
            "final_stage": config.final_stage,
            "iou_edge_min": config.iou_edge_min,
            "entries": entries,
        }, indent=2), encoding="utf-8")
        print(f"wrote {run_dir}")
        return 0

    pdfs = config.resolved_pdfs()
    if not pdfs:
        _LOG.error("no input PDFs (set input_dir and/or input_files)")
        return 1
    for pdf_path in bar(pdfs, desc="inputs", unit="pdf"):
        process_pdf(pdf_path, config, run_dir)
    print(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
