"""Config-driven pipeline stage report generator.

Reads a JSON config, runs the pipeline once per (pdf, page), and writes a
timestamped run folder:

    outputs/pipeline_report/<ts>__<config-stem>/
        config_and_hyperparameters.txt
        <pdf-stem>/
            manifest.json
            <stage>__<layer>.pdf                       (one multi-page PDF per
                                                       visual layer, e.g.
                                                       vector_classification__group_bbox.pdf;
                                                       the viewer toggles each)
            native_text.txt ...                       (one stats file per stage)
            dump.json                                 (every Text + Vector, reloadable)
            radon_images/    (one PNG per FAST-surviving cluster: the rendered
                             cluster image Radon sees *before* segmentation,
                             with the detected word/segment boxes drawn on it)
            paddle_images/   (one PNG per elected unique segment: the exact
                             deskewed, white-padded crop handed to PaddleOCR
                             *before* recognition, unpadded word region boxed,
                             recognised text in the filename)

Replaces `rastervec/notebooks/pipeline_stage_visualization.ipynb`.

With `benchmark: true` the same `<pdf-stem>/` folder is also a scoring
artifact for `pipeline_report_benchmark.py`: one `convert_page_to_vector_text`
run per page, plus `ground_truth_{auto,manual}.json` and the split
`{auto,manual}_{bbox,text}.pdf` overlays (registered in the manifest under
stage `benchmark`), and a run-root `benchmark.json` marker. See
`docs/SCRIPTS.md`.

    .venv/Scripts/python.exe scripts/generate_pipeline_report.py --config run.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pymupdf as fitz
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field, field_validator, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rastervec.config as rvconfig
from rastervec.Evaluation import conversion, dump_io
from rastervec.Evaluation.Evaluate import adapters, label_overlays, metrics
from rastervec.Evaluation.Evaluate.variants import resolve_variant
from rastervec.Evaluation.Labelling import auto_label
from rastervec.Evaluation.Labelling.label_schema import (
    LabelSet,
    load_labels,
    save_labels,
    split_labelset_by_source,
)
from rastervec.helpers.geometry import union_bbox
from rastervec.Evaluation.Report import stage_stats
from rastervec.logging_setup import configure_logging, get_logger
from rastervec.paths import output_dir
from rastervec.pipelines._common import STEP_NAMES
from rastervec.renderer import render_boxes_pdf, render_reconstructed_pdf, stages

_LOG = get_logger("generate_pipeline_report")

_CONVERT = {
    "text_only": conversion.convert_page_text_only,
    "drawings_only": conversion.convert_page_drawings_only,
    "to_vector_text": conversion.convert_page_to_vector_text,
}

# stage stem -> (stage_key for stages.render_stage_layers, stats-key or
# None, gate STEP_NAME). Each stage emits one single-purpose PDF per visual
# layer: `<stem>__<layer-slug>.pdf`.
_ARTIFACTS: list[tuple[str, str, str | None, str]] = [
    ("native_text", "native", "native", "native"),
    ("vector_extraction", "vectors", "vectors", "vectors"),
    ("separation", "separation", "separation", "classify"),
    ("vector_classification", "classify", "classify", "classify"),
    ("fast_heatmap", "fast", "fast", "fast"),
    ("segmentation", "segment", "segment", "segment"),
    ("similarity", "similarity", "similarity", "similarity"),
    ("paddle_ocr", "ocr", "ocr", "restore"),
    ("drawing_vectors", "drawing", None, "drawing"),
    ("reconstructed", "reconstructed", None, "drawing"),
]


def _layer_slug(label: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in label.lower())
    while "__" in keep:
        keep = keep.replace("__", "_")
    return keep.strip("_") or "layer"


class ReportConfig(BaseModel):
    pipeline: str = "current"
    final_stage: str | None = None
    input_dir: Path | None = None
    input_files: list[Path] = Field(default_factory=list)
    label_files: dict[str, Path] = Field(default_factory=dict)
    pages: list[int] | dict[str, list[int]] | None = None
    vectorise: bool = False
    vectorise_mode: str = "to_vector_text"
    benchmark: bool = False
    iou_edge_min: float = metrics.MetricConfig().iou_edge_min
    dpi: int = 300
    output_root: Path | None = None

    @field_validator("pipeline")
    @classmethod
    def _known_pipeline(cls, v: str) -> str:
        resolve_variant(v)  # raises ValueError listing valid names
        return v

    @field_validator("final_stage")
    @classmethod
    def _known_stage(cls, v: str | None) -> str | None:
        if v is not None and v not in STEP_NAMES:
            raise ValueError(f"final_stage must be one of {STEP_NAMES}")
        return v

    @field_validator("vectorise_mode")
    @classmethod
    def _known_mode(cls, v: str) -> str:
        if v not in _CONVERT:
            raise ValueError(f"vectorise_mode must be one of {list(_CONVERT)}")
        return v

    @model_validator(mode="after")
    def _benchmark_excludes_vectorise(self) -> "ReportConfig":
        if self.benchmark and self.vectorise:
            raise ValueError(
                "benchmark mode runs its own text-only / drawings-only conversions; "
                "set vectorise=false"
            )
        return self

    def resolved_pdfs(self) -> list[Path]:
        pdfs: list[Path] = list(self.input_files)
        if self.input_dir is not None:
            pdfs += sorted(Path(self.input_dir).glob("*.pdf"))
        seen, out = set(), []
        for p in pdfs:
            p = Path(p).resolve()
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def pages_for(self, stem: str) -> list[int]:
        if self.pages is None:
            return [0]
        if isinstance(self.pages, dict):
            return list(self.pages.get(stem, self.pages.get("*", [0])))
        return list(self.pages)

    def benchmark_inputs(self) -> list["BenchInput"]:
        """One `BenchInput` per config input, benchmark mode only. A `.json`
        input (or a pdf with a `label_files` entry) carries manual ground
        truth and is keyed `labels:<json-stem>`; a bare `.pdf` is auto-only
        and keyed `pdf:<pdf-stem>`. The key is what the benchmark script
        matches shared inputs on."""
        out: list[BenchInput] = []
        seen: set[str] = set()

        def _add(key: str, pdf: Path, labels: Path | None) -> None:
            if key not in seen:
                seen.add(key)
                out.append(BenchInput(key=key, pdf_path=pdf.resolve(), labels_path=labels))

        raw: list[Path] = list(self.input_files)
        if self.input_dir is not None:
            raw += sorted(Path(self.input_dir).glob("*.pdf"))
        for item in raw:
            item = Path(item)
            if item.suffix.lower() == ".json":
                labels = item.resolve()
                pdf = Path(load_labels(str(labels)).pdf_path)
                _add(f"labels:{labels.stem}", pdf, labels)
                continue
            paired = self.label_files.get(item.stem)
            if paired is not None:
                paired = Path(paired).resolve()
                _add(f"labels:{paired.stem}", item, paired)
            else:
                _add(f"pdf:{item.stem}", item, None)
        return out


class BenchInput(NamedTuple):
    key: str
    pdf_path: Path
    labels_path: Path | None


ReportConfig.model_rebuild()


def _reached(step: str, final_stage: str | None) -> bool:
    if final_stage is None:
        return True
    return STEP_NAMES.index(step) <= STEP_NAMES.index(final_stage)


def _safe_slug(text: str, limit: int = 40) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in (text or "")).strip("_")
    return keep[:limit] or "blank"


def _draw_boxes(img: Image.Image, boxes, outline=(220, 30, 30), width=2) -> Image.Image:
    out = img.convert("RGB")
    d = ImageDraw.Draw(out)
    for b in boxes:
        if b is None:
            continue
        x0, y0, x1, y1 = b
        d.rectangle(
            [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
            outline=outline, width=width,
        )
    return out


# `_common.run_current_pipeline` calls `segment_clusters` with the default
# dpi, so the report must render Radon's input at the same dpi to line the
# detected boxes up.
_RADON_DPI = 300


def _save_radon_inputs(res, folder: Path, page_index: int) -> int:
    """One PNG per FAST-surviving cluster: the rendered cluster image exactly
    as Radon sees it (pre-deskew, pre-split), with the original cluster bbox
    (green) and the final grown crop box sent to PaddleOCR (blue) drawn on
    top (`res.segmentation_debug`)."""
    clusters = getattr(res, "fast_passed", None) or []
    if not clusters:
        return 0
    from rastervec.OCR.radon import render_cluster_for_radon
    from rastervec.renderer import page_points_to_pixel

    dbg_by_bbox = {
        tuple(round(c, 2) for c in d["cluster_bbox"]): d
        for d in (res.segmentation_debug or [])
    }
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, cluster in enumerate(clusters):
        if not cluster:
            continue
        gray, dpi_used = render_cluster_for_radon(cluster, _RADON_DPI)
        img = Image.fromarray(gray)
        key = tuple(round(c, 2) for c in union_bbox([v.bbox for v in cluster]))
        dbg = dbg_by_bbox.get(key)

        def _to_px(page_bboxes):
            out = []
            for x0, y0, x1, y1 in page_bboxes:
                (px0, py0), (px1, py1) = page_points_to_pixel(
                    cluster, dpi_used, [(x0, y0), (x1, y1)]
                )
                out.append((px0, py0, px1, py1))
            return out

        if dbg is not None:
            img = _draw_boxes(img, _to_px([dbg["cluster_bbox"]]), outline=(22, 163, 74))
            img = _draw_boxes(
                img, _to_px(dbg.get("grown_segment_bboxes", [])), outline=(37, 99, 235)
            )
        img.save(folder / f"p{page_index}_cluster_{i:03d}.png")
        n += 1
    return n


def _save_paddle_inputs(res, folder: Path, page_index: int) -> int:
    """One PNG per elected unique segment -- the exact deskewed, white-padded
    crop handed to PaddleOCR verbatim, with the recognised text in the
    filename. No overlay: PaddleOCR here is recognition-only and returns no
    box."""
    segs = getattr(res, "unique_segments", None) or []
    if not segs:
        return 0
    texts = getattr(res, "unique_texts", None) or []
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, seg in enumerate(segs):
        if seg.image is None:
            continue
        img = Image.fromarray(np.asarray(seg.image))
        rec = texts[i].text if i < len(texts) else ""
        img.save(folder / f"p{page_index}_uniq_{i:03d}__{_safe_slug(rec)}.png")
        n += 1
    return n


def _merge_pdfs(page_bytes: list[bytes], out_path: Path) -> None:
    out = fitz.open()
    try:
        for b in page_bytes:
            src = fitz.open("pdf", b)
            try:
                out.insert_pdf(src)
            finally:
                src.close()
        out.save(str(out_path))
    finally:
        out.close()


def _write_hyperparams(path: Path, config: ReportConfig, variant) -> None:
    lines = ["# Run config\n", config.model_dump_json(indent=2), "\n\n# Variant\n"]
    lines.append(json.dumps(
        {"name": variant.name, "engine": variant.engine, "enable_fast": variant.enable_fast},
        indent=2,
    ))
    lines.append("\n\n# rastervec.config constants\n")
    for name in sorted(vars(rvconfig)):
        if name.isupper():
            lines.append(f"{name} = {getattr(rvconfig, name)!r}\n")
    path.write_text("".join(lines), encoding="utf-8")


def _restamp_page(res, page_index: int) -> None:
    """A run on a per-page *converted* PDF reports `page_meta.index == 0`.
    Stamp the real source page index back on so every `dump.json` PageDump /
    label overlay / benchmark score keys off the right page."""
    if res.page is not None and res.page.meta is not None:
        res.page.meta.index = page_index
        res.page.meta.number = page_index + 1


def _active_artifacts(config: ReportConfig, variant) -> list[tuple]:
    if variant.engine == "legacy":
        return [row for row in _ARTIFACTS if row[0] == "reconstructed"]
    return [row for row in _ARTIFACTS if _reached(row[3], config.final_stage)]


def _accumulate_page(
    res, page_index: int, active: list[tuple],
    layer_pages: dict[str, list[bytes]], layer_meta: dict[str, dict],
    stats_pages: dict[str, list[tuple[int, dict]]],
    radon_dir: Path, paddle_dir: Path,
) -> None:
    """Render every active stage's layer PDFs + numeric stats for one page,
    accumulating into the caller's dicts; also dump the radon / paddle
    input PNGs."""
    for stem, stage_key, stats_key, _gate in active:
        try:
            layers = stages.render_stage_layers(res, stage_key)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("%s render failed for page %d: %s", stem, page_index, exc)
            layers = []
        for label, hexc, pdf_bytes in layers:
            fname = f"{stem}__{_layer_slug(label)}.pdf"
            layer_pages.setdefault(fname, []).append(pdf_bytes)
            layer_meta.setdefault(
                fname, {"stage": stem, "layer": label, "file": fname, "color": hexc}
            )
        if stats_key is not None:
            stats_pages[stem].append(
                (page_index, stage_stats.stats_for_stage(res, stats_key))
            )
    _save_radon_inputs(res, radon_dir, page_index)
    _save_paddle_inputs(res, paddle_dir, page_index)


def _finalize_doc_dir(
    doc_dir: Path, source_pdf: Path, pages: list[int], config: ReportConfig, variant,
    active: list[tuple], layer_pages: dict[str, list[bytes]], layer_meta: dict[str, dict],
    stats_pages: dict[str, list[tuple[int, dict]]], dumps: list[dump_io.PageDump],
    *, extra_layers: tuple[dict, ...] = (),
) -> None:
    for fname, page_bytes in layer_pages.items():
        _merge_pdfs(page_bytes, doc_dir / fname)

    for stem, _sk, stats_key, _gate in active:
        if stats_key is None:
            continue
        body = "".join(
            f"\n## page {pi}\n{stage_stats.format_stats(stats_key, data)}"
            for pi, data in stats_pages[stem]
        )
        (doc_dir / f"{stem}.txt").write_text(
            f"# {stats_key} stats for {Path(source_pdf).name}\n{body}", encoding="utf-8"
        )

    dump_io.write_dump(doc_dir / "dump.json", str(source_pdf), dumps)
    (doc_dir / "manifest.json").write_text(json.dumps({
        "source_pdf": str(source_pdf),
        "pages": pages,
        "engine": variant.engine,
        "variant": variant.name,
        "final_stage": config.final_stage,
        "vectorised": config.vectorise or config.benchmark,
        "layers": [layer_meta[f] for f in layer_pages] + list(extra_layers),
    }, indent=2), encoding="utf-8")
    _LOG.info("wrote %s", doc_dir)


def _process_pdf(pdf_path: Path, config: ReportConfig, variant, run_dir: Path) -> None:
    from rastervec.pipelines.current import run_pipeline as run_current
    from rastervec.pipelines.legacy import run_pipeline as run_legacy

    doc_dir = run_dir / pdf_path.stem
    doc_dir.mkdir(parents=True, exist_ok=True)
    radon_dir = doc_dir / "radon_images"
    paddle_dir = doc_dir / "paddle_images"

    pages = config.pages_for(pdf_path.stem)
    active = _active_artifacts(config, variant)

    layer_pages: dict[str, list[bytes]] = {}
    layer_meta: dict[str, dict] = {}
    stats_pages: dict[str, list[tuple[int, dict]]] = {row[0]: [] for row in active}
    dumps: list[dump_io.PageDump] = []

    for page_index in pages:
        run_input, run_page = str(pdf_path), page_index
        if config.vectorise:
            conv_path = doc_dir / f"converted_p{page_index}.pdf"
            _CONVERT[config.vectorise_mode](str(pdf_path), page_index, str(conv_path))
            run_input, run_page = str(conv_path), 0

        _LOG.info("running %s page %d (%s)", pdf_path.name, page_index, variant.name)
        if variant.engine == "legacy":
            res = run_legacy(run_input, run_page, verbose=True)
        else:
            res = run_current(
                run_input, run_page, enable_fast=variant.enable_fast, verbose=True,
                stop_after=config.final_stage,
            )
        if run_page != page_index:
            _restamp_page(res, page_index)

        _accumulate_page(res, page_index, active, layer_pages, layer_meta, stats_pages,
                         radon_dir, paddle_dir)
        dumps.append(dump_io.PageDump(
            page_meta=res.page.meta, texts=list(res.texts or []),
            vectors=list(res.vectors or []), engine=res.engine,
            step_durations=dict(res.step_durations or {}),
        ))

    _finalize_doc_dir(doc_dir, pdf_path, pages, config, variant, active,
                      layer_pages, layer_meta, stats_pages, dumps)


_OVERLAY_COLORS = {"auto": "#22c55e", "manual": "#2563eb"}


def _bench_ground_truth(source: str, bench: BenchInput, pages: list[int]) -> LabelSet:
    """The combined `LabelSet` (all `pages`) for one label source. `auto`
    merges `auto_label_pdf` per page; `manual` reads the paired label JSON's
    `source="manual"` entries filtered to `pages`."""
    if source == "auto":
        entries = []
        for p in pages:
            entries += auto_label.auto_label_pdf(str(bench.pdf_path), p).entries
        return LabelSet(pdf_path=str(bench.pdf_path), entries=entries)
    if bench.labels_path is None:
        return LabelSet(pdf_path=str(bench.pdf_path), entries=[])
    manual = split_labelset_by_source(load_labels(str(bench.labels_path)))["manual"]
    return LabelSet(
        pdf_path=str(bench.pdf_path),
        entries=[e for e in manual.entries if e.page_index in pages],
    )


def _write_label_overlays(
    doc_dir: Path, source: str, gt: LabelSet, dumps: list[dump_io.PageDump],
    cfg: metrics.MetricConfig,
) -> None:
    """`<source>_bbox.pdf` (GT boxes green=covered by a prediction / red=missed)
    and `<source>_text.pdf` (GT text, per word green/yellow/red by read
    accuracy) -- one page per report page, scored against the single shared
    run's OCR predictions."""
    gt_regions = adapters.gt_regions_from_labelset(gt)
    bbox_pages: list[bytes] = []
    text_pages: list[bytes] = []
    for pd in dumps:
        pi = pd.page_meta.index
        regions = [g for g in gt_regions if g.page_index == pi]
        preds = adapters.predictions_from_texts(
            [t for t in pd.texts if t.source == "ocr"]
        )
        graph = metrics.build_overlap_graph(regions, preds, cfg)
        bbox_pages.append(
            render_boxes_pdf(pd.page_meta, label_overlays.gt_bbox_overlay(graph))
        )
        text_pages.append(
            render_reconstructed_pdf(
                pd.page_meta, text_boxes=label_overlays.gt_word_overlay(graph)
            )
        )
    _merge_pdfs(bbox_pages, doc_dir / f"{source}_bbox.pdf")
    _merge_pdfs(text_pages, doc_dir / f"{source}_text.pdf")


def _process_pdf_benchmark(
    bench: BenchInput, config: ReportConfig, variant, run_dir: Path,
) -> dict:
    """One benchmark input -> `run_dir/<pdf-stem>/`: the full per-stage
    report for a **single** `convert_page_to_vector_text` run per page, plus
    `ground_truth_auto.json` (+ `ground_truth_manual.json`) and the split
    `{auto,manual}_{bbox,text}.pdf` overlays (all scored against that one
    run). Returns the `benchmark.json` entry."""
    from rastervec.pipelines.current import run_pipeline as run_current
    from rastervec.pipelines.legacy import run_pipeline as run_legacy

    is_legacy = variant.engine == "legacy"
    doc_dir = run_dir / bench.pdf_path.stem
    doc_dir.mkdir(parents=True, exist_ok=True)
    radon_dir = doc_dir / "radon_images"
    paddle_dir = doc_dir / "paddle_images"
    pages = config.pages_for(bench.pdf_path.stem)
    cfg = metrics.MetricConfig(iou_edge_min=config.iou_edge_min)
    active = _active_artifacts(config, variant)

    layer_pages: dict[str, list[bytes]] = {}
    layer_meta: dict[str, dict] = {}
    stats_pages: dict[str, list[tuple[int, dict]]] = {row[0]: [] for row in active}
    dumps: list[dump_io.PageDump] = []

    for p in pages:
        conv_path = doc_dir / f"converted_p{p}.pdf"
        conversion.convert_page_to_vector_text(str(bench.pdf_path), p, str(conv_path))
        _LOG.info("benchmark %s page %d (%s)", bench.key, p, variant.name)
        if is_legacy:
            res = run_legacy(str(conv_path), 0, verbose=True)
        else:
            res = run_current(
                str(conv_path), 0, enable_fast=variant.enable_fast, verbose=True,
                stop_after=config.final_stage,
            )
        _restamp_page(res, p)
        _accumulate_page(res, p, active, layer_pages, layer_meta, stats_pages,
                         radon_dir, paddle_dir)
        dumps.append(dump_io.PageDump(
            page_meta=res.page.meta, texts=list(res.texts or []),
            vectors=list(res.vectors or []), engine=res.engine,
            step_durations=dict(res.step_durations or {}),
        ))

    sources: list[str] = []
    extra_layers: list[dict] = []
    gt_auto = _bench_ground_truth("auto", bench, pages)
    save_labels(gt_auto, str(doc_dir / "ground_truth_auto.json"))
    _write_label_overlays(doc_dir, "auto", gt_auto, dumps, cfg)
    sources.append("auto")

    gt_manual = _bench_ground_truth("manual", bench, pages)
    if gt_manual.entries:
        save_labels(gt_manual, str(doc_dir / "ground_truth_manual.json"))
        _write_label_overlays(doc_dir, "manual", gt_manual, dumps, cfg)
        sources.append("manual")
    elif bench.labels_path is not None:
        _LOG.info("%s: no manual labels for pages %s", bench.key, pages)

    for s in sources:
        for kind in ("bbox", "text"):
            extra_layers.append({
                "stage": "benchmark", "layer": f"{s} label {kind}",
                "file": f"{s}_{kind}.pdf", "color": _OVERLAY_COLORS[s],
            })

    _finalize_doc_dir(doc_dir, bench.pdf_path, pages, config, variant, active,
                      layer_pages, layer_meta, stats_pages, dumps,
                      extra_layers=tuple(extra_layers))

    (doc_dir / "benchmark_meta.json").write_text(json.dumps({
        "key": bench.key,
        "source_pdf": str(bench.pdf_path),
        "labels": str(bench.labels_path) if bench.labels_path else None,
        "pages": pages,
        "sources": sources,
        "final_stage": config.final_stage,
        "variant": variant.name,
    }, indent=2), encoding="utf-8")
    return {
        "key": bench.key,
        "pdf_stem": bench.pdf_path.stem,
        "dir": bench.pdf_path.stem,
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
    variant = resolve_variant(config.pipeline)

    root = config.output_root or output_dir("pipeline_report")
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / f"{ts}__{Path(args.config).stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_hyperparams(run_dir / "config_and_hyperparameters.txt", config, variant)

    if config.benchmark:
        inputs = config.benchmark_inputs()
        if not inputs:
            _LOG.error("no benchmark inputs (set input_dir and/or input_files)")
            return 1
        entries = [
            _process_pdf_benchmark(bench, config, variant, run_dir) for bench in inputs
        ]
        (run_dir / "benchmark.json").write_text(json.dumps({
            "benchmark": True,
            "config_stem": Path(args.config).stem,
            "variant": variant.name,
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
    for pdf_path in pdfs:
        _process_pdf(pdf_path, config, variant, run_dir)

    print(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
