"""Config-driven pipeline stage report generator.

Reads a JSON config, runs the pipeline once per (pdf, page), and writes a
timestamped run folder:

    outputs/pipeline_report/<ts>__<config-stem>/
        config_and_hyperparameters.txt        (also records the source config path)
        <pdf-stem>/
            manifest.json
            <stage>__<layer>.pdf                       (one multi-page PDF per
                                                       visual layer, e.g.
                                                       vector_classification__group_bbox.pdf;
                                                       the viewer toggles each)
            native_text.txt ...                       (one stats file per stage)
            dump.json                                 (every Text + Vector, reloadable)

            Pre-OCR debug image folders (skipped with `debug_images: false`)
            -- each `p3` backend's own distinct set (read from
            res.extra["p3_debug"], see
            `report_artifacts._accumulate_page`/each backend's own
            `parse.py`):
              p3=VectorClassification (classify+FAST filter clusters, then
              run PaddleOCR's full detect+recognize per FAST-surviving
              cluster, page-wide batched -- see that backend's own
              parse.py):
                paddle_detect_images/  (one PNG per seqno-cluster's own
                                       rendered+padded image, every detected
                                       quad drawn on top)
                paddle_recog_images/   (one PNG per detected quad's own
                                       crop, recognised text in the filename)
                hough_line_images/     (one PNG per detection's own dilated
                                       ink mask, the detected Hough line
                                       drawn on it, hough/minarea/combined
                                       angles in the filename)
                minarea_rect_images/   (one PNG per detection's own
                                       non-dilated ink mask, the fitted
                                       cv2.minAreaRect angle drawn on it,
                                       same hough/minarea/combined angles in
                                       the filename)
                paddle_classifier_before_images/  (one PNG per crop right
                                       before a recognizer call -- pass 1's
                                       pre-classifier-flip crop, or a blank-
                                       retry pass's pre-rotation crop)
                paddle_classifier_after_images/   (the matching crop right
                                       after -- what the recognizer actually
                                       saw for that pass, incl. every +90/
                                       180/270 retry attempt)
              p3=LegacyRecreation (no FAST stage):
                paddle_ocr_images/     (one PNG per word group's own padded/
                                       DPI-boosted render, recognised text
                                       in the filename)

Replaces `rastervec/notebooks/pipeline_stage_visualization.ipynb`.

With `benchmark: true`, each input gets its own report folder (named from
`BenchInput.key`, not the literal `<pdf-stem>` -- see `_bench_doc_name`; a
`scripts/label/master_label.py` folder's `pdf_path` is always that folder's
own `original.pdf`, so naming by PDF filename would collide across inputs)
that is also a scoring artifact for `pipeline_report_benchmark.py`: one
`convert_page_to_vector_text`
run per page (predictions for `native_to_vector`/`original_vector`) plus,
when the input is a master_label.py folder with its own `rasterised.pdf`, a
SEPARATE pipeline run directly on that rasterised page (predictions for
`vector_to_raster`/`original_raster`/`native_to_raster` -- these 3 types are
never scored from the vectorised run). Writes `ground_truth_<text_type>.json`
(only the non-empty of the 5 text types) and the matching
`<text_type>_{bbox,text}.pdf` overlays (registered in the manifest under
stage `benchmark`), and a run-root `benchmark.json` marker.
`input_files`/`input_dir` entries may be a `.pdf`
(auto-only), a `.json` label sidecar, or a directory (a
`scripts/label/master_label.py` output folder, merging its
`native_labels.json`/`vector_labels.json`/`raster_labels.json`). See
`docs/SCRIPTS.md`.

This module's own schema (`ReportConfig`/`BenchInput`) lives in
`report_config.py`, the per-backend debug-image dumpers in
`debug_image_savers.py`, and the artifact-writing machinery
(`_LayerWriter`/`_accumulate_page`/`_finalize_doc_dir`/`_write_label_
overlays`/...) in `report_artifacts.py` -- all re-exported here since
several names are imported directly from this module's own path.

    .venv/Scripts/python.exe scripts/generate_pipeline_report.py --config run.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import pymupdf as fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rastervec.Evaluation import conversion, dump_io
from rastervec.Evaluation.Evaluate import adapters, metrics
from rastervec.Evaluation.Evaluate.variants import PipelineVariant, resolve_variant
from rastervec.Evaluation.Labelling import native_label
from rastervec.Evaluation.Evaluate.metrics import TEXT_TYPES
from rastervec.Evaluation.Labelling.label_schema import (
    LabelEntry,
    LabelSet,
    load_labels,
    load_labels_from_master_folder,
    save_labels,
)
from rastervec.commons.logging_setup import configure_logging, get_logger
from rastervec.commons.paths import output_dir
from rastervec.commons.step_timing import StepClock

from scripts.debug_image_savers import (  # noqa: F401 -- re-exported for callers/tests
    _draw_boxes,
    _safe_slug,
    _save_crop_text_images,
    _save_legacyrecreation_ocr_images,
    _save_vectorclassification_classifier_after_images,
    _save_vectorclassification_classifier_before_images,
    _save_vectorclassification_detect_images,
    _save_vectorclassification_hough_images,
    _save_vectorclassification_minarea_images,
    _save_vectorclassification_recog_images,
)
from scripts.report_artifacts import (  # noqa: F401 -- re-exported for callers/tests
    _ARTIFACTS,
    _NEW_ARTIFACTS,
    RASTER_TEXT_TYPES,
    _accumulate_page,
    _active_artifacts,
    _add_extra_prediction_layers,
    _debug_layer_sink,
    _finalize_doc_dir,
    _image_reservoirs,
    _LayerWriter,
    _merge_pdfs,
    _reached,
    _restamp_page,
    _write_hyperparams,
    _write_label_overlays,
)
from scripts.report_config import (  # noqa: F401 -- re-exported for callers/tests
    NEW_STEP_NAMES,
    BenchInput,
    ReportConfig,
    _CONVERT,
    _layer_slug,
)

_LOG = get_logger("generate_pipeline_report")


def _filter_valid_pages(pdf_path: Path, pages: list[int], label: str) -> list[int]:
    """Drop any page index out of range for `pdf_path`, warning once per
    input rather than letting a downstream `IndexError: page N not in
    document` abort the whole run -- a config's `pages` list is often shared
    across several input files (`pages_for`'s `"*"` fallback, or an explicit
    per-stem list applied too broadly) that don't all have the same page
    count."""
    doc = fitz.open(str(pdf_path))
    try:
        page_count = doc.page_count
    finally:
        doc.close()
    valid = [p for p in pages if 0 <= p < page_count]
    invalid = [p for p in pages if p not in valid]
    if invalid:
        _LOG.warning(
            "%s: skipping out-of-range page(s) %s (%s has %d page%s)",
            label, invalid, pdf_path.name, page_count, "" if page_count == 1 else "s",
        )
    return valid


def _bench_doc_name(bench: "BenchInput") -> str:
    """Filesystem-safe, per-input-unique folder name for a benchmark input's
    `<run_dir>/<name>/` report folder. Can't use `bench.pdf_path.stem`
    directly -- a `scripts/label/master_label.py` folder's `pdf_path` is
    always that folder's own `original.pdf` (`label_schema.
    load_labels_from_master_folder`), so every master-label input in one
    report run would otherwise share the literal "original" stem and
    silently overwrite each other's output. `bench.key` (`pdf:<stem>` /
    `labels:<stem>`) is already unique per `ReportConfig.benchmark_inputs()`
    entry, so strip its source-type prefix instead."""
    name = bench.key.split(":", 1)[-1]
    return "".join(c if c not in '<>:"/\\|?*' else "_" for c in name) or "input"


def _image_dirs(doc_dir: Path) -> "dict[str, Path]":
    """Every possible debug-image folder for one document, keyed by the
    short name `report_artifacts._accumulate_page`'s per-`p3` dispatch
    uses. Not every backend populates every folder (a folder no saver ever
    calls `.offer()` on is simply never created) -- see that dispatch for
    which backend writes which."""
    return {
        "detect": doc_dir / "paddle_detect_images",
        "recog": doc_dir / "paddle_recog_images",
        "ocr": doc_dir / "paddle_ocr_images",
        "hough": doc_dir / "hough_line_images",
        "minarea": doc_dir / "minarea_rect_images",
        "classifier_before": doc_dir / "paddle_classifier_before_images",
        "classifier_after": doc_dir / "paddle_classifier_after_images",
    }


def _process_pdf(pdf_path: Path, config: ReportConfig, variant, run_dir: Path) -> None:
    from rastervec.core.pipeline import run_pipeline as run_current
    from rastervec.pipelines.legacy import run_pipeline as run_legacy

    is_legacy = variant.engine == "legacy"
    doc_dir = run_dir / pdf_path.stem
    doc_dir.mkdir(parents=True, exist_ok=True)
    reservoirs = (
        _image_reservoirs(_image_dirs(doc_dir), seed_name=doc_dir.name)
        if config.debug_images else None
    )

    pages = _filter_valid_pages(pdf_path, config.pages_for(pdf_path.stem), pdf_path.stem)
    active = _active_artifacts(config, variant)

    writer = _LayerWriter()
    stats_pages: dict[str, list[tuple[int, dict]]] = {row[0]: [] for row in active}
    dumps: list[dump_io.PageDump] = []

    for page_index in pages:
        debug_durations: dict = {}
        clock = StepClock(debug_durations)
        run_input, run_page = str(pdf_path), page_index
        if config.vectorise:
            conv_path = doc_dir / f"converted_p{page_index}.pdf"
            with clock("conversion"):
                _CONVERT[config.vectorise_mode](str(pdf_path), page_index, str(conv_path))
            run_input, run_page = str(conv_path), 0

        _LOG.info("running %s page %d (%s)", pdf_path.name, page_index, variant.name)
        if is_legacy:
            res = run_legacy(run_input, run_page, verbose=True)
        else:
            res = run_current(
                run_input, run_page, p2=variant.p2, p3=variant.p3,
                enable_fast=variant.enable_fast, verbose=True,
                on_debug_layer=_debug_layer_sink(writer),
            )
        if run_page != page_index:
            _restamp_page(res, page_index)

        _accumulate_page(res, page_index, active, writer, stats_pages, reservoirs,
                         is_legacy=is_legacy, p3=variant.p3, clock=clock)
        dumps.append(dump_io.PageDump(
            page_meta=res.page.meta, texts=list(res.texts or []),
            vectors=list(res.vectors or []), engine=variant.engine,
            step_durations=dict(res.step_durations or {}),
            substep_durations=_substeps(res),
            debug_durations=debug_durations,
            fast_cluster_stats=_fast_cluster_stats(res, variant.p3),
            retry_stats=_retry_stats(res, variant.p3),
        ))

    _finalize_doc_dir(doc_dir, pdf_path, pages, config, variant, active,
                      writer, stats_pages, dumps)


_OVERLAY_COLORS = {
    "native_to_vector": "#22c55e",
    "original_vector": "#2563eb",
    "vector_to_raster": "#f97316",
    "original_raster": "#a855f7",
    "native_to_raster": "#eab308",
}


def _bench_ground_truth_by_type(bench: BenchInput, pages: list[int]) -> "dict[str, LabelSet]":
    """One `LabelSet` per text type (all `TEXT_TYPES` keys always present,
    entries + geometry_entries filtered to `pages`).

    `native_to_vector` is live-derived via `native_label.native_label_pdf`
    per page -- UNLESS `bench.labels_path` is a master_label.py folder that
    already has its own `native_labels.json`, in which case that file's
    entries are used instead (avoids redundant re-derivation).

    `original_vector`/`vector_to_raster`/`original_raster`/`native_to_raster`
    come from `bench.labels_path`: `None` -> all empty; a `.json` file ->
    `load_labels` + `entries_by_text_type` (single-file convention); a
    directory -> `load_labels_from_master_folder` + `entries_by_text_type`
    (+ geometry_entries, needed for vector_to_raster/original_raster
    vector-geometry scoring).

    `native_to_raster` ground truth is the SAME text as `native_to_vector`
    (the same native region, just scored against a rasterised-PDF pipeline
    run instead of the vectorised one) -- taken from the label folder's own
    `"natsync:"`-prefixed raster_labels.json entries
    (`raster_label.sync_native_text_from_native_labels`) when present, else
    synthesized on the fly from `native_entries` (a label folder predating
    that sync step).
    """
    pdf_str = str(bench.pdf_path)
    by_type: "dict[str, LabelSet]" = {t: LabelSet(pdf_path=pdf_str, entries=[]) for t in TEXT_TYPES}

    native_from_folder = (
        bench.labels_path is not None
        and bench.labels_path.is_dir()
        and (bench.labels_path / "native_labels.json").is_file()
    )
    if native_from_folder:
        merged = load_labels_from_master_folder(bench.labels_path)
        native_entries = adapters.entries_by_text_type(merged)["native_to_vector"]
    else:
        native_entries = []
        for p in pages:
            native_entries += native_label.native_label_pdf(str(bench.pdf_path), p).entries
    native_entries = [e for e in native_entries if e.page_index in pages]
    by_type["native_to_vector"] = LabelSet(pdf_path=pdf_str, entries=native_entries)

    if bench.labels_path is not None:
        if bench.labels_path.is_dir():
            merged = load_labels_from_master_folder(bench.labels_path)
        else:
            merged = load_labels(str(bench.labels_path))
        buckets = adapters.entries_by_text_type(merged)
        # geometry_entries (vector-geometry GT, distinct from these text
        # `entries`) only ever belong on vector_to_raster ("auto"-source)
        # and original_raster ("manual"-source) -- split the same way
        # `adapters.gt_geometry_by_vector_type` does, and only there, so
        # `_merge_gt` (pipeline_report_benchmark.py) doesn't double them up
        # by finding them duplicated across several ground_truth_*.json.
        geometry_by_source = {
            "vector_to_raster": "auto", "original_raster": "manual",
        }
        for t in ("original_vector", "vector_to_raster", "original_raster", "native_to_raster"):
            geom_source = geometry_by_source.get(t)
            by_type[t] = LabelSet(
                pdf_path=pdf_str,
                entries=[e for e in buckets[t] if e.page_index in pages],
                geometry_entries=(
                    [g for g in merged.geometry_entries
                     if g.page_index in pages and g.source == geom_source]
                    if geom_source else []
                ),
            )

    if not by_type["native_to_raster"].entries:
        by_type["native_to_raster"] = LabelSet(
            pdf_path=pdf_str,
            entries=[
                LabelEntry(
                    page_index=e.page_index, cluster_bbox=e.cluster_bbox,
                    cluster_signature=e.cluster_signature,
                    label_id=f"natsync:{e.label_id}", text=e.text, source="raster",
                    expected_rotation=e.expected_rotation,
                )
                for e in native_entries
            ],
        )

    return by_type


def _substeps(res) -> dict:
    """The P3 backend's own sub-step seconds (`core.result.PipelineResult.
    substep_durations`); `{}` for the legacy engine's older result shape,
    which has no such field."""
    return dict(getattr(res, "substep_durations", None) or {})


def _fast_cluster_stats(res, p3: str) -> "dict | None":
    """`{"total": N, "passed": P}` cluster counts off the VectorClassification
    P3 backend's own `FastPageResult` (`res.extra["p3_debug"]["fast_result"]`,
    only present on a `verbose=True` run) -- `None` for any other P3 backend
    (no such concept) or a run without debug data."""
    if p3 != "VectorClassification":
        return None
    p3_debug = (getattr(res, "extra", None) or {}).get("p3_debug") or {}
    fast_result = p3_debug.get("fast_result")
    if fast_result is None:
        return None
    return {"total": fast_result.n_clusters, "passed": fast_result.n_passed_clusters}


def _retry_stats(res, p3: str) -> "dict | None":
    """`{"0": N, "1": N, "2": N, "3": N, "failed": N}` blank-recognition
    retry counts off the VectorClassification P3 backend's own
    `debug_out["retry_stats"]` (`res.extra["p3_debug"]["retry_stats"]`, only
    present on a `verbose=True` run) -- `None` for any other P3 backend (no
    such concept) or a run without debug data. Mirrors `_fast_cluster_stats`
    exactly."""
    if p3 != "VectorClassification":
        return None
    p3_debug = (getattr(res, "extra", None) or {}).get("p3_debug") or {}
    return p3_debug.get("retry_stats")


def _extract_single_page(src_pdf: Path, page_index: int, out_path: Path) -> None:
    doc = fitz.open(str(src_pdf))
    try:
        out = fitz.open()
        try:
            out.insert_pdf(doc, from_page=page_index, to_page=page_index)
            out.save(str(out_path))
        finally:
            out.close()
    finally:
        doc.close()


def _process_pdf_benchmark(
    bench: BenchInput, config: ReportConfig, variant, run_dir: Path,
) -> dict:
    """One benchmark input -> `run_dir/<name>/` (named from `bench.key`, see
    `_bench_doc_name`): the full per-stage report for a
    `convert_page_to_vector_text` run per page (predictions for
    `native_to_vector`/`original_vector`), plus -- when `bench.rasterised_
    pdf_path` is set -- a SEPARATE pipeline run directly on that rasterised
    PDF's own page (predictions for `vector_to_raster`/`original_raster`/
    `native_to_raster`; that page has no vector paths at all, so this run
    currently always yields empty predictions -- see `PageDump.raster_
    texts`'s docstring). Writes `ground_truth_<type>.json` (only the
    non-empty types) and `<type>_{bbox,text}.pdf` overlays for whichever
    types have labels. Returns the `benchmark.json` entry."""
    from rastervec.core.pipeline import run_pipeline as run_current
    from rastervec.pipelines.legacy import run_pipeline as run_legacy

    is_legacy = variant.engine == "legacy"
    doc_name = _bench_doc_name(bench)
    doc_dir = run_dir / doc_name
    doc_dir.mkdir(parents=True, exist_ok=True)
    reservoirs = (
        _image_reservoirs(_image_dirs(doc_dir), seed_name=doc_name)
        if config.debug_images else None
    )
    pages = _filter_valid_pages(bench.pdf_path, config.pages_for(doc_name), bench.key)
    cfg = metrics.MetricConfig(iou_edge_min=config.iou_edge_min)
    active = _active_artifacts(config, variant)
    gt_by_type = _bench_ground_truth_by_type(bench, pages)
    # Extra-prediction layers only when this input has manual vector labels.
    has_manual = bool(gt_by_type["original_vector"].entries)

    writer = _LayerWriter()
    stats_pages: dict[str, list[tuple[int, dict]]] = {row[0]: [] for row in active}
    dumps: list[dump_io.PageDump] = []

    for p in pages:
        debug_durations: dict = {}
        clock = StepClock(debug_durations)
        conv_path = doc_dir / f"converted_p{p}.pdf"
        with clock("conversion"):
            conversion.convert_page_to_vector_text(str(bench.pdf_path), p, str(conv_path))
        _LOG.info("benchmark %s page %d (%s)", bench.key, p, variant.name)
        if is_legacy:
            res = run_legacy(str(conv_path), 0, verbose=True)
        else:
            res = run_current(
                str(conv_path), 0, p2=variant.p2, p3=variant.p3,
                enable_fast=variant.enable_fast, verbose=True,
                on_debug_layer=_debug_layer_sink(writer),
            )
        _restamp_page(res, p)
        _accumulate_page(res, p, active, writer, stats_pages, reservoirs,
                         is_legacy=is_legacy, p3=variant.p3, clock=clock)
        if has_manual:
            with clock("extra_predictions"):
                _add_extra_prediction_layers(
                    writer, res, res.page.meta,
                    gt_boxes=[
                        tuple(e.cluster_bbox)
                        for t in ("native_to_vector", "original_vector")
                        for e in gt_by_type[t].entries if e.page_index == p
                    ],
                    manual_boxes=[
                        tuple(e.cluster_bbox)
                        for e in gt_by_type["original_vector"].entries if e.page_index == p
                    ],
                )

        raster_texts = []
        raster_steps: dict = {}
        raster_substeps: dict = {}
        if bench.rasterised_pdf_path is not None:
            raster_page_path = doc_dir / f"rasterised_p{p}.pdf"
            try:
                with clock("conversion"):
                    _extract_single_page(bench.rasterised_pdf_path, p, raster_page_path)
                if is_legacy:
                    res_raster = run_legacy(str(raster_page_path), 0, verbose=True)
                else:
                    res_raster = run_current(
                        str(raster_page_path), 0, p2=variant.p2, p3=variant.p3,
                        enable_fast=variant.enable_fast, verbose=True,
                    )
                _restamp_page(res_raster, p)
                raster_texts = list(res_raster.texts or [])
                raster_steps = dict(res_raster.step_durations or {})
                raster_substeps = _substeps(res_raster)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning(
                    "%s: rasterised-PDF run failed for page %d: %s", bench.key, p, exc,
                )

        dumps.append(dump_io.PageDump(
            page_meta=res.page.meta, texts=list(res.texts or []),
            vectors=list(res.vectors or []), engine=variant.engine,
            step_durations=dict(res.step_durations or {}),
            raster_texts=raster_texts,
            substep_durations=_substeps(res),
            raster_step_durations=raster_steps,
            raster_substep_durations=raster_substeps,
            debug_durations=debug_durations,
            fast_cluster_stats=_fast_cluster_stats(res, variant.p3),
            retry_stats=_retry_stats(res, variant.p3),
        ))

    sources: list[str] = []
    extra_layers: list[dict] = []
    doc_durations: dict = {}
    for text_type in TEXT_TYPES:
        gt = gt_by_type[text_type]
        if not gt.entries and not gt.geometry_entries:
            if text_type != "native_to_vector":
                _LOG.info("%s: no %s labels for pages %s", bench.key, text_type, pages)
            continue
        save_labels(gt, str(doc_dir / f"ground_truth_{text_type}.json"))
        if gt.entries:
            with StepClock(doc_durations)("label_overlays"):
                _write_label_overlays(doc_dir, text_type, gt, dumps, cfg)
        sources.append(text_type)

    for s in sources:
        for kind in ("bbox", "text"):
            extra_layers.append({
                "stage": "benchmark", "layer": f"{s} label {kind}",
                "file": f"{s}_{kind}.pdf", "color": _OVERLAY_COLORS[s],
            })

    _finalize_doc_dir(doc_dir, bench.pdf_path, pages, config, variant, active,
                      writer, stats_pages, dumps,
                      extra_layers=tuple(extra_layers), doc_durations=doc_durations)

    (doc_dir / "benchmark_meta.json").write_text(json.dumps({
        "key": bench.key,
        "dir": doc_name,
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
    # `config.pipeline` picks the engine; `config.p2`/`config.p3` (only
    # meaningful for "current") come straight from the run config, not from
    # a named `variants.VARIANTS` preset -- a report run wants exactly the
    # backend combo the config asks for, not a fixed default combo.
    if config.pipeline == "legacy":
        variant = resolve_variant("legacy")
    else:
        variant = PipelineVariant(
            name=f"current:{config.p2}:{config.p3}", engine="current",
            p2=config.p2, p3=config.p3, enable_fast=config.enable_fast,
        )

    root = config.output_root or output_dir("pipeline_report")
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / f"{ts}__{Path(args.config).stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_hyperparams(run_dir / "config_and_hyperparameters.txt", config, variant, Path(args.config))

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
