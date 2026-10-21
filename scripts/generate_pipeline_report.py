"""Config-driven pipeline stage report generator.

Reads a JSON config, runs the pipeline once per (pdf, page, input form), and
writes a timestamped run folder. Input forms (`inputs`, any of
"original" / "vectorised" / "rasterised", see `rastervec/Evaluation/
forms.py`): the page as-is, its `convert_page_to_vector_text` conversion,
or its `conversion.rasterise_page` flattening at `dpi`. One report folder
per form -- `<pdf-stem>/` when a single form runs, else
`<pdf-stem>__<form>/` each:

    outputs/pipeline_report/<ts>__<config-stem>/
        config_and_hyperparameters.txt        (also records the source config path)
        <pdf-stem>[__<form>]/
            manifest.json
            <stage>__<layer>.pdf                       (one multi-page PDF per
                                                       visual layer, e.g.
                                                       vector_classification__group_bbox.pdf;
                                                       the viewer toggles each)
            native_text.txt ...                       (one stats file per stage)
            dump.json                                 (every Text + Vector, reloadable)

            Every layer PDF above (and the benchmark `<text_type>_{bbox,text}.pdf`
            overlays below) is skipped with `debug_layers: false`, along
            with the work done only to render it -- dump.json, ground truth
            and debug images are still written, so the benchmark still
            scores the run.

            `converted_p<N>.pdf` / `rasterised_p<N>.pdf` (the vectorised /
            rasterised forms' per-page pipeline inputs) are deleted after
            each page's run with `keep_converted_pdfs: false`.

            Pre-OCR debug image folders (skipped with `debug_images: false`,
            each leaf folder capped at `debug_image_cap` images -- default
            100, `None` uncapped) -- each `p3` backend's own distinct set
            (read from res.extra["p3_debug"], see
            `report_artifacts._accumulate_page`/each backend's own
            `parse.py`). Every saved image is exactly (channel order aside,
            never an overlay) the array that model/algorithm call actually
            received:
              p3=LatestVectorClassification (classify clusters, then run
              PaddleOCR's detect, quad-angle upright crop, classifier +
              recognize per cluster, page-wide batched -- see that backend's
              own parse.py), organized by which call each image was the
              input to:
                for_paddle_detect/     (one PNG per cluster's own rendered
                                       image -- exactly what
                                       PaddleDetectBackend.detect saw)
                for_rotation_correction/
                  quad_rotation/       (one PNG per detection -- exactly the
                                       region around the quad that the
                                       quad-angle rotation ran on)
                  paddle_classifier/   (one PNG per detection -- exactly the
                                       upright crop handed to the
                                       recognizer's own internal
                                       text_classifier call, pass 1
                                       pre-flip)
                for_paddle_recog/
                  0_retry/ 1_retry/ 2_retry/ 3_retry/ failed/
                                       (one PNG per detection, bucketed by
                                       which recognize_crops/
                                       recognize_crops_raw pass decided its
                                       outcome -- "failed" holds the last
                                       rotation variant tried before giving
                                       up; recognised text in the filename
                                       where available)
              p3=OldVectorClassification (frozen 2026-09-29 snapshot; the
              savers and folder names that report used on that date, which
              draw quads/angle lines on top):
                paddle_detect_images/ paddle_recog_images/
                hough_line_images/ minarea_rect_images/
                paddle_classifier_before_images/
                paddle_classifier_after_images/
              every p2 backend except Stub (the shared raster prelude,
              streamed via `on_debug_image`, an annotated chart rather than
              a model input):
                color_layer_histograms/  (one PNG per embedded image: L
                                         histogram + Otsu threshold, hue
                                         histogram + peaks + layer arcs)

Replaces `rastervec/notebooks/pipeline_stage_visualization.ipynb`.

With `benchmark: true`, each input gets its own folder (named from
`BenchInput.key`, not the literal `<pdf-stem>` -- see `_bench_doc_name`; a
`scripts/label/master_label.py` folder's `pdf_path` is always that folder's
own `original.pdf`, so naming by PDF filename would collide across inputs)
that is also a scoring artifact for `pipeline_report_benchmark.py`:
`ground_truth_<text_type>.json` (only the non-empty of the 5 text types) +
`benchmark_meta.json`, and one sub-folder per input form (default
`["vectorised", "rasterised"]`), each a complete viewer folder of that
form's run with its own `dump.json`, the `<text_type>_{bbox,text}.pdf`
overlays of the types scored on that form (stage `benchmark`) and the
`vector_eval` layers of its Phase 2 vectors (`rastervec/Evaluation/
forms.py` says what each form is scored on). The rasterised form uses a
master_label.py folder's own `rasterised.pdf` (the file the raster labels
were drawn on), else flattens each page on the fly. A form run that raises
is logged and recorded as a failed page (`counts["run_failed"]`), not
fatal. Plus a run-root `benchmark.json` marker (`entries[].forms`).
`input_files`/`input_dir` entries may be a `.pdf`
(auto-only), a `.json` label sidecar, or a directory (a
`scripts/label/master_label.py` output folder, merging its
`native_labels.json`/`vector_labels.json`/`raster_labels.json`). See
`docs/SCRIPTS.md`.

This module's own schema (`ReportConfig`/`BenchInput`) lives in
`rastervec/Evaluation/Report/report_config.py`, the per-backend
debug-image dumpers in `rastervec/Evaluation/Report/debug_image_savers.py`,
and the artifact-writing machinery (`_LayerWriter`/`_accumulate_page`/
`_finalize_doc_dir`/`_write_label_overlays`/...) in
`rastervec/Evaluation/Report/report_artifacts.py` -- all re-exported here since
several names are imported directly from this module's own path.

    .venv/Scripts/python.exe scripts/generate_pipeline_report.py --config run.json
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import gc
import json
import sys
from pathlib import Path

import pymupdf as fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rastervec.Evaluation import conversion, dump_io
from rastervec.Evaluation.Evaluate import adapters, metrics, vector_metrics
from rastervec.Evaluation.Evaluate.variants import PipelineVariant, resolve_variant
from rastervec.Evaluation.Labelling import native_label
from rastervec.Evaluation.Evaluate.metrics import TEXT_TYPES
from rastervec.Evaluation.forms import (
    FORM_TEXT_TYPES,
    FORM_VECTOR_GT,
    all_texts_are_predictions,
    type_label,
)
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

from rastervec.Evaluation.Report.debug_image_savers import (  # noqa: F401 -- re-exported for callers/tests
    _draw_boxes,
    _safe_slug,
    _save_crop_text_images,
    _save_latestvectorclassification_detect_images,
    _save_latestvectorclassification_quad_rotation_images,
    _save_latestvectorclassification_recog_bucket_images,
    _save_latestvectorclassification_rotation_classifier_images,
    _save_oldvectorclassification_classifier_after_images,
    _save_oldvectorclassification_classifier_before_images,
    _save_oldvectorclassification_detect_images,
    _save_oldvectorclassification_hough_images,
    _save_oldvectorclassification_minarea_images,
    _save_oldvectorclassification_recog_images,
)
from rastervec.Evaluation.Report.report_artifacts import (  # noqa: F401 -- re-exported for callers/tests
    _ARTIFACTS,
    _NEW_ARTIFACTS,
    _accumulate_page,
    _active_artifacts,
    _add_extra_prediction_layers,
    _add_vector_eval_layers,
    _debug_image_sink,
    _debug_layer_sink,
    _finalize_doc_dir,
    _image_reservoirs,
    _LayerWriter,
    _merge_pdfs,
    _page_geometry,
    _reached,
    _restamp_page,
    _write_hyperparams,
    _write_label_overlays,
)
from rastervec.Evaluation.Report.report_config import (  # noqa: F401 -- re-exported for callers/tests
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
    which backend writes which. LatestVectorClassification's own folders are
    organized by which model/algorithm call the saved image was the actual
    input to (see `rastervec/Evaluation/Report/debug_image_savers.py`'s module docstring);
    `for_rotation_correction`/`for_paddle_recog` fan out into subfolders,
    which a plain `_ImageReservoir(path, ...)` handles the same as a
    top-level folder (`Path.mkdir(parents=True, ...)`)."""
    return {
        "detect": doc_dir / "for_paddle_detect",
        "rotation_quad": doc_dir / "for_rotation_correction" / "quad_rotation",
        "rotation_classifier": doc_dir / "for_rotation_correction" / "paddle_classifier",
        "recog_0": doc_dir / "for_paddle_recog" / "0_retry",
        "recog_1": doc_dir / "for_paddle_recog" / "1_retry",
        "recog_2": doc_dir / "for_paddle_recog" / "2_retry",
        "recog_3": doc_dir / "for_paddle_recog" / "3_retry",
        "recog_failed": doc_dir / "for_paddle_recog" / "failed",
        # OldVectorClassification's 2026-09-29 folders.
        "old_detect": doc_dir / "paddle_detect_images",
        "old_recog": doc_dir / "paddle_recog_images",
        "old_hough": doc_dir / "hough_line_images",
        "old_minarea": doc_dir / "minarea_rect_images",
        "old_classifier_before": doc_dir / "paddle_classifier_before_images",
        "old_classifier_after": doc_dir / "paddle_classifier_after_images",
        # Every P2 backend's shared raster prelude streams one hue/lightness
        # histogram per embedded image here via `on_debug_image`
        # (`RasterText.raster_text.HISTOGRAM_FOLDER`).
        "color_layer_histograms": doc_dir / "color_layer_histograms",
    }


def _form_input(
    form: str, pdf_path: Path, page_index: int, out_dir: Path, config: ReportConfig,
    rasterised_pdf: "Path | None" = None,
) -> "tuple[str, int, Path | None]":
    """`(pipeline input path, page index in it, per-page file written)` of one
    input form (`Evaluation/forms.py`). "vectorised" converts the page
    (`ReportConfig.vectorise_fn`); "rasterised" takes the page from a label
    folder's own `rasterised.pdf` when given (the file the raster labels
    were drawn on), else flattens it at `config.dpi`."""
    if form == "original":
        return str(pdf_path), page_index, None
    if form == "vectorised":
        path = out_dir / f"converted_p{page_index}.pdf"
        config.vectorise_fn()(str(pdf_path), page_index, str(path))
    else:
        path = out_dir / f"rasterised_p{page_index}.pdf"
        if rasterised_pdf is not None:
            _extract_single_page(rasterised_pdf, page_index, path)
        else:
            conversion.rasterise_page(str(pdf_path), page_index, str(path), dpi=config.dpi)
    return str(path), 0, path


def _run_form_page(
    run_input: str, run_page: int, page_index: int, config: ReportConfig, variant,
    writer: "_LayerWriter", reservoirs,
):
    """One verbose pipeline run (debug layers / images streamed into this
    form's writer / reservoirs), restamped to the source page index."""
    from rastervec.core.pipeline import run_pipeline as run_current
    from rastervec.pipelines.legacy import run_pipeline as run_legacy

    if variant.engine == "legacy":
        res = run_legacy(run_input, run_page, verbose=True)
    else:
        res = run_current(
            run_input, run_page, p2=variant.p2, p3=variant.p3,
            enable_fast=variant.enable_fast, verbose=True,
            on_debug_layer=_debug_layer_sink(writer) if config.debug_layers else None,
            on_debug_image=(
                _debug_image_sink(reservoirs, page_index) if reservoirs is not None else None
            ),
            keep_debug_arrays=config.debug_images,
            stop_after=config.final_stage,
        )
    if run_page != page_index:
        _restamp_page(res, page_index)
    return res


def _hex(color) -> str:
    return "#" + "".join(f"{int(round(float(c) * 255)):02x}" for c in (color or (0.0, 0.0, 0.0))[:3])


def _run_counts(res, p3: str) -> dict:
    """Per-page counts of one run (reported per form by the benchmark): text
    by source, vectors per phase, OCR blanks and P2's per-colour-layer
    output. Phase 1/2 detail comes off a verbose run's `extra`; a key whose
    source is missing (legacy engine, a phase `final_stage` skipped) is
    simply absent."""
    texts = list(getattr(res, "texts", None) or [])
    vectors = list(getattr(res, "vectors", None) or [])
    extra = getattr(res, "extra", None) or {}
    out: dict = {"run_failed": 0, "texts": len(texts), "vectors": len(vectors)}
    phase1 = extra.get("phase1")
    p2_texts = extra.get("phase2_texts")
    p2_vectors = extra.get("phase2_vectors")
    if phase1 is not None:
        out["images"] = len(phase1.images or [])
        out["native_texts"] = len(phase1.texts or [])
        out["p1_vectors"] = len(phase1.vectors or [])
    if p2_texts is not None:
        out["p2_ocr_texts"] = sum(1 for t in p2_texts if t.source == "ocr")
        out["p3_ocr_texts"] = sum(1 for t in texts if t.source == "ocr") - out["p2_ocr_texts"]
    if p2_vectors is not None:
        out["p2_vectors"] = len(p2_vectors)
        out["p2_items"] = sum(len(v.items or []) for v in p2_vectors)
        layers: dict = {}
        for v in p2_vectors:
            layer = layers.setdefault(_hex(v.color), {"vectors": 0, "items": 0})
            layer["vectors"] += 1
            layer["items"] += len(v.items or [])
        out["p2_layers"] = layers
        if "phase3" in (getattr(res, "step_durations", None) or {}):
            out["p3_drawing_vectors"] = len(vectors) - len(p2_vectors)
            if "p1_vectors" in out:
                out["p3_consumed_as_text"] = out["p1_vectors"] - out["p3_drawing_vectors"]
    prelude = (extra.get("p2_debug") or {}).get("raster_prelude")
    if prelude is not None:
        out["p2_ocr_blank"] = len(prelude.debug.boxes.get("refined_failed", []))
    retry = _retry_stats(res, p3)
    if retry is not None:
        out["p3_ocr_blank"] = int(retry.get("failed", 0))
    return out


def _failed_page_dump(form: str, run_input: str, run_page: int, page_index: int, variant, exc) -> dump_io.PageDump:
    """A placeholder page for a run that raised: empty output, counted as a
    failed page (`counts["run_failed"]`) so the benchmark shows it rather
    than silently scoring an empty page as a perfect miss."""
    from rastervec.P1_Reading_Native.reader import Reader

    meta = Reader(run_input).get_page(run_page).meta
    meta.index, meta.number = page_index, page_index + 1
    return dump_io.PageDump(
        page_meta=meta, texts=[], vectors=[], engine=variant.engine, step_durations={},
        counts={"run_failed": 1, "error": f"{type(exc).__name__}: {exc}"},
    )


def _form_dir_name(stem: str, form: str, forms: "list[str]") -> str:
    """Normal mode: `<stem>/` for a single form, `<stem>__<form>/` each
    when several forms run."""
    return stem if len(forms) == 1 else f"{stem}__{form}"


def _process_pdf(pdf_path: Path, config: ReportConfig, variant, run_dir: Path) -> None:
    forms = config.forms()
    for form in forms:
        _process_pdf_form(pdf_path, form, run_dir / _form_dir_name(pdf_path.stem, form, forms), config, variant)


def _process_pdf_form(pdf_path: Path, form: str, doc_dir: Path, config: ReportConfig, variant) -> None:
    """One normal-mode report folder: `pdf_path`'s pages run as `form`."""
    is_legacy = variant.engine == "legacy"
    doc_dir.mkdir(parents=True, exist_ok=True)
    reservoirs = (
        _image_reservoirs(_image_dirs(doc_dir), seed_name=doc_dir.name, cap=config.debug_image_cap)
        if config.debug_images else None
    )

    pages = _filter_valid_pages(pdf_path, config.pages_for(pdf_path.stem), pdf_path.stem)
    active = _active_artifacts(config, variant)

    writer = _LayerWriter(doc_dir / ".layer_parts")
    stats_pages: dict[str, list[tuple[int, dict]]] = {row[0]: [] for row in active}
    # Pages go to disk as they finish (not one list of every page's vectors).
    dump = dump_io.DumpWriter(doc_dir / "dump.json", str(pdf_path), form=form)

    for pos, page_index in enumerate(pages):
        debug_durations: dict = {}
        clock = StepClock(debug_durations)
        with clock("conversion"):
            run_input, run_page, temp = _form_input(form, pdf_path, page_index, doc_dir, config)
        writer.begin_page(pos, *_page_geometry(run_input, run_page))

        _LOG.info("running %s page %d as %s (%s)", pdf_path.name, page_index, form, variant.name)
        res = _run_form_page(run_input, run_page, page_index, config, variant, writer, reservoirs)
        _accumulate_page(res, page_index, active, writer, stats_pages, reservoirs,
                         is_legacy=is_legacy, p3=variant.p3, clock=clock,
                         render_layers=config.debug_layers)
        dump.add(dump_io.PageDump(
            page_meta=res.page.meta, texts=list(res.texts or []),
            vectors=list(res.vectors or []), engine=variant.engine,
            step_durations=dict(res.step_durations or {}),
            substep_durations=_substeps(res),
            debug_durations=debug_durations,
            fast_cluster_stats=_fast_cluster_stats(res, variant.p3),
            retry_stats=_retry_stats(res, variant.p3),
            raster_dpi=adapters.raster_dpi_for_page(run_input, run_page) if form != "vectorised" else None,
            p2_vectors=_p2_vectors(res) if form == "original" else None,
            counts=_run_counts(res, variant.p3),
        ))
        if temp is not None and not config.keep_converted_pdfs:
            temp.unlink(missing_ok=True)
        # Drop this page's verbose result (Phase-1 images, debug_out, the
        # classification) before the next page runs -- otherwise two pages'
        # results are alive at once.
        del res
        gc.collect()

    _finalize_doc_dir(doc_dir, pdf_path, pages, config, variant, active,
                      writer, stats_pages, dump)


def _p2_vectors(res) -> "list | None":
    """Phase 2's own output vectors off a verbose run (`None` without)."""
    vs = (getattr(res, "extra", None) or {}).get("phase2_vectors")
    return None if vs is None else list(vs)


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
    """`{"total": N, "passed": P}` cluster counts off the
    OldVectorClassification P3 backend's own pre-OCR `FastPageResult`
    (`res.extra["p3_debug"]["fast_result"]`, only present on a
    `verbose=True` run) -- `None` for any other P3 backend (no such concept)
    or a run without debug data."""
    if p3 != "OldVectorClassification":
        return None
    p3_debug = (getattr(res, "extra", None) or {}).get("p3_debug") or {}
    fast_result = p3_debug.get("fast_result")
    if fast_result is None:
        return None
    return {"total": fast_result.n_clusters, "passed": fast_result.n_passed_clusters}


def _retry_stats(res, p3: str) -> "dict | None":
    """`{"0": N, "1": N, "2": N, "3": N, "failed": N}` blank-recognition
    retry counts off the Latest/OldVectorClassification P3 backend's own
    `debug_out["retry_stats"]` (`res.extra["p3_debug"]["retry_stats"]`, only
    present on a `verbose=True` run) -- `None` for any other P3 backend (no
    such concept) or a run without debug data. Mirrors `_fast_cluster_stats`
    exactly."""
    if p3 not in ("LatestVectorClassification", "OldVectorClassification"):
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
    `_bench_doc_name`): the ground truth (`ground_truth_<type>.json`, only
    the non-empty types) + `benchmark_meta.json`, and one sub-folder per
    input form (`config.forms()`) -- each a complete viewer folder of that
    form's run (layers, `dump.json`, debug images, the label overlays of the
    types scored on that form, `vector_eval` layers; see
    `rastervec/Evaluation/forms.py` for what each form is scored on).
    Returns the `benchmark.json` entry."""
    doc_name = _bench_doc_name(bench)
    doc_dir = run_dir / doc_name
    doc_dir.mkdir(parents=True, exist_ok=True)
    pages = _filter_valid_pages(bench.pdf_path, config.pages_for(doc_name), bench.key)
    cfg = metrics.MetricConfig(iou_edge_min=config.iou_edge_min)
    gt_by_type = _bench_ground_truth_by_type(bench, pages)

    sources: list[str] = []
    for text_type in TEXT_TYPES:
        gt = gt_by_type[text_type]
        if not gt.entries and not gt.geometry_entries:
            if text_type != "native_to_vector":
                _LOG.info("%s: no %s labels for pages %s", bench.key, text_type, pages)
            continue
        save_labels(gt, str(doc_dir / f"ground_truth_{text_type}.json"))
        sources.append(text_type)

    forms = config.forms()
    for form in forms:
        _process_benchmark_form(bench, form, doc_dir / form, pages, gt_by_type, config, variant, cfg)

    (doc_dir / "benchmark_meta.json").write_text(json.dumps({
        "key": bench.key,
        "dir": doc_name,
        "source_pdf": str(bench.pdf_path),
        "labels": str(bench.labels_path) if bench.labels_path else None,
        "rasterised_pdf": str(bench.rasterised_pdf_path) if bench.rasterised_pdf_path else None,
        "pages": pages,
        "sources": sources,
        "forms": forms,
        "final_stage": config.final_stage,
        "variant": variant.name,
    }, indent=2), encoding="utf-8")
    return {
        "key": bench.key,
        "pdf_stem": bench.pdf_path.stem,
        "dir": doc_name,
        "sources": sources,
        "forms": {form: f"{doc_name}/{form}" for form in forms},
    }


def _process_benchmark_form(
    bench: BenchInput, form: str, form_dir: Path, pages: list[int], gt_by_type: "dict[str, LabelSet]",
    config: ReportConfig, variant, cfg: "metrics.MetricConfig",
) -> None:
    is_legacy = variant.engine == "legacy"
    form_dir.mkdir(parents=True, exist_ok=True)
    reservoirs = (
        _image_reservoirs(_image_dirs(form_dir), seed_name=f"{form_dir.parent.name}/{form}",
                          cap=config.debug_image_cap)
        if config.debug_images else None
    )
    active = _active_artifacts(config, variant)
    writer = _LayerWriter(form_dir / ".layer_parts")
    stats_pages: dict[str, list[tuple[int, dict]]] = {row[0]: [] for row in active}
    # Pages go to disk as they finish; the label overlays only need each
    # page's texts, so `overlay_pages` keeps copies without any vectors.
    dump = dump_io.DumpWriter(form_dir / "dump.json", str(bench.pdf_path), form=form)
    overlay_pages: list[dump_io.PageDump] = []
    text_types = FORM_TEXT_TYPES[form]
    vector_gt = FORM_VECTOR_GT[form]
    # Extra-prediction layers where manual vector text is scored on this form.
    has_manual = "original_vector" in text_types and bool(gt_by_type["original_vector"].entries)

    for pos, p in enumerate(pages):
        debug_durations: dict = {}
        clock = StepClock(debug_durations)
        with clock("conversion"):
            run_input, run_page, temp = _form_input(
                form, bench.pdf_path, p, form_dir, config, bench.rasterised_pdf_path,
            )
        writer.begin_page(pos, *_page_geometry(run_input, run_page))
        _LOG.info("benchmark %s page %d as %s (%s)", bench.key, p, form, variant.name)
        try:
            res = _run_form_page(run_input, run_page, p, config, variant, writer, reservoirs)
        except Exception as exc:  # noqa: BLE001 -- one failed page must not end the benchmark
            _LOG.warning("%s: %s run failed for page %d: %s", bench.key, form, p, exc)
            page_dump = _failed_page_dump(form, run_input, run_page, p, variant, exc)
            dump.add(page_dump)
            overlay_pages.append(page_dump)
            if temp is not None and not config.keep_converted_pdfs:
                temp.unlink(missing_ok=True)
            continue

        _accumulate_page(res, p, active, writer, stats_pages, reservoirs,
                         is_legacy=is_legacy, p3=variant.p3, clock=clock,
                         render_layers=config.debug_layers)
        if has_manual and config.debug_layers:
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

        dpi = adapters.raster_dpi_for_page(run_input, run_page) if form != "vectorised" else None
        p2_vectors = _p2_vectors(res) if form == "original" else None
        if vector_gt and config.debug_layers:
            page_gt = LabelSet(
                pdf_path=str(bench.pdf_path),
                entries=[e for t in TEXT_TYPES for e in gt_by_type[t].entries if e.page_index == p],
                geometry_entries=[
                    g for t in vector_gt for g in gt_by_type[t].geometry_entries if g.page_index == p
                ],
            )
            if page_gt.geometry_entries:
                try:
                    with clock("vector_eval"):
                        _add_vector_eval_layers(
                            writer, res.page.meta, page_gt,
                            p2_vectors if p2_vectors is not None else list(res.vectors or []),
                            dpi or vector_metrics.VectorMetricConfig().default_dpi,
                            gt_types=vector_gt,
                        )
                except Exception as exc:  # noqa: BLE001
                    _LOG.warning("%s: %s vector_eval layers failed for page %d: %s", bench.key, form, p, exc)

        page_dump = dump_io.PageDump(
            page_meta=res.page.meta, texts=list(res.texts or []),
            vectors=list(res.vectors or []), engine=variant.engine,
            step_durations=dict(res.step_durations or {}),
            substep_durations=_substeps(res),
            debug_durations=debug_durations,
            fast_cluster_stats=_fast_cluster_stats(res, variant.p3),
            retry_stats=_retry_stats(res, variant.p3),
            raster_dpi=dpi,
            p2_vectors=p2_vectors,
            counts=_run_counts(res, variant.p3),
        )
        dump.add(page_dump)
        overlay_pages.append(dataclasses.replace(page_dump, vectors=[], p2_vectors=None))
        del page_dump, p2_vectors
        if temp is not None and not config.keep_converted_pdfs:
            temp.unlink(missing_ok=True)
        del res
        gc.collect()

    extra_layers: list[dict] = []
    doc_durations: dict = {}
    all_texts = all_texts_are_predictions(form)
    for text_type in text_types:
        gt = gt_by_type[text_type]
        if not gt.entries or not config.debug_layers:
            continue
        with StepClock(doc_durations)("label_overlays"):
            _write_label_overlays(form_dir, text_type, gt, overlay_pages, cfg, all_texts=all_texts)
        for kind in ("bbox", "text"):
            extra_layers.append({
                "stage": "benchmark", "layer": f"{type_label(form, text_type)} label {kind}",
                "file": f"{text_type}_{kind}.pdf", "color": _OVERLAY_COLORS[text_type],
            })

    _finalize_doc_dir(form_dir, bench.pdf_path, pages, config, variant, active,
                      writer, stats_pages, dump,
                      extra_layers=tuple(extra_layers), doc_durations=doc_durations)


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
