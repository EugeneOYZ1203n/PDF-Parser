"""The benchmark's per-page job as importable, picklable, top-level code
(it was inlined in `notebooks/benchmark_vector_classification.ipynb`), so
`run_parallel` can fan it across a process pool.

`PageTask.variant` names a `rastervec.Evaluation.Evaluate.variants.VARIANTS`
entry (engine current/legacy, `enable_fast`).

Reworked for the 4 text-type / 3 vector-type metrics suite
(`Evaluation.Evaluate.metrics`/`vector_metrics`). Two pipeline runs happen
per page, each on a disjoint synthetic input, mirroring the old auto/manual
split but generalized:

- **native_to_vector run** -- `convert_page_text_only` input (native text as
  vectors, drawings removed), scored vs the `native_to_vector` GT bucket
  (`source="native"` labels). Only fires when the page has native labels.
- **original_vector run** -- `convert_page_drawings_only` input (original
  drawings only, native text removed), scored vs the `original_vector` GT
  bucket (`source="vector"` labels). Only fires when the page has
  vector-sourced labels.

`vector_to_raster`/`original_raster` (both text and vector tables) have no
converter/pipeline stage (no raster-tracing stage exists yet) -- they are
always scored with an empty prediction set, so their tables report GT-only
stats + 0 recall, never a crash.

One `PageTask` in -> one `PageResult` out (small, picklable). The per-page
output PDFs go into `RECONSTRUCT_DIR/<stem>_p<N>_<variant>/`
(`input_native_to_vector.pdf` / `input_original_vector.pdf` / `current.pdf`
/ `legacy.pdf` / `boxes.pdf`). Every failure -- whole job or one of the two
runs -- is captured into `PageResult.error` / a `report_blocks` line; the
pool never sees an exception.

The `current` engine is always run `verbose=True` here -- `reclassify_
result`/`vectors`/`reassigned_text` (verbose-only) feed the classification
funnel + vector-pairing tables, and the showcase sampler needs
`rotated_segments`/`restored_texts` (also verbose-only).
"""
from __future__ import annotations

import io
import random
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from rastervec.Evaluation.conversion import (
    convert_page_drawings_only,
    convert_page_text_only,
)
from rastervec.Evaluation.Evaluate import vector_metrics as vm
from rastervec.Evaluation.Evaluate.adapters import (
    build_vector_eval_inputs,
    entries_by_text_type,
    enrich_native_vector_signatures,
    gt_regions_by_text_type,
    gt_vector_signatures_by_text_type,
    predictions_from_texts,
    reclass_passed_signatures,
)
from rastervec.Evaluation.Evaluate.benchmark import format_report
from rastervec.Evaluation.Evaluate.variants import PipelineVariant, resolve_variant
from rastervec.Evaluation.Evaluate.metrics import (
    TEXT_TYPES,
    MetricConfig,
    TextMetricSuiteResult,
    evaluate_text_metrics,
    overlay_boxes_by_type,
    build_overlap_graphs_by_type,
)
from rastervec.Evaluation.Labelling.native_label import native_label_pdf
from rastervec.Evaluation.Labelling.label_schema import (
    LabelEntry,
    LabelSet,
)
from rastervec.logging_setup import get_logger
from rastervec.models import PageMeta, Segment, Text
from rastervec.pipelines.current import run_pipeline
from rastervec.Reader.reader import Reader
from rastervec.renderer import render_boxes_pdf, render_reconstructed_pdf
from rastervec.Vector.vector import extract_vectors

_LOG = get_logger("reader.parallel.jobs")


@dataclass
class PageTask:
    pdf_path: str
    page_index: int
    manual_entries: list[LabelEntry] = field(default_factory=list)
    iou_edge_min: float = MetricConfig().iou_edge_min
    variant: str = "current"
    reconstruct_dir: str | None = None
    showcase_per_page: int = 4
    enable_archive_raster_pass: bool = False
    showcase_seed: int = 0


@dataclass
class ShowcaseSample:
    png: bytes
    text: str
    passed: bool


@dataclass
class PageResult:
    pdf_path: str
    page_index: int
    variant: str
    text_metrics: TextMetricSuiteResult | None = None
    vector_metrics: "vm.VectorMetricSuiteResult | None" = None
    stage_durations: dict[str, float] = field(default_factory=dict)
    total_seconds: float = 0.0
    report_blocks: list[str] = field(default_factory=list)
    showcase: list[ShowcaseSample] = field(default_factory=list)
    error: str | None = None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _ground_truth(task: PageTask) -> LabelSet:
    labels = native_label_pdf(task.pdf_path, task.page_index)  # source="native"
    labels.entries.extend(task.manual_entries)  # source in ("vector", "raster")
    return labels


def _fresh_vectors(pdf_path: str, page_index: int) -> list:
    with Reader(pdf_path) as reader:
        page = reader.get_page(page_index)
        return extract_vectors(page)


def _run_pipeline(
    input_bytes: bytes, *, enable_fast: bool = True,
    compute=None, progress_counter=None,
):
    """Full `pipelines.current.run_pipeline` run on one input PDF -> its
    PipelineResult ("legacy" is handled separately by `_run_legacy`, never
    reaches here). Always `verbose=True` -- see this module's docstring."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "in.pdf"
        path.write_bytes(input_bytes)
        return run_pipeline(
            str(path), 0, enable_fast=enable_fast, verbose=True, compute=compute,
            progress_counter=progress_counter,
        )


def _original_page_meta(pdf_path: str, page_index: int) -> PageMeta:
    with Reader(pdf_path) as reader:
        return reader.get_page(page_index).meta


def _showcase(
    unique_pairs: "list[tuple[Segment, Text]]", per_page: int, seed: int,
) -> list[ShowcaseSample]:
    if per_page <= 0 or not unique_pairs:
        return []
    passed = [(seg, t) for seg, t in unique_pairs if t.text.strip()]
    blank = [(seg, t) for seg, t in unique_pairs if not t.text.strip()]
    rng = random.Random(seed)
    half = per_page // 2
    pick = rng.sample(passed, min(half, len(passed)))
    pick += rng.sample(blank, min(per_page - len(pick), len(blank)))
    chosen = {id(seg) for seg, _t in pick}
    rest = [(seg, t) for seg, t in unique_pairs if id(seg) not in chosen]
    rng.shuffle(rest)
    pick += rest[: max(0, per_page - len(pick))]

    out: list[ShowcaseSample] = []
    for seg, t in pick:
        if seg.image is None:
            continue
        buf = io.BytesIO()
        Image.fromarray(seg.image).save(buf, format="PNG")
        text = t.text.strip()
        out.append(ShowcaseSample(png=buf.getvalue(), text=text, passed=bool(text)))
    return out


def _page_dir(task: PageTask) -> Path | None:
    if not task.reconstruct_dir:
        return None
    directory = (
        Path(task.reconstruct_dir)
        / f"{Path(task.pdf_path).stem}_p{task.page_index}_{task.variant}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _evaluate_one_type(
    text_type: str,
    gt_by_type: dict, entries_by_type_: dict, predictions: list,
    gt_vec_sigs_by_type: dict, survived_signatures: set,
    cfg: MetricConfig,
):
    """`evaluate_text_metrics` scores all 4 types from one shared
    `predictions` list; since `native_to_vector`/`original_vector` come
    from two disjoint pipeline runs here, this is called once per run and
    only that run's own type is kept from the result (the other 3 types'
    entries in that call used an empty/irrelevant gt bucket and are
    discarded)."""
    restricted_gt = {t: (gt_by_type.get(t, []) if t == text_type else []) for t in TEXT_TYPES}
    restricted_entries = {
        t: (entries_by_type_.get(t, []) if t == text_type else []) for t in TEXT_TYPES
    }
    result = evaluate_text_metrics(
        restricted_gt, restricted_entries, predictions,
        gt_vector_signatures_by_type=gt_vec_sigs_by_type,
        survived_signatures=survived_signatures,
        cfg=cfg,
    )
    return result


def _merge_text_results(
    empty_result: TextMetricSuiteResult,
    per_type_results: "dict[str, TextMetricSuiteResult]",
) -> TextMetricSuiteResult:
    by_type = dict(empty_result.by_type)
    bbox_unclassified = empty_result.bbox_unclassified
    for text_type, result in per_type_results.items():
        by_type[text_type] = result.by_type[text_type]
        bbox_unclassified = result.bbox_unclassified  # last real run wins
    return TextMetricSuiteResult(by_type=by_type, bbox_unclassified=bbox_unclassified)


# --------------------------------------------------------------------------
# the job -- current pipeline (run twice, disjoint inputs)
# --------------------------------------------------------------------------
def _run_current(
    task: PageTask, gt: LabelSet, cfg: MetricConfig, variant: PipelineVariant,
    compute=None, progress_counter=None,
) -> PageResult:
    entries = entries_by_text_type(gt)
    has_native = bool(entries["native_to_vector"])
    has_vector = bool(entries["original_vector"])

    if has_native:
        enrich_native_vector_signatures(gt, task.pdf_path, task.page_index)

    gt_by_type = gt_regions_by_text_type(gt)
    gt_vec_sigs_by_type = gt_vector_signatures_by_text_type(gt)

    result = PageResult(
        pdf_path=task.pdf_path, page_index=task.page_index, variant=task.variant,
    )
    per_type_results: "dict[str, TextMetricSuiteResult]" = {}
    native_ctx = vector_ctx = None
    total = 0.0
    run_kw = dict(
        enable_fast=variant.enable_fast,
        compute=compute, progress_counter=progress_counter,
    )
    lbl = task.variant

    if has_native:
        try:
            native_input = convert_page_text_only(task.pdf_path, task.page_index)
            native_ctx = _run_pipeline(native_input, **run_kw)
            ocr_texts = [t for t in native_ctx.texts if t.source == "ocr"]
            preds = predictions_from_texts(ocr_texts)
            per_type_results["native_to_vector"] = _evaluate_one_type(
                "native_to_vector", gt_by_type, entries, preds,
                gt_vec_sigs_by_type, reclass_passed_signatures(native_ctx), cfg,
            )
            result.report_blocks.append(
                f"[{lbl}/native_to_vector] {task.pdf_path} p{task.page_index}: scored"
            )
            total += sum((native_ctx.step_durations or {}).values())
        except Exception as exc:  # noqa: BLE001
            result.report_blocks.append(
                f"[{lbl}/native_to_vector] {task.pdf_path} p{task.page_index}: run failed: {exc}"
            )

    if has_vector:
        try:
            vector_input = convert_page_drawings_only(task.pdf_path, task.page_index)
            vector_ctx = _run_pipeline(vector_input, **run_kw)
            ocr_texts = [t for t in vector_ctx.texts if t.source == "ocr"]
            preds = predictions_from_texts(ocr_texts)
            per_type_results["original_vector"] = _evaluate_one_type(
                "original_vector", gt_by_type, entries, preds,
                gt_vec_sigs_by_type, reclass_passed_signatures(vector_ctx), cfg,
            )
            result.report_blocks.append(
                f"[{lbl}/original_vector] {task.pdf_path} p{task.page_index}: scored"
            )
            total += sum((vector_ctx.step_durations or {}).values())
        except Exception as exc:  # noqa: BLE001
            result.report_blocks.append(
                f"[{lbl}/original_vector] {task.pdf_path} p{task.page_index}: run failed: {exc}"
            )

    empty_result = evaluate_text_metrics(
        {t: [] for t in TEXT_TYPES}, {t: [] for t in TEXT_TYPES}, [],
        cfg=cfg,
    )
    result.text_metrics = _merge_text_results(empty_result, per_type_results)

    try:
        fresh_vectors = _fresh_vectors(task.pdf_path, task.page_index)
        vector_inputs = build_vector_eval_inputs(gt, vector_ctx, fresh_vectors)
        result.vector_metrics = vm.evaluate_vector_metrics(
            vector_inputs.gt_by_type, vector_inputs.preds_by_type, vector_inputs.label_counts,
        )
    except Exception as exc:  # noqa: BLE001
        result.report_blocks.append(
            f"[{lbl}/vectors] {task.pdf_path} p{task.page_index}: vector scoring failed: {exc}"
        )

    result.stage_durations = dict((native_ctx.step_durations or {}) if native_ctx else {})
    result.total_seconds = total

    unique_pairs: list[tuple] = []
    for ctx in (native_ctx, vector_ctx):
        if ctx is not None:
            unique_pairs.extend(zip(ctx.rotated_segments or [], ctx.restored_texts or []))
    result.showcase = _showcase(unique_pairs, task.showcase_per_page, task.showcase_seed)

    ocr_texts_all = [
        t for ctx in (native_ctx, vector_ctx) if ctx is not None
        for t in ctx.texts if t.source == "ocr"
    ]
    _write_current_outputs(
        task,
        page_meta=(
            (native_ctx or vector_ctx).page.meta if (native_ctx or vector_ctx)
            else _original_page_meta(task.pdf_path, task.page_index)
        ),
        native_input=(convert_page_text_only(task.pdf_path, task.page_index) if has_native else None),
        vector_input=(convert_page_drawings_only(task.pdf_path, task.page_index) if has_vector else None),
        merged_ocr_results=ocr_texts_all,
        gt_by_type=gt_by_type,
        preds_native=(predictions_from_texts([t for t in native_ctx.texts if t.source == "ocr"]) if native_ctx else []),
        preds_vector=(predictions_from_texts([t for t in vector_ctx.texts if t.source == "ocr"]) if vector_ctx else []),
        cfg=cfg,
    )
    return result


def _write_current_outputs(
    task: PageTask, *, page_meta: PageMeta, native_input: "bytes | None",
    vector_input: "bytes | None", merged_ocr_results, gt_by_type,
    preds_native, preds_vector, cfg: MetricConfig,
) -> None:
    directory = _page_dir(task)
    if directory is None:
        return
    if native_input is not None:
        (directory / "input_native_to_vector.pdf").write_bytes(native_input)
    if vector_input is not None:
        (directory / "input_original_vector.pdf").write_bytes(vector_input)
    (directory / "current.pdf").write_bytes(
        render_reconstructed_pdf(page_meta, ocr_results=merged_ocr_results)
    )
    # Box overlay: native predictions scored against native_to_vector GT,
    # vector predictions against original_vector GT; the other two types
    # (no pipeline run) contribute GT-only (all-red) boxes.
    graphs_by_type = build_overlap_graphs_by_type(
        {
            "native_to_vector": gt_by_type.get("native_to_vector", []),
            "original_vector": [],
            "vector_to_raster": [],
            "original_raster": [],
        },
        preds_native, cfg,
    )
    graphs_by_type["original_vector"] = build_overlap_graphs_by_type(
        {"native_to_vector": [], "original_vector": gt_by_type.get("original_vector", []),
         "vector_to_raster": [], "original_raster": []},
        preds_vector, cfg,
    )["original_vector"]
    for t in ("vector_to_raster", "original_raster"):
        graphs_by_type[t] = build_overlap_graphs_by_type(
            {tt: (gt_by_type.get(t, []) if tt == t else []) for tt in TEXT_TYPES}, [], cfg,
        )[t]
    (directory / "boxes.pdf").write_bytes(
        render_boxes_pdf(page_meta, overlay_boxes_by_type(graphs_by_type))
    )


# --------------------------------------------------------------------------
# the job -- legacy pipeline (run twice, disjoint inputs)
# --------------------------------------------------------------------------
def _run_legacy(task: PageTask, gt: LabelSet, cfg: MetricConfig) -> PageResult:
    from rastervec.Evaluation.Evaluate.legacy_adapter import run_archive_pipeline, to_texts

    entries = entries_by_text_type(gt)
    has_native = bool(entries["native_to_vector"])
    has_vector = bool(entries["original_vector"])

    if has_native:
        enrich_native_vector_signatures(gt, task.pdf_path, task.page_index)
    gt_by_type = gt_regions_by_text_type(gt)
    gt_vec_sigs_by_type = gt_vector_signatures_by_text_type(gt)

    result = PageResult(
        pdf_path=task.pdf_path, page_index=task.page_index, variant=task.variant,
    )
    per_type_results: "dict[str, TextMetricSuiteResult]" = {}
    merged_ocr: list = []
    total = 0.0
    lbl = task.variant

    def _legacy_run(input_bytes: bytes, text_type: str) -> None:
        nonlocal total
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "in.pdf"
            path.write_bytes(input_bytes)
            t0 = time.perf_counter()
            elements = run_archive_pipeline(
                str(path), 0, enable_raster_pass=task.enable_archive_raster_pass,
            )
            total += time.perf_counter() - t0
        texts = to_texts(elements, page_index=task.page_index)
        merged_ocr.extend(texts)
        preds = predictions_from_texts(texts)
        # Legacy has no ReclassifyResult -- funnel is always N/A for it.
        per_type_results[text_type] = _evaluate_one_type(
            text_type, gt_by_type, entries, preds, gt_vec_sigs_by_type, set(), cfg,
        )
        result.report_blocks.append(
            f"[{lbl}/{text_type}] {task.pdf_path} p{task.page_index}: scored"
        )

    # No per-run try/except here: a legacy failure (archive import, LibreOffice,
    # PaddleOCR) must propagate to `run_page_task`'s outer boundary -- which
    # logs it and records `PageResult.error` -- rather than be swallowed into a
    # `report_blocks` line with `None` metrics (a silent near-zero score).
    if has_native:
        native_input = convert_page_text_only(task.pdf_path, task.page_index)
        _legacy_run(native_input, "native_to_vector")
    if has_vector:
        vector_input = convert_page_drawings_only(task.pdf_path, task.page_index)
        _legacy_run(vector_input, "original_vector")

    empty_result = evaluate_text_metrics(
        {t: [] for t in TEXT_TYPES}, {t: [] for t in TEXT_TYPES}, [], cfg=cfg,
    )
    result.text_metrics = _merge_text_results(empty_result, per_type_results)

    try:
        fresh_vectors = _fresh_vectors(task.pdf_path, task.page_index)
        # Legacy has no comparable "final drawing vector" population exposed
        # here -- original_vector vector-type scoring is skipped (res=None).
        vector_inputs = build_vector_eval_inputs(gt, None, fresh_vectors)
        result.vector_metrics = vm.evaluate_vector_metrics(
            vector_inputs.gt_by_type, vector_inputs.preds_by_type, vector_inputs.label_counts,
        )
    except Exception as exc:  # noqa: BLE001
        result.report_blocks.append(
            f"[{lbl}/vectors] {task.pdf_path} p{task.page_index}: vector scoring failed: {exc}"
        )

    result.total_seconds = total
    directory = _page_dir(task)
    if directory is not None:
        (directory / "legacy.pdf").write_bytes(
            render_reconstructed_pdf(
                _original_page_meta(task.pdf_path, task.page_index),
                ocr_results=merged_ocr,
            )
        )
    return result


def run_page_task(task: PageTask, compute=None, progress_counter=None) -> PageResult:
    """One benchmarked page, end to end. Never raises -- a failure is
    captured into `PageResult.error` (a whole-job failure) or a
    `report_blocks` line (one of the two runs)."""
    cfg = MetricConfig(iou_edge_min=task.iou_edge_min)
    try:
        variant = resolve_variant(task.variant)
        gt = _ground_truth(task)
        if variant.engine == "legacy":
            return _run_legacy(task, gt, cfg)
        return _run_current(task, gt, cfg, variant, compute=compute, progress_counter=progress_counter)
    except Exception as exc:  # noqa: BLE001 -- keep benchmarking the rest
        _LOG.warning("%s page %d failed: %s", task.pdf_path, task.page_index, exc)
        return PageResult(
            pdf_path=task.pdf_path, page_index=task.page_index, variant=task.variant,
            error=f"{exc}\n{traceback.format_exc()}",
        )


def run_benchmark(
    tasks: list[PageTask], *, workers: int = 1, compute_workers: int = 0,
    desc: str = "benchmark",
) -> list[PageResult]:
    """Run `run_page_task` over `tasks` (serial when `workers <= 1`,
    otherwise a spawn process pool -- Pool 1), results in input order.
    `compute_workers > 0` additionally starts a shared Pool 2 -- see
    `Reader/Parallel/pool.py`."""
    import functools
    import multiprocessing

    from rastervec.Reader.Parallel.pool import compute_pool, run_parallel

    progress_manager = multiprocessing.Manager()
    try:
        progress_counter = progress_manager.Value("i", 0)
        with compute_pool(compute_workers) as compute:
            fn = functools.partial(
                run_page_task, compute=compute, progress_counter=progress_counter,
            )
            return run_parallel(
                tasks, fn, workers=workers, desc=desc, progress_counter=progress_counter,
            )
    finally:
        progress_manager.shutdown()
