"""The benchmark's per-page job as importable, picklable, top-level code
(it was inlined in `notebooks/benchmark_vector_classification.ipynb`), so
`run_parallel` can fan it across a process pool.

`PageTask.variant` names a `rastervec.Evaluation.Evaluate.variants.VARIANTS`
entry (engine current/legacy, `enable_fast`, light/heavy OCR backend).
Each variant is run **twice** per page on disjoint inputs, so auto and
manual ground truth are scored against physically separate runs that
cannot contaminate each other:

- **auto run** -- `convert_page_text_only` input (native text as vectors,
  drawings removed), scored vs the `source="auto"` labels.
- **manual run** -- `convert_page_drawings_only` input (original drawings
  only, native text removed), scored vs the `source="manual"` labels. Only
  fires when the page has manual labels.

One `PageTask` in -> one `PageResult` out (small, picklable). The per-page
output PDFs go into `RECONSTRUCT_DIR/<stem>_p<N>_<variant>/`
(`input_auto.pdf` / `input_manual.pdf` / `current.pdf` / `legacy.pdf` /
`boxes.pdf`). Every failure -- whole job or one of the runs -- is captured
into `PageResult.error` / a `report_blocks` line; the pool never sees an
exception.

The `current` engine is always run `verbose=True` here -- `build_eval_
inputs` needs `PipelineResult.clustering`/`fast_dropped_vectors` (both
verbose-only) for `attribute_miss`, and the showcase sampler needs
`unique_segments`/`unique_texts` (also verbose-only).
"""
from __future__ import annotations

import io
import math
import random
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from rastervec.Evaluation.conversion import (
    convert_page_drawings_only,
    convert_page_text_only,
)
from rastervec.Evaluation.Evaluate.adapters import (
    build_eval_inputs,
    gt_regions_from_labelset,
    predictions_from_texts,
    text_candidate_boxes,
)
from rastervec.Evaluation.Evaluate.benchmark import format_report
from rastervec.Evaluation.Evaluate.variants import PipelineVariant, resolve_variant
from rastervec.Evaluation.Evaluate.metrics import (
    MetricConfig,
    MetricSuiteResult,
    evaluate_metrics,
    overlay_boxes_split,
)
from rastervec.Evaluation.Labelling.auto_label import auto_label_pdf
from rastervec.Evaluation.Labelling.label_schema import (
    LabelEntry,
    LabelSet,
    split_labelset_by_source,
)
from rastervec.config import MIN_RENDER_SIDE_PX
from rastervec.helpers.geometry import PDF_POINTS_PER_INCH
from rastervec.logging_setup import get_logger
from rastervec.models import PageMeta
from rastervec.pipelines.current import run_pipeline
from rastervec.Reader.reader import Reader
from rastervec.renderer import (
    cluster_frame_size,
    render_boxes_pdf,
    render_reconstructed_pdf,
    render_vector_cluster,
)

_LOG = get_logger("reader.parallel.jobs")


@dataclass
class PageTask:
    pdf_path: str
    page_index: int
    manual_entries: list[LabelEntry] = field(default_factory=list)
    iou_edge_min: float = MetricConfig().iou_edge_min
    # A name from rastervec.Evaluation.Evaluate.variants.VARIANTS -- selects
    # the engine (current/legacy), enable_fast, and the OCR backend.
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
    auto: MetricSuiteResult | None = None
    manual: MetricSuiteResult | None = None
    stage_durations: dict[str, float] = field(default_factory=dict)
    total_seconds: float = 0.0
    report_blocks: list[str] = field(default_factory=list)
    showcase: list[ShowcaseSample] = field(default_factory=list)
    error: str | None = None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _ground_truth(task: PageTask) -> LabelSet:
    labels = auto_label_pdf(task.pdf_path, task.page_index)  # source="auto"
    labels.entries.extend(task.manual_entries)  # source="manual"
    return labels


def _page_inputs(task: PageTask, has_manual: bool) -> tuple[bytes, bytes | None]:
    """The two disjoint benchmark inputs for this page: text-only (auto) and,
    when the page has manual labels, drawings-only (manual)."""
    auto_input = convert_page_text_only(task.pdf_path, task.page_index)
    manual_input = (
        convert_page_drawings_only(task.pdf_path, task.page_index)
        if has_manual else None
    )
    return auto_input, manual_input


def _run_pipeline(input_bytes: bytes, *, enable_fast: bool = True, compute=None):
    """Full current-pipeline run on one input PDF -> its PipelineResult.
    Always `verbose=True` -- see this module's docstring."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "in.pdf"
        path.write_bytes(input_bytes)
        return run_pipeline(str(path), 0, enable_fast=enable_fast, verbose=True, compute=compute)


def _original_page_meta(pdf_path: str, page_index: int) -> PageMeta:
    with Reader(pdf_path) as reader:
        return reader.get_page(page_index).meta


def _render_ocr_input(vectors, dpi: int = 300):
    """The exact image OCR's own render step feeds PaddleOCR for one
    UniqueSegment/cluster (dpi bumped up the same way for a tiny render)."""
    width_pt, height_pt = cluster_frame_size(vectors)
    min_side_pt = min(width_pt, height_pt)
    if min_side_pt > 0:
        dpi = max(
            dpi,
            math.ceil(MIN_RENDER_SIDE_PX * PDF_POINTS_PER_INCH / min_side_pt),
        )
    return render_vector_cluster(vectors, dpi)


def _showcase(
    unique_pairs: list[tuple], per_page: int, seed: int,
) -> list[ShowcaseSample]:
    """`unique_pairs` is `[(UniqueSegment, Text), ...]` -- one real render +
    OCR call per pair (mirrors the old per-cluster showcase, now sampling
    the deduped unique segments instead of every candidate cluster)."""
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
        try:
            image = _render_ocr_input(seg.vectors)
        except Exception:  # noqa: BLE001 -- a bad crop shouldn't kill the page
            continue
        buf = io.BytesIO()
        image.save(buf, format="PNG")
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


# --------------------------------------------------------------------------
# the job -- current pipeline (run twice)
# --------------------------------------------------------------------------
def _run_current(
    task: PageTask, gt: LabelSet, cfg: MetricConfig, variant: PipelineVariant,
    compute=None,
) -> PageResult:
    by_src = split_labelset_by_source(gt)
    auto_gt = gt_regions_from_labelset(by_src["auto"])
    manual_gt = gt_regions_from_labelset(by_src["manual"])
    has_manual = bool(by_src["manual"].entries)
    auto_input, manual_input = _page_inputs(task, has_manual)

    result = PageResult(
        pdf_path=task.pdf_path, page_index=task.page_index, variant=task.variant,
    )
    auto_ctx = manual_ctx = None
    auto_preds: list = []
    manual_preds: list = []
    total = 0.0
    run_kw = dict(enable_fast=variant.enable_fast, compute=compute)
    lbl = task.variant

    try:
        auto_ctx = _run_pipeline(auto_input, **run_kw)
        inp = build_eval_inputs(auto_ctx)
        auto_preds = inp.predictions
        result.auto = evaluate_metrics(
            auto_gt, inp.predictions, inp.text_candidate_boxes,
            clustering=inp.clustering, fast_dropped=inp.fast_dropped,
            ocr_failed=inp.ocr_failed, cfg=cfg,
        )
        result.report_blocks.append(
            format_report(f"[{lbl}/auto]   {task.pdf_path}", task.page_index, result.auto)
        )
        total += sum((auto_ctx.step_durations or {}).values())
    except Exception as exc:  # noqa: BLE001
        result.report_blocks.append(
            f"[{lbl}/auto] {task.pdf_path} p{task.page_index}: run failed: {exc}"
        )

    if has_manual and manual_input is not None:
        try:
            manual_ctx = _run_pipeline(manual_input, **run_kw)
            inp = build_eval_inputs(manual_ctx)
            manual_preds = inp.predictions
            result.manual = evaluate_metrics(
                manual_gt, inp.predictions, inp.text_candidate_boxes,
                clustering=inp.clustering, fast_dropped=inp.fast_dropped,
                ocr_failed=inp.ocr_failed, cfg=cfg,
            )
            result.report_blocks.append(
                format_report(f"[{lbl}/manual] {task.pdf_path}", task.page_index, result.manual)
            )
            total += sum((manual_ctx.step_durations or {}).values())
        except Exception as exc:  # noqa: BLE001
            result.report_blocks.append(
                f"[{lbl}/manual] {task.pdf_path} p{task.page_index}: run failed: {exc}"
            )

    result.stage_durations = dict((auto_ctx.step_durations or {}) if auto_ctx else {})
    result.total_seconds = total

    unique_pairs: list[tuple] = []
    for ctx in (auto_ctx, manual_ctx):
        if ctx is not None:
            unique_pairs.extend(zip(ctx.unique_segments or [], ctx.unique_texts or []))
    result.showcase = _showcase(unique_pairs, task.showcase_per_page, task.showcase_seed)

    ocr_texts = [
        t for ctx in (auto_ctx, manual_ctx) if ctx is not None
        for t in ctx.texts if t.source == "ocr"
    ]
    _write_current_outputs(
        task,
        page_meta=(
            (auto_ctx or manual_ctx).page.meta if (auto_ctx or manual_ctx)
            else _original_page_meta(task.pdf_path, task.page_index)
        ),
        auto_input=auto_input,
        manual_input=manual_input if has_manual else None,
        merged_ocr_results=ocr_texts,
        auto_gt=auto_gt, auto_preds=auto_preds,
        manual_gt=manual_gt if has_manual else [], manual_preds=manual_preds,
        cfg=cfg,
    )
    return result


def _write_current_outputs(
    task: PageTask, *, page_meta: PageMeta, auto_input: bytes,
    manual_input: bytes | None, merged_ocr_results, auto_gt, auto_preds,
    manual_gt, manual_preds, cfg: MetricConfig,
) -> None:
    directory = _page_dir(task)
    if directory is None:
        return
    (directory / "input_auto.pdf").write_bytes(auto_input)
    if manual_input is not None:
        (directory / "input_manual.pdf").write_bytes(manual_input)
    (directory / "current.pdf").write_bytes(
        render_reconstructed_pdf(page_meta, ocr_results=merged_ocr_results)
    )
    (directory / "boxes.pdf").write_bytes(
        render_boxes_pdf(
            page_meta,
            overlay_boxes_split(auto_gt, auto_preds, manual_gt, manual_preds, cfg),
        )
    )


# --------------------------------------------------------------------------
# the job -- legacy pipeline (run twice)
# --------------------------------------------------------------------------
def _run_legacy(task: PageTask, gt: LabelSet, cfg: MetricConfig) -> PageResult:
    from rastervec.Evaluation.Evaluate.legacy_adapter import run_archive_pipeline, to_texts

    by_src = split_labelset_by_source(gt)
    auto_gt = gt_regions_from_labelset(by_src["auto"])
    manual_gt = gt_regions_from_labelset(by_src["manual"])
    has_manual = bool(by_src["manual"].entries)
    auto_input, manual_input = _page_inputs(task, has_manual)

    result = PageResult(
        pdf_path=task.pdf_path, page_index=task.page_index, variant=task.variant,
    )
    merged_ocr: list = []
    total = 0.0
    lbl = task.variant

    def _legacy_run(input_bytes: bytes, gt_regions, label: str) -> MetricSuiteResult:
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
        res = evaluate_metrics(
            gt_regions, predictions_from_texts(texts),
            text_candidate_boxes(texts), cfg=cfg,
        )
        result.report_blocks.append(
            format_report(f"[{lbl}/{label}] {task.pdf_path}", task.page_index, res)
        )
        return res

    # No per-run try/except here: a legacy failure (archive import, LibreOffice,
    # PaddleOCR) must propagate to `run_page_task`'s outer boundary -- which
    # logs it and records `PageResult.error` -- rather than be swallowed into a
    # `report_blocks` line with `None` metrics (a silent near-zero score).
    result.auto = _legacy_run(auto_input, auto_gt, "auto")
    if has_manual and manual_input is not None:
        result.manual = _legacy_run(manual_input, manual_gt, "manual")

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


def run_page_task(task: PageTask, compute=None) -> PageResult:
    """One benchmarked page, end to end. Never raises -- a failure is
    captured into `PageResult.error` (a whole-job failure) or a
    `report_blocks` line (one of the two runs). `compute`, when given a
    shared compute-pool proxy (see `run_benchmark`'s `compute_workers`),
    is forwarded to the `current` engine only -- the `legacy` engine is
    completely unaffected by this parameter."""
    cfg = MetricConfig(iou_edge_min=task.iou_edge_min)
    try:
        variant = resolve_variant(task.variant)
        gt = _ground_truth(task)
        if variant.engine == "legacy":
            return _run_legacy(task, gt, cfg)
        return _run_current(task, gt, cfg, variant, compute=compute)
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

    `compute_workers > 0` additionally starts a `multiprocessing.Manager`
    -hosted `Pool` (Pool 2) sized `compute_workers` (via `pool.compute_pool`),
    shared by every page job regardless of which Pool-1 worker runs it -- a
    complex page's many FAST/OCR jobs and simple pages' few jobs all queue
    into this one pool, so idle capacity is never stranded on a page that
    finished early. Pool 2 never imports `fitz`/`pymupdf`. `compute_workers=0`
    (the default) preserves today's fully-local-per-page behavior."""
    import functools

    from rastervec.Reader.Parallel.pool import compute_pool, run_parallel

    with compute_pool(compute_workers) as compute:
        fn = (
            run_page_task
            if compute is None
            else functools.partial(run_page_task, compute=compute)
        )
        return run_parallel(tasks, fn, workers=workers, desc=desc)
