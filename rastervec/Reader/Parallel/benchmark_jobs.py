"""The benchmark's per-page job as importable, picklable, top-level code
(it was inlined in `notebooks/benchmark_vector_classification.ipynb`), so
`run_parallel` can fan it across a process pool.

One `PageTask` in -> one `PageResult` out (small, picklable: metric
suites, timings, a few PNG-bytes showcase samples, formatted report
text). Reconstruction / input / box-overlay PDFs are written straight to
`reconstruct_dir` from inside the job (distinct filenames, process-safe).
Every failure is caught into `PageResult.error` -- the pool never sees an
exception.
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
from typing import Literal

import pymupdf as fitz

from rastervec.Evaluation.Conversion.conversion import convert_page_to_vector_text
from rastervec.Evaluation.Evaluate.adapters import (
    build_eval_inputs,
    gt_regions_from_labelset,
    predictions_from_cluster_ocr,
    text_candidate_boxes,
)
from rastervec.Evaluation.Evaluate.benchmark import format_report
from rastervec.Evaluation.Evaluate.evaluate import split_labelset_by_source
from rastervec.Evaluation.Evaluate.metrics import (
    MetricConfig,
    MetricSuiteResult,
    build_overlap_graph,
    evaluate_metrics,
    overlay_boxes,
)
from rastervec.Evaluation.Labelling.auto_label import auto_label_pdf
from rastervec.Evaluation.Labelling.label_schema import LabelEntry, LabelSet
from rastervec.logging_setup import get_logger
from rastervec.models import PageMeta
from rastervec.OCR.Paddle_OCR.render_ocr import MIN_RENDER_SIDE_PX
from rastervec.pipeline import run_page_context
from rastervec.Reader.reader import Reader
from rastervec.renderer import (
    cluster_frame_size,
    render_boxes_pdf,
    render_reconstructed_pdf,
    render_vector_cluster,
)

_LOG = get_logger("reader.parallel.jobs")

Pipeline_ = Literal["current", "legacy"]


@dataclass
class PageTask:
    pdf_path: str
    page_index: int
    manual_entries: list[LabelEntry] = field(default_factory=list)
    iou_edge_min: float = MetricConfig().iou_edge_min
    pipeline: Pipeline_ = "current"
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
    pipeline: Pipeline_
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


def _original_page_meta(pdf_path: str, page_index: int) -> PageMeta:
    with Reader(pdf_path) as reader:
        return reader.get_page(page_index).meta


def _original_page_bytes(pdf_path: str, page_index: int) -> bytes:
    src = fitz.open(pdf_path)
    try:
        one = fitz.open()
        one.insert_pdf(src, from_page=page_index, to_page=page_index)
        return one.tobytes()
    finally:
        src.close()


def _render_ocr_input(cluster, dpi: int = 300):
    """The exact image RenderOCR.ocr_cluster feeds PaddleOCR for this
    cluster (dpi bumped up the same way for a tiny cluster)."""
    width_pt, height_pt = cluster_frame_size(cluster)
    min_side_pt = min(width_pt, height_pt)
    if min_side_pt > 0:
        dpi = max(dpi, math.ceil(MIN_RENDER_SIDE_PX * 72.0 / min_side_pt))
    return render_vector_cluster(cluster, dpi)


def _showcase(cluster_ocr_results, per_page: int, seed: int) -> list[ShowcaseSample]:
    if per_page <= 0 or not cluster_ocr_results:
        return []
    passed = [r for r in cluster_ocr_results if r.resolved.text.strip()]
    blank = [r for r in cluster_ocr_results if not r.resolved.text.strip()]
    rng = random.Random(seed)
    half = per_page // 2
    pick = rng.sample(passed, min(half, len(passed)))
    pick += rng.sample(blank, min(per_page - len(pick), len(blank)))
    chosen = {id(r) for r in pick}
    rest = [r for r in cluster_ocr_results if id(r) not in chosen]
    rng.shuffle(rest)
    pick += rest[: max(0, per_page - len(pick))]

    out: list[ShowcaseSample] = []
    for r in pick:
        try:
            image = _render_ocr_input(r.cluster)
        except Exception:  # noqa: BLE001 -- a bad crop shouldn't kill the page
            continue
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        text = r.resolved.text.strip()
        out.append(ShowcaseSample(png=buf.getvalue(), text=text, passed=bool(text)))
    return out


def _write_pdf(directory: Path, name: str, data: bytes) -> None:
    (directory / name).write_bytes(data)


def _write_outputs(
    task: PageTask,
    gt: LabelSet,
    page_meta: PageMeta,
    input_pdf: bytes,
    ocr_results,
    predictions,
    cfg: MetricConfig,
) -> None:
    """groundtruth / <pipeline> reconstruction (text-only) / <pipeline>
    input / <pipeline> box-overlay PDFs into `reconstruct_dir`."""
    if not task.reconstruct_dir:
        return
    directory = Path(task.reconstruct_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = Path(task.pdf_path).stem
    tag = f"{stem}_p{task.page_index}"

    _write_pdf(
        directory, f"{tag}_groundtruth.pdf",
        render_reconstructed_pdf(
            page_meta,
            text_boxes=[(e.text, e.cluster_bbox, e.expected_rotation) for e in gt.entries],
        ),
    )
    _write_pdf(
        directory, f"{tag}_{task.pipeline}.pdf",
        render_reconstructed_pdf(page_meta, ocr_results=ocr_results),
    )
    _write_pdf(directory, f"{tag}_{task.pipeline}_input.pdf", input_pdf)

    for source, labels in split_labelset_by_source(gt).items():
        if not labels.entries:
            continue
        graph = build_overlap_graph(gt_regions_from_labelset(labels), predictions, cfg)
        _write_pdf(
            directory, f"{tag}_{task.pipeline}_boxes_{source}.pdf",
            render_boxes_pdf(page_meta, overlay_boxes(graph)),
        )


# --------------------------------------------------------------------------
# the job
# --------------------------------------------------------------------------
def _run_current(task: PageTask, gt: LabelSet, cfg: MetricConfig) -> PageResult:
    converted = convert_page_to_vector_text(task.pdf_path, task.page_index)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "converted.pdf")
        Path(path).write_bytes(converted)
        with Reader(path) as reader:
            ctx = run_page_context(reader, 0)

    inp = build_eval_inputs(ctx)
    by_src = split_labelset_by_source(gt)
    result = PageResult(
        pdf_path=task.pdf_path, page_index=task.page_index, pipeline="current",
        stage_durations=dict(ctx.stage_durations or {}),
    )
    result.total_seconds = sum(result.stage_durations.values())

    result.auto = evaluate_metrics(
        gt_regions_from_labelset(by_src["auto"]), inp.predictions,
        inp.text_candidate_boxes, clustering=inp.clustering,
        fast_dropped=inp.fast_dropped, ocr_failed=inp.ocr_failed, cfg=cfg,
    )
    result.report_blocks.append(
        format_report(f"[current/auto]   {task.pdf_path}", task.page_index, result.auto)
    )
    if by_src["manual"].entries:
        result.manual = evaluate_metrics(
            gt_regions_from_labelset(by_src["manual"]), inp.predictions,
            inp.text_candidate_boxes, clustering=inp.clustering,
            fast_dropped=inp.fast_dropped, ocr_failed=inp.ocr_failed, cfg=cfg,
        )
        result.report_blocks.append(
            format_report(f"[current/manual] {task.pdf_path}", task.page_index, result.manual)
        )

    result.showcase = _showcase(
        ctx.cluster_ocr_results or [], task.showcase_per_page, task.showcase_seed,
    )
    _write_outputs(
        task, gt, ctx.page.meta, converted,
        ctx.ocr_results or [], inp.predictions, cfg,
    )
    return result


def _run_legacy(task: PageTask, gt: LabelSet, cfg: MetricConfig) -> PageResult:
    from rastervec.Evaluation.Evaluate.legacy_adapter import (
        run_archive_pipeline,
        to_cluster_ocr_results,
    )

    t0 = time.perf_counter()
    elements = run_archive_pipeline(
        task.pdf_path, task.page_index,
        enable_raster_pass=task.enable_archive_raster_pass,
    )
    cluster_ocr_results = to_cluster_ocr_results(elements, page_index=task.page_index)
    elapsed = time.perf_counter() - t0

    predictions = predictions_from_cluster_ocr(cluster_ocr_results)
    tcb = text_candidate_boxes(None, cluster_ocr_results)
    by_src = split_labelset_by_source(gt)
    result = PageResult(
        pdf_path=task.pdf_path, page_index=task.page_index, pipeline="legacy",
        total_seconds=elapsed,
    )
    result.auto = evaluate_metrics(
        gt_regions_from_labelset(by_src["auto"]), predictions, tcb, cfg=cfg,
    )
    result.report_blocks.append(
        format_report(f"[legacy/auto]   {task.pdf_path}", task.page_index, result.auto)
    )
    if by_src["manual"].entries:
        result.manual = evaluate_metrics(
            gt_regions_from_labelset(by_src["manual"]), predictions, tcb, cfg=cfg,
        )
        result.report_blocks.append(
            format_report(f"[legacy/manual] {task.pdf_path}", task.page_index, result.manual)
        )

    _write_outputs(
        task, gt, _original_page_meta(task.pdf_path, task.page_index),
        _original_page_bytes(task.pdf_path, task.page_index),
        [c.resolved for c in cluster_ocr_results], predictions, cfg,
    )
    return result


def run_page_task(task: PageTask) -> PageResult:
    """One benchmarked page, end to end. Never raises -- a failure is
    captured into `PageResult.error`."""
    cfg = MetricConfig(iou_edge_min=task.iou_edge_min)
    try:
        gt = _ground_truth(task)
        if task.pipeline == "legacy":
            return _run_legacy(task, gt, cfg)
        return _run_current(task, gt, cfg)
    except Exception as exc:  # noqa: BLE001 -- keep benchmarking the rest
        _LOG.warning("%s page %d failed: %s", task.pdf_path, task.page_index, exc)
        return PageResult(
            pdf_path=task.pdf_path, page_index=task.page_index, pipeline=task.pipeline,
            error=f"{exc}\n{traceback.format_exc()}",
        )


def run_benchmark(
    tasks: list[PageTask], *, workers: int = 1, desc: str = "benchmark",
) -> list[PageResult]:
    """Run `run_page_task` over `tasks` (serial when `workers <= 1`,
    otherwise a spawn process pool), results in input order."""
    from rastervec.Reader.Parallel.pool import run_parallel

    return run_parallel(tasks, run_page_task, workers=workers, desc=desc)
