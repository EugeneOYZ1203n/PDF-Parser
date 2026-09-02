"""Benchmark CLI: wires Conversion -> auto_label -> a real full pipeline
run -> the independent metric suite (`metrics.py`) together end-to-end,
over one or more PDF pages, and prints a per-page + aggregate report.

    .venv/Scripts/python.exe -m rastervec.Evaluation.Evaluate.benchmark \
        --pdf path/to.pdf --pages 0,1,2 [--iou-threshold 0.3] [--eval-pdf-dir DIR]

`--eval-pdf-dir`, when given, writes one <stem>_p<page>_eval.pdf per page via
the *legacy* evaluate.render_evaluation_pdf (matched pairs green, unmatched
labels red, unmatched predictions yellow) -- that overlay still uses the old
1:1-match `evaluate_pipeline`; the scored metrics come from `metrics.py`.

Runs the real `Pipeline.STAGES` chain (via `pipeline.run_page_context`)
through OCR (`RenderOCR`/PaddleOCR) -- deliberately the same real,
possibly-slow pipeline run being scored. The first run downloads PaddleOCR's
models. `main()`'s actual PDF/OCR path is a documented manual smoke test only.

`format_report`/`aggregate_results` are the pure, OCR-free parts (formatting /
micro-averaging over already-computed `MetricSuiteResult`s) -- those ARE
unit-tested, see `tests/rastervec/Evaluation/Evaluate/test_benchmark.py`.
"""
from __future__ import annotations

import argparse
import math
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from rastervec.Evaluation.Conversion.conversion import convert_page_to_vector_text
from rastervec.Evaluation.Evaluate.adapters import (
    build_eval_inputs,
    gt_regions_from_labelset,
)
from rastervec.Evaluation.Evaluate.evaluate import (
    DEFAULT_IOU_THRESHOLD,
    evaluate_pipeline,
    render_evaluation_pdf,
)
from rastervec.Evaluation.Evaluate.metrics import (
    DERIVED_F1_FIELDS,
    METRIC_GROUPS,
    MetricConfig,
    MetricSuiteResult,
    aggregate_suite,
    evaluate_metrics,
)
from rastervec.Evaluation.Labelling.auto_label import auto_label_pdf
from rastervec.logging_setup import configure_logging, get_logger
from rastervec.pipeline import run_page_context
from rastervec.Reader.reader import Reader

_LOG = get_logger("benchmark")


def run_one_page(
    pdf_path: str, page_index: int, iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    eval_pdf_path: Path | None = None,
) -> MetricSuiteResult:
    """Ground truth (no pipeline run) -> Conversion -> a real full pipeline
    run (OCR included) -> `metrics.evaluate_metrics`, for one page.
    `iou_threshold` maps onto `MetricConfig.iou_edge_min`. When
    `eval_pdf_path` is given, also writes the legacy
    matched(green)/unmatched-label(red)/unmatched-prediction(yellow) bbox
    overlay there (--eval-pdf-dir)."""
    labels = auto_label_pdf(pdf_path, page_index)
    converted_bytes = convert_page_to_vector_text(pdf_path, page_index)

    with tempfile.TemporaryDirectory() as tmp_dir:
        converted_path = str(Path(tmp_dir) / "converted.pdf")
        Path(converted_path).write_bytes(converted_bytes)

        with Reader(converted_path) as reader:
            ctx = run_page_context(reader, 0)

    inputs = build_eval_inputs(ctx)
    result = evaluate_metrics(
        gt_regions_from_labelset(labels),
        inputs.predictions,
        inputs.text_candidate_boxes,
        clustering=inputs.clustering,
        fast_dropped=inputs.fast_dropped,
        ocr_failed=inputs.ocr_failed,
        cfg=MetricConfig(iou_edge_min=iou_threshold),
    )

    if eval_pdf_path is not None:
        eval_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        legacy = evaluate_pipeline(
            labels, ctx.cluster_ocr_results or [], ctx.drawing_vectors or [],
            iou_threshold=DEFAULT_IOU_THRESHOLD, clustering=ctx.clustering,
            fast_dropped=ctx.fast_dropped, ocr_failed=ctx.ocr_failed,
        )
        eval_pdf_path.write_bytes(render_evaluation_pdf(legacy, ctx.page.meta))

    return result


def _fmt_metric_line(name: str, result: MetricSuiteResult) -> str:
    if name in DERIVED_F1_FIELDS:
        v = result.get(name)
        return f"  {name}: {'n/a' if math.isnan(v) else f'{v:.3f}'}"
    ratio = result.ratios[name]
    v = ratio.value
    val = "n/a" if math.isnan(v) else f"{v:.3f}"
    return f"  {name}: {ratio.numerator:.4g}/{ratio.denominator:.4g}  ({val})"


def format_report(pdf_path: str, page_index: int, result: MetricSuiteResult) -> str:
    """Per-page report -- absolute `numerator/denominator (value)` per metric,
    grouped by dimension, then a diagnostics block."""
    lines = [f"{pdf_path} page {page_index}:"]
    for dimension, names in METRIC_GROUPS:
        lines.append(f"  [{dimension}]")
        for name in names:
            lines.append(_fmt_metric_line(name, result))
    lines.append("  [diagnostics]")
    if result.per_stage_miss_counts:
        lines.append(f"    per_stage_miss_counts: {result.per_stage_miss_counts}")
    c = result.counts
    lines.append(
        f"    counts: gt={c.n_gt} pred={c.n_pred}(nonblank {c.n_pred_nonblank}) "
        f"candidates={c.n_text_candidates} localized={c.n_gt_localized} missed={c.n_gt_missed}"
    )
    return "\n".join(lines)


def aggregate_results(results: list[MetricSuiteResult]) -> MetricSuiteResult | None:
    """Micro-averaged aggregate (`Ratio(sum num, sum den)` per metric) over
    every page's `MetricSuiteResult`. `None` for an empty input."""
    if not results:
        return None
    return aggregate_suite(results)


def format_aggregate(
    result: MetricSuiteResult | None, n_pages: int, *, label: str = "Aggregate",
) -> str:
    if result is None:
        return f"{label}: (no results)"
    lines = [f"{label} (micro-averaged over {n_pages} page-score(s)):"]
    for dimension, names in METRIC_GROUPS:
        lines.append(f"  [{dimension}]")
        for name in names:
            lines.append(_fmt_metric_line(name, result))
    if result.per_stage_miss_counts:
        lines.append(f"  per_stage_miss_counts: {result.per_stage_miss_counts}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Timing statistics -- pure, unit-tested. Used by
# notebooks/benchmark_vector_classification.ipynb to summarize the per-stage
# and per-page wall-clock times a run records in
# `PipelineContext.stage_durations` (see pipeline.py).
# ---------------------------------------------------------------------------

_TIMING_STAT_COLUMNS = ("n", "min", "q1", "median", "mean", "q3", "max")


def distribution_stats(values: Sequence[float]) -> dict:
    """min / Q1 / median / mean / Q3 / max (plus `n`) of `values`. Quartiles
    are linear-interpolated (`numpy.percentile` default). `{}` for an empty
    input rather than raising."""
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    q1, median, q3 = (float(x) for x in np.percentile(arr, [25, 50, 75]))
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "q1": q1,
        "median": median,
        "mean": float(arr.mean()),
        "q3": q3,
        "max": float(arr.max()),
    }


def summarize_stage_timings(
    per_page: list[dict[str, float]], stage_order: list[str],
) -> dict[str, dict]:
    """One `distribution_stats` per stage key (ordered by `stage_order`,
    only keys that appear in at least one page), plus a `"total"` key over
    each page's summed stage time. `{}` for no pages."""
    if not per_page:
        return {}

    summary: dict[str, dict] = {}
    for key in stage_order:
        column = [page[key] for page in per_page if key in page]
        if column:
            summary[key] = distribution_stats(column)
    summary["total"] = distribution_stats([sum(page.values()) for page in per_page])
    return summary


def format_timing_report(
    summary: dict[str, dict], *, title: str = "Stage timing (seconds)",
) -> str:
    """Fixed-width table of a `summarize_stage_timings` result."""
    if not summary:
        return f"{title}\n  (no timing data)"

    name_width = max(len(name) for name in summary)
    header = f"  {'stage':<{name_width}}  " + "  ".join(f"{c:>8}" for c in _TIMING_STAT_COLUMNS)
    lines = [title, header, "  " + "-" * (len(header) - 2)]
    for name, stats in summary.items():
        cells = []
        for c in _TIMING_STAT_COLUMNS:
            value = stats.get(c, 0.0)
            cells.append(f"{int(value):>8}" if c == "n" else f"{value:>8.3f}")
        lines.append(f"  {name:<{name_width}}  " + "  ".join(cells))
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the Vector Classification + OCR pipeline against auto-labelled ground truth."
    )
    parser.add_argument(
        "--pdf", action="append", required=True, help="Path to a PDF (repeatable).",
    )
    parser.add_argument(
        "--pages", default="0", help="Comma-separated 0-based page indices (default: 0).",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=MetricConfig().iou_edge_min,
        help=f"MetricConfig.iou_edge_min -- minimum IoU for a gt<->prediction "
        f"localisation edge (default: {MetricConfig().iou_edge_min}).",
    )
    parser.add_argument(
        "--eval-pdf-dir", type=Path, default=None,
        help="Write one <stem>_p<page>_eval.pdf per page here -- matched pairs in green, "
        "unmatched labels in red, unmatched predictions in yellow (see evaluate.render_evaluation_pdf).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_arg_parser().parse_args(argv)
    pages = [int(p) for p in args.pages.split(",")]

    results: list[MetricSuiteResult] = []
    for pdf_path in args.pdf:
        for page_index in pages:
            _LOG.info("running %s page %d", pdf_path, page_index)
            eval_pdf_path = (
                args.eval_pdf_dir / f"{Path(pdf_path).stem}_p{page_index}_eval.pdf"
                if args.eval_pdf_dir is not None else None
            )
            result = run_one_page(
                pdf_path, page_index, args.iou_threshold, eval_pdf_path=eval_pdf_path,
            )
            results.append(result)
            print(format_report(pdf_path, page_index, result))
            print()

    if len(results) > 1:
        print(format_aggregate(aggregate_results(results), len(results)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
