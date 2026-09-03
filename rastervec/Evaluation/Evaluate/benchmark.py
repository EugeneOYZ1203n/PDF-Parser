"""Benchmark CLI: wires Conversion -> auto_label -> a real full pipeline
run -> the independent metric suite (`metrics.py`) together end-to-end,
over one or more PDF pages, and prints a per-page + aggregate report.

    .venv/Scripts/python.exe -m rastervec.Evaluation.Evaluate.benchmark \
        --pdf path/to.pdf --pages 0,1,2 [--iou-threshold 0.3] \
        [--reconstruct-dir DIR] [--workers N]

`--reconstruct-dir`, when given, writes per page (see
`Reader/Parallel/benchmark_jobs._write_outputs`): the ground-truth /
pipeline text reconstruction, the exact PDF fed to the pipeline, and a
green(matched) / yellow(spurious pred) / red(missed gt) box overlay.

`--workers N` (>1) runs the pages across a spawn process pool (`Reader/
Parallel`); the model caches are warmed once up front so the first run is
safe. The real `Pipeline.STAGES` chain runs through OCR (PaddleOCR) --
`main()`'s actual PDF/OCR path is a documented manual smoke test only.

`format_report`/`aggregate_results` are the pure, OCR-free parts (formatting /
micro-averaging over already-computed `MetricSuiteResult`s) -- those ARE
unit-tested, see `tests/rastervec/Evaluation/Evaluate/test_benchmark.py`.
"""
from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from rastervec.Evaluation.Evaluate.metrics import (
    DERIVED_F1_FIELDS,
    METRIC_GROUPS,
    MetricConfig,
    MetricSuiteResult,
    aggregate_suite,
)
from rastervec.logging_setup import configure_logging, get_logger

_LOG = get_logger("benchmark")


def run_one_page(
    pdf_path: str, page_index: int,
    iou_threshold: float = MetricConfig().iou_edge_min,
    reconstruct_dir: Path | None = None,
) -> MetricSuiteResult:
    """Ground truth (no pipeline run) -> `convert_page_text_only` -> a real
    full pipeline run (OCR included) -> `metrics.evaluate_metrics`, for one
    page's auto labels. `iou_threshold` maps onto `MetricConfig.iou_edge_min`.
    Delegates to `Reader/Parallel/benchmark_jobs.run_page_task`."""
    from rastervec.Reader.Parallel.benchmark_jobs import PageTask, run_page_task

    result = run_page_task(PageTask(
        pdf_path=pdf_path, page_index=page_index, iou_edge_min=iou_threshold,
        reconstruct_dir=str(reconstruct_dir) if reconstruct_dir else None,
        showcase_per_page=0,
    ))
    if result.error is not None:
        raise RuntimeError(result.error)
    if result.auto is None:
        raise RuntimeError(
            "auto run produced no result: " + " | ".join(result.report_blocks)
        )
    return result.auto


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
        "--reconstruct-dir", type=Path, default=None,
        help="Write per-page reconstruction / pipeline-input / green-yellow-red "
        "box-overlay PDFs here (see Reader/Parallel/benchmark_jobs._write_outputs).",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Run pages across a spawn process pool of this size (>1). Default 1 "
        "(serial). See rastervec.Reader.Parallel.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_arg_parser().parse_args(argv)
    pages = [int(p) for p in args.pages.split(",")]

    from rastervec.Reader.Parallel.benchmark_jobs import PageTask, run_benchmark

    tasks = [
        PageTask(
            pdf_path=pdf_path, page_index=page_index, iou_edge_min=args.iou_threshold,
            reconstruct_dir=str(args.reconstruct_dir) if args.reconstruct_dir else None,
            showcase_per_page=0,
        )
        for pdf_path in args.pdf
        for page_index in pages
    ]
    page_results = run_benchmark(tasks, workers=args.workers, desc="benchmark")

    results: list[MetricSuiteResult] = []
    for pr in page_results:
        if pr.error is not None:
            _LOG.warning("%s page %d failed: %s", pr.pdf_path, pr.page_index, pr.error)
            continue
        results.append(pr.auto)
        print(format_report(pr.pdf_path, pr.page_index, pr.auto))
        print()

    if len(results) > 1:
        print(format_aggregate(aggregate_results(results), len(results)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
