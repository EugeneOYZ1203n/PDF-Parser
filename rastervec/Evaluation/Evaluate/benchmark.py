"""Benchmark CLI: wires Conversion -> auto_label -> a real full pipeline
run -> evaluate_pipeline together end-to-end, over one or more PDF pages,
and prints a per-page + aggregate report.

    .venv/Scripts/python.exe -m rastervec.Evaluation.Evaluate.benchmark \
        --pdf path/to.pdf --pages 0,1,2 [--iou-threshold 0.3] [--eval-pdf-dir DIR]

`--eval-pdf-dir`, when given, writes one <stem>_p<page>_eval.pdf per page via
evaluate.render_evaluation_pdf -- matched pairs in green, unmatched labels in
red, unmatched predictions in yellow.

Runs the real `Pipeline.STAGES` chain (via `pipeline.run_page_context`)
through OCR (`RenderOCR`/PaddleOCR) -- this is deliberately the same real,
possibly-slow pipeline run being scored, not a stubbed-out one. The first
run downloads PaddleOCR's models. Not part of the automated test suite for
that reason (same reasoning as `manual_label.py`'s Tk UI, and the existing
`RASTERVEC_RUN_OCR_TESTS`-gated convention for OCR-dependent tests) --
`main()`'s actual PDF/OCR path is a documented manual smoke test only.

`format_report`/`aggregate_results` are the pure, OCR-free parts of this
module (formatting/averaging over already-computed `EvaluationResult`s) --
those ARE unit-tested, see `tests/rastervec/Evaluation/Evaluate/
test_benchmark.py`.
"""
from __future__ import annotations

import argparse
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from rastervec.Evaluation.Conversion.conversion import convert_page_to_vector_text
from rastervec.Evaluation.Evaluate.evaluate import (
    DEFAULT_IOU_THRESHOLD,
    EvaluationResult,
    evaluate_pipeline,
    render_evaluation_pdf,
)
from rastervec.Evaluation.Labelling.auto_label import auto_label_pdf
from rastervec.logging_setup import configure_logging, get_logger
from rastervec.pipeline import run_page_context
from rastervec.Reader.reader import Reader

_LOG = get_logger("benchmark")

_NUMERIC_FIELDS = (
    "characters_found_pct",
    "character_accuracy",
    "character_error_rate",
    "rotation_accuracy",
    "bbox_accuracy",
    "classification_precision",
    "classification_recall",
)


def run_one_page(
    pdf_path: str, page_index: int, iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    eval_pdf_path: Path | None = None,
) -> EvaluationResult:
    """Ground truth (no pipeline run) -> Conversion -> a real full pipeline
    run (OCR included) -> evaluate_pipeline, for one page. When
    `eval_pdf_path` is given, also writes render_evaluation_pdf's
    matched(green)/unmatched-label(red)/unmatched-prediction(yellow) bbox
    overlay there (--eval-pdf-dir)."""
    labels = auto_label_pdf(pdf_path, page_index)
    converted_bytes = convert_page_to_vector_text(pdf_path, page_index)

    with tempfile.TemporaryDirectory() as tmp_dir:
        converted_path = str(Path(tmp_dir) / "converted.pdf")
        Path(converted_path).write_bytes(converted_bytes)

        with Reader(converted_path) as reader:
            ctx = run_page_context(reader, 0)

    result = evaluate_pipeline(
        labels,
        ctx.cluster_ocr_results or [],
        ctx.drawing_vectors or [],
        iou_threshold=iou_threshold,
        clustering=ctx.clustering,
        fast_dropped=ctx.fast_dropped,
        ocr_failed=ctx.ocr_failed,
    )

    if eval_pdf_path is not None:
        eval_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        # ctx.page.meta is the converted page's own meta -- Conversion sizes
        # and rotates it to match the source page, so it lines up with both
        # the auto/manual labels (source page space) and the predictions
        # (converted page space, same geometry).
        eval_pdf_path.write_bytes(render_evaluation_pdf(result, ctx.page.meta))

    return result


def format_report(pdf_path: str, page_index: int, result: EvaluationResult) -> str:
    lines = [f"{pdf_path} page {page_index}:"]
    for field_name in _NUMERIC_FIELDS:
        lines.append(f"  {field_name}: {getattr(result, field_name):.3f}")
    lines.append(f"  drawing_vector_count: {result.drawing_vector_count}")
    lines.append(
        f"  matched: {len(result.matched)}  unmatched_labels: "
        f"{len(result.unmatched_labels)}  unmatched_predictions: {result.unmatched_predictions}"
    )
    if result.miss_attributions:
        counts: dict[str, int] = {}
        for miss in result.miss_attributions:
            counts[miss.reason] = counts.get(miss.reason, 0) + 1
        lines.append(f"  miss reasons: {counts}")
    return "\n".join(lines)


def aggregate_results(results: list[EvaluationResult]) -> dict:
    """Mean of each numeric metric plus total miss-attribution counts by
    reason, across every page's EvaluationResult. Empty dict for an empty
    input rather than raising."""
    if not results:
        return {}

    aggregate: dict = {
        field_name: sum(getattr(r, field_name) for r in results) / len(results)
        for field_name in _NUMERIC_FIELDS
    }
    aggregate["drawing_vector_count_total"] = sum(r.drawing_vector_count for r in results)
    aggregate["matched_total"] = sum(len(r.matched) for r in results)
    aggregate["unmatched_labels_total"] = sum(len(r.unmatched_labels) for r in results)
    aggregate["unmatched_predictions_total"] = sum(r.unmatched_predictions for r in results)

    miss_reason_counts: dict[str, int] = {}
    for result in results:
        for miss in result.miss_attributions:
            miss_reason_counts[miss.reason] = miss_reason_counts.get(miss.reason, 0) + 1
    aggregate["miss_reason_counts"] = miss_reason_counts

    return aggregate


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
        "--iou-threshold", type=float, default=DEFAULT_IOU_THRESHOLD,
        help=f"IoU threshold for label<->prediction matching (default: {DEFAULT_IOU_THRESHOLD}).",
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

    results: list[EvaluationResult] = []
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
        print("Aggregate:")
        for key, value in aggregate_results(results).items():
            print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
