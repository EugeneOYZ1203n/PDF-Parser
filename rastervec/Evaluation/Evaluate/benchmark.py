"""Benchmark CLI: wires Conversion -> auto_label -> a real full pipeline
run -> the independent metric suite (`metrics.py`) together end-to-end,
over one or more PDF pages and one or more pipeline variants, and prints a
per-page report plus cross-variant accuracy + timing comparison tables.

    .venv/Scripts/python.exe -m rastervec.Evaluation.Evaluate.benchmark \
        --pdf path/to.pdf --pages 0,1,2 [--iou-threshold 0.3] \
        [--reconstruct-dir DIR] [--workers N] [--compute-workers N] \
        [--variants current,current_vectorclassification,legacy]

`--variants` selects which `Evaluation/Evaluate/variants.VARIANTS` to run
and compare (default `DEFAULT_VARIANTS`).

`--reconstruct-dir` (default `outputs/benchmark_cli/reconstructions/`)
writes per page x variant (see
`Reader/Parallel/benchmark_jobs._write_current_outputs`): the ground-truth
/ pipeline text reconstruction, the exact PDF fed to the pipeline, and a
green(matched) / yellow(spurious pred) / red(missed gt) box overlay.

`--workers N` (>1) runs the pages across a spawn process pool (`Reader/
Parallel`); the model caches are warmed once up front so the first run is
safe. `--compute-workers N` (>0) additionally starts a shared
`multiprocessing.Manager`-hosted pool (Pool 2, `fitz`-free) that every page
job dispatches its FAST/OCR work into, regardless of which `--workers`
process runs that page -- see `Reader/Parallel/benchmark_jobs.run_benchmark`.
The real `pipeline.STAGES` chain runs through OCR (PaddleOCR) --
`main()`'s actual PDF/OCR path is a documented manual smoke test only.

`format_report` / `aggregate_results` / `format_aggregate_comparison` /
`format_variant_timing_comparison` are the pure, OCR-free parts (formatting
/ micro-averaging over already-computed `MetricSuiteResult`s) -- those ARE
unit-tested, see `tests/rastervec/Evaluation/Evaluate/test_benchmark.py`.
"""
from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from rastervec.Evaluation.Evaluate.metrics import (
    TEXT_TYPES,
    MetricConfig,
    TextMetricSuiteResult,
    aggregate_text_metrics,
)
from rastervec.Evaluation.Evaluate.vector_metrics import (
    VECTOR_TYPES,
    VectorMetricSuiteResult,
    aggregate_vector_metrics,
)
from rastervec.Evaluation.Evaluate.variants import DEFAULT_VARIANTS, resolve_variant
from rastervec.commons.logging_setup import configure_logging, get_logger
from rastervec.commons.paths import output_dir

_LOG = get_logger("benchmark")


def _fmt_ratio(ratio) -> str:
    v = ratio.value
    val = "n/a" if math.isnan(v) else f"{v:.3f}"
    return f"{ratio.numerator:.4g}/{ratio.denominator:.4g} ({val})"


def format_text_report(pdf_path: str, page_index: int, result: TextMetricSuiteResult) -> str:
    """Per-page text-metric report: one block per text type, categories 1-8."""
    lines = [f"{pdf_path} page {page_index} [text]:"]
    for text_type in TEXT_TYPES:
        r = result.by_type[text_type]
        lines.append(f"  [{text_type}]")
        ls = r.label_stats
        lines.append(
            f"    labels: {ls.label_count} | chars: {ls.char_count} | "
            f"words: {ls.word_count} | vectors: {ls.vector_count}"
        )
        co = r.char_overlap
        lines.append(
            f"    char overlap: {co.matched}/{co.total_gt} matched | "
            f"{co.unclassified} extra unmatched (FP) | {co.missing} missing (FN) | "
            f"precision {_fmt_ratio(co.precision)} | recall {_fmt_ratio(co.recall)}"
        )
        wo = r.word_overlap
        lines.append(
            f"    word overlap: {wo.matched}/{wo.total_gt} matched | "
            f"{wo.unclassified} extra unmatched (FP) | {wo.missing} missing (FN) | "
            f"precision {_fmt_ratio(wo.precision)} | recall {_fmt_ratio(wo.recall)}"
        )
        fs = r.font_size
        lines.append(
            f"    font size ({fs.unit}): all n={len(fs.all_sizes)} | "
            f"detected n={len(fs.detected_sizes)}"
        )
        ba = r.bbox_accuracy
        lines.append(
            f"    bbox mean IoU: {_fmt_ratio(ba.mean_iou)} "
            f"(n_gt={ba.n_gt}, n_localized={ba.n_localized})"
        )
        rot = r.rotation
        lines.append(
            f"    rotation: correct={rot.buckets.correct} off_90={rot.buckets.off_90} "
            f"off_180={rot.buckets.off_180} (n={rot.n_localized}) | "
            f"mean_err={rot.mean_error_deg:.1f}deg | rmse={rot.rmse_deg:.1f}deg"
        )
        if r.funnel is not None:
            f_ = r.funnel
            lines.append(
                f"    classification funnel: {f_.n_survived}/{f_.n_gt_vectors} survived "
                f"({_fmt_ratio(f_.survival_rate)})"
            )
        ro = r.reading_order
        lines.append(
            f"    reading order: in_order={ro.n_in_order}/{ro.n_with_overlap} "
            f"({_fmt_ratio(ro.in_order_rate)}) | edit dist min/max/mean/median = "
            f"{ro.edit_distance_min}/{ro.edit_distance_max}/{ro.edit_distance_mean}/{ro.edit_distance_median}"
        )
    bu = result.bbox_unclassified
    lines.append(
        f"  [unclassified] spurious preds: {bu.spurious_pred_count} "
        f"(area frac {bu.spurious_pred_area_frac})"
    )
    return "\n".join(lines)


def format_confusion_table(result: TextMetricSuiteResult, *, top_n: int = 5) -> str:
    lines = ["OCR confusion characters (top {} replacements per gt char):".format(top_n)]
    for text_type in TEXT_TYPES:
        confusion = result.by_type[text_type].confusion
        if not confusion:
            continue
        lines.append(f"  [{text_type}]")
        for ch, counter in sorted(confusion.items(), key=lambda kv: -sum(kv[1].values())):
            total = sum(counter.values())
            top = counter.most_common(top_n)
            cells = ", ".join(
                f"{repr(r) if r else '(none)'}: {c} ({100 * c / total:.0f}%)" for r, c in top
            )
            lines.append(f"    {ch!r}: {cells}")
    extra = result.extra_chars
    if extra:
        lines.append("Extra predicted characters (no ground truth overlap at all):")
        total = sum(extra.values())
        cells = ", ".join(
            f"{ch!r}: {c} ({100 * c / total:.0f}%)" for ch, c in extra.most_common()
        )
        lines.append(f"  {cells}")
    return "\n".join(lines)


def format_vector_report(pdf_path: str, page_index: int, result: VectorMetricSuiteResult) -> str:
    lines = [f"{pdf_path} page {page_index} [vectors]:"]
    for vector_type in VECTOR_TYPES:
        r = result.by_type[vector_type]
        lines.append(f"  [{vector_type}]")
        lines.append(f"    labels: {r.label_stats.count}")
        cs = r.count_stats
        lines.append(
            f"    count: paired={cs.n_paired} missed={cs.n_missed} spurious={cs.n_spurious} | "
            f"precision {_fmt_ratio(cs.precision)} | recall {_fmt_ratio(cs.recall)}"
        )
        es = r.endpoint_stats
        lines.append(f"    endpoint RMSE: {es.rmse} (n_paired={es.n_paired})")
        for row in r.property_rows:
            if not row.applicable:
                lines.append(f"    property {row.property_name}: n/a")
                continue
            lines.append(
                f"    property {row.property_name} ({row.kind}): {row.metric_value} "
                f"(n={row.n_applicable}, none_vs_none={row.none_vs_none_count})"
            )
    return "\n".join(lines)


def format_report(pdf_path: str, page_index: int, result: TextMetricSuiteResult) -> str:
    """Back-compat name -- text-only report."""
    return format_text_report(pdf_path, page_index, result)


def aggregate_results(results: list[TextMetricSuiteResult]) -> TextMetricSuiteResult | None:
    """Micro-averaged aggregate over every page's `TextMetricSuiteResult`.
    `None` for an empty input."""
    if not results:
        return None
    return aggregate_text_metrics(results)


def format_aggregate(
    result: TextMetricSuiteResult | None, n_pages: int, *, label: str = "Aggregate",
) -> str:
    if result is None:
        return f"{label}: (no results)"
    header = f"{label} (micro-averaged over {n_pages} page-score(s)):"
    return header + "\n" + format_text_report("", 0, result).split(":", 1)[1].lstrip("\n")


# ---------------------------------------------------------------------------
# Timing statistics -- pure, unit-tested. Used by
# notebooks/benchmark_vector_classification.ipynb to summarize the per-stage
# and per-page wall-clock times a run records in
# `PipelineResult.step_durations`.
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


def format_variant_timing_comparison(
    summaries_by_variant: dict[str, dict], *,
    title: str = "Per-stage median wall-clock (seconds) by variant",
) -> str:
    """Side-by-side per-stage median timings, one column per variant, plus
    a `d:<variant>` delta-vs-first column for every variant after the
    first. Each value in `summaries_by_variant` is a
    `summarize_stage_timings` result (may be `{}` -- e.g. the legacy
    variant, which has no per-stage breakdown; its cells show `nan`)."""
    if not summaries_by_variant:
        return f"{title}\n  (no timing data)"

    variants = list(summaries_by_variant)
    base = variants[0]
    stage_keys: list[str] = []
    for summary in summaries_by_variant.values():
        for key in summary:
            if key != "total" and key not in stage_keys:
                stage_keys.append(key)
    stage_keys.append("total")

    name_w = max([len("stage")] + [len(k) for k in stage_keys])
    col_w = max(9, max(len(v) for v in variants) + 2)

    def _median(variant: str, key: str) -> float:
        return summaries_by_variant[variant].get(key, {}).get("median", float("nan"))

    header_cols = [f"{v:>{col_w}}" for v in variants]
    header_cols += [f"{'d:' + v:>{col_w}}" for v in variants[1:]]
    header = f"  {'stage':<{name_w}}  " + "  ".join(header_cols)
    lines = [title, header, "  " + "-" * (len(header) - 2)]
    for key in stage_keys:
        cells = [f"{_median(v, key):>{col_w}.3f}" for v in variants]
        cells += [f"{_median(v, key) - _median(base, key):>+{col_w}.3f}" for v in variants[1:]]
        lines.append(f"  {key:<{name_w}}  " + "  ".join(cells))
    return "\n".join(lines)


def format_aggregate_comparison(
    aggregates_by_variant: dict[str, TextMetricSuiteResult | None], *,
    title: str = "Aggregate text metrics by variant (micro-averaged)",
) -> str:
    """One `format_text_report`-style block per variant. A `None` aggregate
    (a variant that produced no scored page) shows `(no results)`."""
    if not aggregates_by_variant:
        return f"{title}\n  (no results)"
    lines = [title]
    for variant, result in aggregates_by_variant.items():
        lines.append(f"[{variant}]")
        if result is None:
            lines.append("  (no results)")
            continue
        lines.append(format_text_report("", 0, result))
    return "\n".join(lines)


def format_vector_aggregate_comparison(
    aggregates_by_variant: "dict[str, VectorMetricSuiteResult | None]", *,
    title: str = "Aggregate vector metrics by variant (micro-averaged)",
) -> str:
    if not aggregates_by_variant:
        return f"{title}\n  (no results)"
    lines = [title]
    for variant, result in aggregates_by_variant.items():
        lines.append(f"[{variant}]")
        if result is None:
            lines.append("  (no results)")
            continue
        lines.append(format_vector_report("", 0, result))
    return "\n".join(lines)


def format_fast_cluster_comparison(
    stats_by_run: "dict[str, dict | None]", *,
    title: str = "Clusters dropped by FAST (VectorClassification only)",
) -> str:
    """One line per run: how many classification clusters the FAST-filter
    step (`P3_Vector_Parsing/VectorClassification/fast_filter.py`) kept vs.
    dropped to drawing output (`{"total": N, "passed": P}`, summed across
    every page -- see `scripts/benchmark_run_loading.py::
    _load_fast_cluster_stats`). `None` (a non-VectorClassification P3
    backend, the legacy engine, or an older dump with no recorded stats)
    shows `n/a`."""
    if not stats_by_run:
        return f"{title}\n  (no results)"
    lines = [title]
    for run, stats in stats_by_run.items():
        if stats is None:
            lines.append(f"  [{run}] n/a")
            continue
        total, passed = stats["total"], stats["passed"]
        dropped = total - passed
        rate = f"{passed / total:.3f}" if total else "n/a"
        lines.append(f"  [{run}] {passed}/{total} clusters kept ({rate}) -- {dropped} dropped by FAST")
    return "\n".join(lines)


def format_retry_stats_comparison(
    stats_by_run: "dict[str, dict | None]", *,
    title: str = "Blank-recognition retries (VectorClassification only)",
) -> str:
    """One line per run: how many detections needed 0/1/2/3 extra
    +90-degree recognition passes before recovering non-blank text, and how
    many never recovered (`{"0": N, "1": N, "2": N, "3": N, "failed": N}`,
    summed across every page -- see `scripts/benchmark_run_loading.py::
    _load_retry_stats`). `None` (a non-VectorClassification P3 backend, the
    legacy engine, or an older dump with no recorded stats) shows `n/a`.
    Mirrors `format_fast_cluster_comparison` exactly."""
    if not stats_by_run:
        return f"{title}\n  (no results)"
    lines = [title]
    for run, stats in stats_by_run.items():
        if stats is None:
            lines.append(f"  [{run}] n/a")
            continue
        parts = " ".join(f"{k}:{stats.get(k, 0)}" for k in ("0", "1", "2", "3", "failed"))
        lines.append(f"  [{run}] {parts}")
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
        "box-overlay PDFs here (see Reader/Parallel/benchmark_jobs._write_outputs). "
        "Default: outputs/benchmark_cli/reconstructions/.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Run pages across a spawn process pool of this size (>1). Default 1 "
        "(serial). See rastervec.core.parallel.",
    )
    parser.add_argument(
        "--compute-workers", type=int, default=0,
        help="Run FAST tile detection + OCR crop recognition on a shared "
        "multiprocessing.Manager-hosted pool of this size (>0), used by every "
        "page job regardless of which --workers process runs it. Default 0 "
        "(fully local per page, today's behavior). Legacy variant is unaffected.",
    )
    parser.add_argument(
        "--variants", default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated pipeline variant names to run and compare "
        f"(default: {','.join(DEFAULT_VARIANTS)}). See "
        "rastervec.Evaluation.Evaluate.variants.VARIANTS.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    reconstruct_dir = args.reconstruct_dir or output_dir("benchmark_cli", "reconstructions")
    pages = [int(p) for p in args.pages.split(",")]
    variant_names = [name.strip() for name in args.variants.split(",") if name.strip()]
    for name in variant_names:
        try:
            resolve_variant(name)  # validate before any work
        except ValueError as exc:
            parser.error(str(exc))

    from rastervec.pipelines.current import STEP_NAMES as _OLD_STEP_NAMES
    from rastervec.core.parallel.benchmark_jobs import PageTask, run_benchmark

    # Old (legacy-engine-adjacent) step names + the new core.pipeline's
    # phase1/phase2/phase3 -- whichever variant ran, its keys are a subset
    # of this combined order.
    stage_order = [*_OLD_STEP_NAMES, "phase1", "phase2", "phase3"]
    aggregates: "dict[str, TextMetricSuiteResult | None]" = {}
    vector_aggregates: "dict[str, VectorMetricSuiteResult | None]" = {}
    timings: dict[str, dict] = {}

    for name in variant_names:
        tasks = [
            PageTask(
                pdf_path=pdf_path, page_index=page_index, iou_edge_min=args.iou_threshold,
                variant=name,
                reconstruct_dir=str(reconstruct_dir),
                showcase_per_page=0,
            )
            for pdf_path in args.pdf
            for page_index in pages
        ]
        page_results = run_benchmark(
            tasks, workers=args.workers, compute_workers=args.compute_workers, desc=name,
        )

        results: "list[TextMetricSuiteResult]" = []
        vector_results: "list[VectorMetricSuiteResult]" = []
        per_page_timings: list[dict[str, float]] = []
        for pr in page_results:
            if pr.error is not None:
                _LOG.warning("[%s] %s page %d failed: %s", name, pr.pdf_path, pr.page_index, pr.error)
                continue
            for block in pr.report_blocks:
                print(f"[{name}] {block}")
            if pr.text_metrics is not None:
                results.append(pr.text_metrics)
                print(format_text_report(f"[{name}] {pr.pdf_path}", pr.page_index, pr.text_metrics))
                print()
            if pr.vector_metrics is not None:
                vector_results.append(pr.vector_metrics)
                print(format_vector_report(f"[{name}] {pr.pdf_path}", pr.page_index, pr.vector_metrics))
                print()
            per_page_timings.append(
                pr.stage_durations or {"pipeline_total": pr.total_seconds}
            )

        aggregates[name] = aggregate_results(results)
        vector_aggregates[name] = aggregate_vector_metrics(vector_results) if vector_results else None
        timings[name] = summarize_stage_timings(per_page_timings, stage_order)

    print(format_aggregate_comparison(aggregates))
    print()
    print(format_vector_aggregate_comparison(vector_aggregates))
    print()
    print(format_variant_timing_comparison(timings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
