"""Plain-text tables for `scripts/pipeline_report_benchmark.py`'s `benchmark.txt` (the
grep/diff-friendly companion to `report.html`). Pure: formatting only."""

from __future__ import annotations

import math

from raster_parser.evaluation.evaluate.metrics import TEXT_TYPES, TextMetricSuiteResult


def fmt_ratio(ratio) -> str:
    v = ratio.value
    val = "n/a" if math.isnan(v) else f"{v:.3f}"
    return f"{ratio.numerator:.4g}/{ratio.denominator:.4g} ({val})"


def format_text_report(result: TextMetricSuiteResult) -> str:
    """One block per text type: labels, char / word overlap, bbox IoU, rotation."""
    lines = []
    for text_type in TEXT_TYPES:
        r = result.by_type[text_type]
        lines.append(f"  [{text_type}]")
        ls = r.label_stats
        lines.append(
            f"    labels: {ls.label_count} | chars: {ls.char_count} | words: {ls.word_count}"
        )
        for name, ov in (("char", r.char_overlap), ("word", r.word_overlap)):
            lines.append(
                f"    {name} overlap: {ov.matched}/{ov.total_gt} matched | "
                f"{ov.unclassified} extra unmatched (FP) | {ov.missing} missing (FN) | "
                f"precision {fmt_ratio(ov.precision)} | recall {fmt_ratio(ov.recall)}"
            )
        ba = r.bbox_accuracy
        lines.append(
            f"    bbox mean IoU: {fmt_ratio(ba.mean_iou)} (n_gt={ba.n_gt}, n_localized={ba.n_localized})"
        )
        rot = r.rotation
        lines.append(
            f"    rotation: correct={rot.buckets.correct} off_90={rot.buckets.off_90} "
            f"off_180={rot.buckets.off_180} (n={rot.n_localized}) | "
            f"mean_err={rot.mean_error_deg:.1f}deg | rmse={rot.rmse_deg:.1f}deg"
        )
    bu = result.bbox_unclassified
    lines.append(
        f"  [unclassified] spurious preds: {bu.spurious_pred_count} (area frac {bu.spurious_pred_area_frac})"
    )
    return "\n".join(lines)


def format_aggregate_comparison(
    aggregates_by_run: "dict[str, TextMetricSuiteResult | None]", *, title: str
) -> str:
    """One `format_text_report` block per run; a run with no scored page shows `(no results)`."""
    lines = [title]
    for run, result in aggregates_by_run.items():
        lines.append(f"[{run}]")
        lines.append("  (no results)" if result is None else format_text_report(result))
    return "\n".join(lines)


def format_confusion_table(result: TextMetricSuiteResult, *, top_n: int = 5) -> str:
    lines = [f"OCR confusion characters (top {top_n} replacements per gt char):"]
    for text_type in TEXT_TYPES:
        confusion = result.by_type[text_type].confusion
        if not confusion:
            continue
        lines.append(f"  [{text_type}]")
        for ch, counter in sorted(confusion.items(), key=lambda kv: -sum(kv[1].values())):
            total = sum(counter.values())
            cells = ", ".join(
                f"{repr(r) if r else '(none)'}: {c} ({100 * c / total:.0f}%)"
                for r, c in counter.most_common(top_n)
            )
            lines.append(f"    {ch!r}: {cells}")
    extra = result.extra_chars
    if extra:
        lines.append("Extra predicted characters (no ground truth overlap at all):")
        total = sum(extra.values())
        lines.append(
            "  "
            + ", ".join(f"{ch!r}: {c} ({100 * c / total:.0f}%)" for ch, c in extra.most_common())
        )
    return "\n".join(lines)


def format_run_timing_comparison(summaries_by_run: "dict[str, dict]", *, title: str) -> str:
    """Median seconds per row, one column per run, plus a `d:<run>` delta-vs-first column for
    every later run. Values are `timing.summarize_timings` results (`{}` shows `nan`)."""
    if not summaries_by_run:
        return f"{title}\n  (no timing data)"
    runs = list(summaries_by_run)
    keys: list[str] = []
    for summary in summaries_by_run.values():
        keys += [k for k in summary if k not in keys]
    name_w = max([len("step")] + [len(k) for k in keys])
    col_w = max(9, max(len(r) for r in runs) + 2)

    def _median(run: str, key: str) -> float:
        return summaries_by_run[run].get(key, {}).get("median", math.nan)

    header = f"  {'step':<{name_w}}  " + "  ".join(
        [f"{r:>{col_w}}" for r in runs] + [f"{'d:' + r:>{col_w}}" for r in runs[1:]]
    )
    lines = [title, header, "  " + "-" * (len(header) - 2)]
    for key in keys:
        cells = [f"{_median(r, key):>{col_w}.3f}" for r in runs]
        cells += [f"{_median(r, key) - _median(runs[0], key):>+{col_w}.3f}" for r in runs[1:]]
        lines.append(f"  {key:<{name_w}}  " + "  ".join(cells))
    return "\n".join(lines)
