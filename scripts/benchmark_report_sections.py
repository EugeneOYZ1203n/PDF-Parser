"""Builds the Text/Vector sections of `pipeline_report_benchmark.py`'s HTML
report from a `{run_name: aggregate_result}` mapping."""
from __future__ import annotations

import math

from rastervec.Evaluation.Evaluate.benchmark import _fmt_ratio
from rastervec.Evaluation.Evaluate.html_report import ReportBuilder
from rastervec.Evaluation.Evaluate.metrics import TEXT_TYPES, TextMetricSuiteResult
from rastervec.Evaluation.Evaluate.vector_metrics import VECTOR_TYPES, VectorMetricSuiteResult


def _fmt_property_rows(agg_by_run: "dict[str, VectorMetricSuiteResult | None]") -> "tuple[list[str], list[list[str]]]":
    headers = ["vector_type", "property"] + list(agg_by_run)
    rows: "list[list[str]]" = []
    for t in VECTOR_TYPES:
        names: "list[str]" = []
        for res in agg_by_run.values():
            if res is not None:
                names = [r.property_name for r in res.by_type[t].property_rows]
                break
        for name in names:
            row = [t, name]
            for res in agg_by_run.values():
                if res is None:
                    row.append("n/a")
                    continue
                r = next((rr for rr in res.by_type[t].property_rows if rr.property_name == name), None)
                if r is None or not r.applicable:
                    row.append("n/a")
                else:
                    row.append(f"{r.kind}={r.metric_value:.3f} (n={r.n_applicable})")
            rows.append(row)
    return headers, rows


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "n/a"


def _fmt_rotation(rot) -> str:
    b, n = rot.buckets, rot.n_localized
    return (
        f"correct {b.correct} ({_pct(b.correct, n)}) | off90 {b.off_90} ({_pct(b.off_90, n)}) | "
        f"off180 {b.off_180} ({_pct(b.off_180, n)}) | error {_pct(b.off_90 + b.off_180, n)} "
        f"(n={n}) | mean_err={rot.mean_error_deg:.1f} | rmse={rot.rmse_deg:.1f}"
    )


def _fmt_char_stats(cs, *, top_n: int = 5) -> str:
    """One per-char table cell: every gt-char outcome as `n (x% of total)`,
    the top-`top_n` misread replacements, and the raw inserted count."""
    t = cs.total
    if not t:
        return f"total 0 | inserted {cs.inserted}"
    top = ", ".join(f"{r!r}: {c}" for r, c in cs.replacements.most_common(top_n))
    return (
        f"total {t} | detected {cs.detected} ({_pct(cs.detected, t)}) | "
        f"dropped {cs.dropped} ({_pct(cs.dropped, t)}) | "
        f"unreached {cs.unreached} ({_pct(cs.unreached, t)}) | "
        f"misclassified {cs.misclassified} ({_pct(cs.misclassified, t)})"
        + (f" → {top}" if top else "")
        + f" | inserted {cs.inserted}"
    )


def char_order(agg_by_run: "dict[str, TextMetricSuiteResult | None]", text_type: str) -> "list[str]":
    """Every char in any run's `char_stats` for `text_type`, worst error rate
    first (ties: more gt occurrences first). A char is ranked by the first
    run that has gt occurrences of it; chars with none anywhere
    (insertion-only) go last."""
    stats_by_char: dict = {}
    for res in agg_by_run.values():
        if res is None:
            continue
        for ch, cs in res.by_type[text_type].char_stats.items():
            prev = stats_by_char.get(ch)
            if prev is None or (not prev.total and cs.total):
                stats_by_char[ch] = cs

    def key(ch: str):
        cs = stats_by_char[ch]
        rate = cs.error_rate
        return (math.isnan(rate), -(0.0 if math.isnan(rate) else rate), -cs.total, ch)

    return sorted(stats_by_char, key=key)


def _add_text_sections(
    builder: ReportBuilder, agg_by_run: "dict[str, TextMetricSuiteResult | None]",
    char_examples_by_run: "dict[str, dict] | None" = None,
) -> None:
    """`char_examples_by_run` (`{run: benchmark_examples._collect_char_examples
    output}`) adds a per-character example gallery under each text type's
    per-char table, in the table's own row order."""
    builder.add_group_header("Text")

    headers = ["text_type"] + list(agg_by_run)
    rows = []
    for t in TEXT_TYPES:
        row = [t]
        for res in agg_by_run.values():
            if res is None:
                row.append("n/a")
                continue
            ls = res.by_type[t].label_stats
            row.append(f"{ls.label_count} labels | {ls.char_count} chars | {ls.word_count} words | {ls.vector_count} vectors")
        rows.append(row)
    builder.add_text_subsection("Label description", headers, rows)

    for field, name in (("char_overlap", "Char overlap"), ("word_overlap", "Word overlap")):
        rows = []
        for t in TEXT_TYPES:
            row = [t]
            for res in agg_by_run.values():
                if res is None:
                    row.append("n/a")
                    continue
                co = getattr(res.by_type[t], field)
                row.append(
                    f"matched {co.matched}/{co.total_gt} | unclassified {co.unclassified} | "
                    f"missing {co.missing} | P {_fmt_ratio(co.precision)} | R {_fmt_ratio(co.recall)}"
                )
            rows.append(row)
        builder.add_text_subsection(name, headers, rows)

    rows = []
    for t in TEXT_TYPES:
        row = [t]
        for res in agg_by_run.values():
            if res is None:
                row.append("n/a")
                continue
            ba = res.by_type[t].bbox_accuracy
            row.append(f"mean IoU {_fmt_ratio(ba.mean_iou)} (n_gt={ba.n_gt} n_localized={ba.n_localized})")
        rows.append(row)
    row = ["unclassified"]
    for res in agg_by_run.values():
        if res is None:
            row.append("n/a")
            continue
        bu = res.bbox_unclassified
        row.append(f"spurious preds {bu.spurious_pred_count} (area frac {bu.spurious_pred_area_frac})")
    rows.append(row)
    builder.add_text_subsection("Bbox accuracy", headers, rows)

    rows = []
    for t in TEXT_TYPES:
        row = [t]
        for res in agg_by_run.values():
            if res is None:
                row.append("n/a")
                continue
            row.append(_fmt_rotation(res.by_type[t].rotation))
        rows.append(row)
    builder.add_text_subsection("Rotation accuracy", headers, rows)

    rows = []
    for t in ("native_to_vector", "original_vector"):
        row = [t]
        for res in agg_by_run.values():
            if res is None:
                row.append("n/a")
                continue
            f_ = res.by_type[t].funnel
            row.append(f"{f_.n_survived}/{f_.n_gt_vectors} ({_fmt_ratio(f_.survival_rate)})" if f_ else "n/a")
        rows.append(row)
    builder.add_text_subsection("Vector classification funnel", ["text_type"] + list(agg_by_run), rows)

    rows = []
    for t in TEXT_TYPES:
        row = [t]
        for res in agg_by_run.values():
            if res is None:
                row.append("n/a")
                continue
            ro = res.by_type[t].reading_order
            row.append(
                f"in_order {ro.n_in_order}/{ro.n_with_overlap} ({_fmt_ratio(ro.in_order_rate)}) | "
                f"edit min/max/mean/median {ro.edit_distance_min}/{ro.edit_distance_max}/"
                f"{ro.edit_distance_mean}/{ro.edit_distance_median}"
            )
        rows.append(row)
    builder.add_text_subsection("Reading order accuracy", headers, rows)

    for t in TEXT_TYPES:
        order = char_order(agg_by_run, t)
        c_rows = []
        for ch in order:
            row = [repr(ch)]
            for res in agg_by_run.values():
                cs = None if res is None else res.by_type[t].char_stats.get(ch)
                row.append("" if cs is None else _fmt_char_stats(cs))
            c_rows.append(row)
        builder.add_text_subsection(
            f"Per-character OCR accuracy -- {t}", ["gt_char"] + list(agg_by_run), c_rows,
        )
        for run, examples in (char_examples_by_run or {}).items():
            by_char = examples.get(t, {})
            builder.add_char_examples(run, t, [(ch, by_char.get(ch, {})) for ch in order])

    chars = set()
    for res in agg_by_run.values():
        if res is not None:
            chars.update(res.extra_chars.keys())
    ec_rows = []
    for ch in sorted(chars):
        row = [repr(ch)]
        for res in agg_by_run.values():
            if res is None:
                row.append("")
                continue
            c = res.extra_chars.get(ch, 0)
            total = sum(res.extra_chars.values()) or 1
            row.append(f"{c} ({100 * c / total:.0f}%)" if c else "")
        ec_rows.append(row)
    builder.add_text_subsection("Extra predicted characters (no GT overlap at all)", ["char"] + list(agg_by_run), ec_rows)


def _add_vector_sections(builder: ReportBuilder, agg_by_run: "dict[str, VectorMetricSuiteResult | None]") -> None:
    if not any(v is not None for v in agg_by_run.values()):
        return
    builder.add_group_header("Vector")

    headers = ["vector_type"] + list(agg_by_run)
    rows = []
    for t in VECTOR_TYPES:
        row = [t]
        for res in agg_by_run.values():
            row.append(str(res.by_type[t].label_stats.count) if res else "n/a")
        rows.append(row)
    builder.add_vector_subsection("Label description", headers, rows)

    rows = []
    for t in VECTOR_TYPES:
        row = [t]
        for res in agg_by_run.values():
            if res is None:
                row.append("n/a")
                continue
            cs = res.by_type[t].count_stats
            row.append(
                f"paired={cs.n_paired} missed={cs.n_missed} spurious={cs.n_spurious} | "
                f"P {_fmt_ratio(cs.precision)} | R {_fmt_ratio(cs.recall)}"
            )
        rows.append(row)
    builder.add_vector_subsection("Count accuracy", headers, rows)

    rows = []
    for t in VECTOR_TYPES:
        row = [t]
        for res in agg_by_run.values():
            if res is None:
                row.append("n/a")
                continue
            es = res.by_type[t].endpoint_stats
            row.append(f"RMSE {es.rmse} (n_paired={es.n_paired})")
        rows.append(row)
    builder.add_vector_subsection("Endpoint accuracy", headers, rows)

    p_headers, p_rows = _fmt_property_rows(agg_by_run)
    builder.add_vector_subsection("Property accuracy", p_headers, p_rows)
