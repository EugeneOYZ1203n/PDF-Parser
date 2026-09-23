"""Matplotlib PNG charts for the benchmark folder-diff
(`scripts/pipeline_report_benchmark.py`).

Headless (`Agg` backend). Not imported by the pipeline -- only the benchmark
scripts and their tests. One chart function per metrics table (see
`Evaluation.Evaluate.metrics`/`vector_metrics`), each taking
`results_by_run: dict[str, <SuiteResult> | None]` -- one bar-group per run.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from rastervec.Evaluation.Evaluate.metrics import (  # noqa: E402
    TEXT_TYPES,
    TextMetricSuiteResult,
)
from rastervec.Evaluation.Evaluate.vector_metrics import (  # noqa: E402
    VECTOR_TYPES,
    VectorMetricSuiteResult,
)

_BAR_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#d97706")


def _grouped_bar(
    categories: list[str], series: dict[str, list[float]], *,
    title: str, ylabel: str, path: Path, ylim: "tuple[float, float] | None" = None,
) -> None:
    """`series`: `{run_name: [value_per_category]}`, `nan` -> gap."""
    runs = list(series)
    n = len(runs)
    x = list(range(len(categories)))
    width = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(categories)), 4))
    for ri, run in enumerate(runs):
        offset = (ri - (n - 1) / 2) * width
        values = series[run]
        xs, ys = [], []
        for i, v in enumerate(values):
            if v == v:  # not nan
                xs.append(i + offset)
                ys.append(v)
        ax.bar(xs, ys, width=width, label=run, color=_BAR_COLORS[ri % len(_BAR_COLORS)])
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def label_description_chart(
    results_by_run: "dict[str, TextMetricSuiteResult | None]", *, title: str, path: Path,
) -> None:
    series = {}
    for run, result in results_by_run.items():
        if result is None:
            series[run] = [float("nan")] * len(TEXT_TYPES)
            continue
        series[run] = [float(result.by_type[t].label_stats.label_count) for t in TEXT_TYPES]
    _grouped_bar(list(TEXT_TYPES), series, title=title, ylabel="label count", path=path)


def char_word_overlap_chart(
    results_by_run: "dict[str, TextMetricSuiteResult | None]", *, field: str,
    title: str, path: Path,
) -> None:
    """`field`: `"char_overlap"` or `"word_overlap"`; plots recall per text type."""
    series = {}
    for run, result in results_by_run.items():
        if result is None:
            series[run] = [float("nan")] * len(TEXT_TYPES)
            continue
        series[run] = [
            getattr(result.by_type[t], field).recall.value for t in TEXT_TYPES
        ]
    _grouped_bar(list(TEXT_TYPES), series, title=title, ylabel="recall", path=path, ylim=(0, 1))


def bbox_accuracy_chart(
    results_by_run: "dict[str, TextMetricSuiteResult | None]", *, title: str, path: Path,
) -> None:
    series = {}
    for run, result in results_by_run.items():
        if result is None:
            series[run] = [float("nan")] * len(TEXT_TYPES)
            continue
        series[run] = [result.by_type[t].bbox_accuracy.mean_iou.value for t in TEXT_TYPES]
    _grouped_bar(list(TEXT_TYPES), series, title=title, ylabel="mean IoU", path=path, ylim=(0, 1))


def rotation_chart(
    results_by_run: "dict[str, TextMetricSuiteResult | None]", *, title: str, path: Path,
) -> None:
    series = {}
    for run, result in results_by_run.items():
        if result is None:
            series[run] = [float("nan")] * len(TEXT_TYPES)
            continue
        rows = []
        for t in TEXT_TYPES:
            rot = result.by_type[t].rotation
            n = rot.n_localized
            rows.append((rot.buckets.correct / n) if n else float("nan"))
        series[run] = rows
    _grouped_bar(list(TEXT_TYPES), series, title=title, ylabel="rotation-correct rate", path=path, ylim=(0, 1))


def funnel_chart(
    results_by_run: "dict[str, TextMetricSuiteResult | None]", *, title: str, path: Path,
) -> None:
    funnel_types = ("native_to_vector", "original_vector")
    series = {}
    for run, result in results_by_run.items():
        if result is None:
            series[run] = [float("nan")] * len(funnel_types)
            continue
        rows = []
        for t in funnel_types:
            f_ = result.by_type[t].funnel
            rows.append(f_.survival_rate.value if f_ is not None else float("nan"))
        series[run] = rows
    _grouped_bar(list(funnel_types), series, title=title, ylabel="survival rate", path=path, ylim=(0, 1))


def reading_order_chart(
    results_by_run: "dict[str, TextMetricSuiteResult | None]", *, title: str, path: Path,
) -> None:
    series = {}
    for run, result in results_by_run.items():
        if result is None:
            series[run] = [float("nan")] * len(TEXT_TYPES)
            continue
        series[run] = [result.by_type[t].reading_order.in_order_rate.value for t in TEXT_TYPES]
    _grouped_bar(list(TEXT_TYPES), series, title=title, ylabel="in-order rate", path=path, ylim=(0, 1))


def font_size_histogram_chart(
    result: "TextMetricSuiteResult | None", text_type: str, *, title: str, path: Path, bins: int = 20,
) -> None:
    """Overlaid histogram: distribution of every GT region's size (bbox
    height, the font-size proxy -- see `metrics.font_size_distribution`)
    against the distribution of just the GT sizes that were actually
    localized by a prediction. Unit (pt for native_to_vector/
    original_vector/vector_to_raster, px for original_raster) comes off
    the result itself."""
    fig, ax = plt.subplots(figsize=(6, 4))
    if result is None or not result.by_type[text_type].font_size.all_sizes:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center")
    else:
        fs = result.by_type[text_type].font_size
        ax.hist(fs.all_sizes, bins=bins, alpha=0.5, label="all GT", color=_BAR_COLORS[0])
        if fs.detected_sizes:
            ax.hist(fs.detected_sizes, bins=bins, alpha=0.5, label="detected GT", color=_BAR_COLORS[1])
        ax.set_xlabel(f"font size ({fs.unit})", fontsize=8)
        ax.set_ylabel("count", fontsize=8)
        ax.legend(fontsize=7)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def extra_chars_table_image(
    result: "TextMetricSuiteResult | None", *, title: str, path: Path, top_n: int = 20,
) -> None:
    """Table image of characters predicted with zero overlapping GT at all
    across EVERY text type (`result.extra_chars`), sorted by count."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axis("off")
    extra = result.extra_chars if result is not None else None
    if not extra:
        ax.text(0.5, 0.5, "(no extra predictions)", ha="center", va="center")
    else:
        rows = extra.most_common(top_n)
        total = sum(extra.values())
        cell_text = [[ch, str(c), f"{100 * c / total:.0f}%"] for ch, c in rows]
        ax.table(cellText=cell_text, colLabels=["char", "count", "%"], loc="center", cellLoc="left")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def timing_chart(summaries_by_run: "dict[str, dict]", *, title: str, path: Path) -> None:
    """One stacked bar per run: mean seconds per page for each leaf timing
    row (`timing.leaf_rows` -- phase3 split into its sub-steps when the
    backend recorded any). Means, not medians, so the segments add up to the
    bar's mean total. `summaries_by_run` values are `timing.
    summarize_timings` results (`{}` -> an empty bar)."""
    from rastervec.Evaluation.Evaluate.timing import leaf_rows

    runs = list(summaries_by_run)
    rows: "list[str]" = []
    for summary in summaries_by_run.values():
        for r in leaf_rows(summary):
            if r not in rows:
                rows.append(r)

    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(runs) + 3), 4.5))
    cmap = plt.get_cmap("tab20")
    bottoms = [0.0] * len(runs)
    for ri, row in enumerate(rows):
        heights = [summaries_by_run[run].get(row, {}).get("mean", 0.0) for run in runs]
        ax.bar(range(len(runs)), heights, bottom=bottoms, width=0.6, label=row, color=cmap(ri % 20))
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    for i, total in enumerate(bottoms):
        ax.text(i, total, f"{total:.2f}s", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(runs, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("mean seconds / page", fontsize=8)
    ax.set_title(title, fontsize=9)
    if rows:
        ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def vector_count_chart(
    results_by_run: "dict[str, VectorMetricSuiteResult | None]", *, title: str, path: Path,
) -> None:
    series = {}
    for run, result in results_by_run.items():
        if result is None:
            series[run] = [float("nan")] * len(VECTOR_TYPES)
            continue
        series[run] = [result.by_type[t].count_stats.recall.value for t in VECTOR_TYPES]
    _grouped_bar(list(VECTOR_TYPES), series, title=title, ylabel="recall", path=path, ylim=(0, 1))


def endpoint_accuracy_chart(
    results_by_run: "dict[str, VectorMetricSuiteResult | None]", *, title: str, path: Path,
) -> None:
    series = {}
    for run, result in results_by_run.items():
        if result is None:
            series[run] = [float("nan")] * len(VECTOR_TYPES)
            continue
        series[run] = [result.by_type[t].endpoint_stats.rmse for t in VECTOR_TYPES]
    _grouped_bar(list(VECTOR_TYPES), series, title=title, ylabel="endpoint RMSE (pt)", path=path)


def property_accuracy_chart(
    results_by_run: "dict[str, VectorMetricSuiteResult | None]", *, property_name: str,
    title: str, path: Path,
) -> None:
    series = {}
    for run, result in results_by_run.items():
        if result is None:
            series[run] = [float("nan")] * len(VECTOR_TYPES)
            continue
        rows = []
        for t in VECTOR_TYPES:
            row = next((r for r in result.by_type[t].property_rows if r.property_name == property_name), None)
            rows.append(row.metric_value if row and row.applicable else float("nan"))
        series[run] = rows
    _grouped_bar(list(VECTOR_TYPES), series, title=title, ylabel=property_name, path=path)
