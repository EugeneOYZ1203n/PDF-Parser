"""Matplotlib PNG charts for `scripts/pipeline_report_benchmark.py`.

Headless (`Agg` backend). Not imported by the pipeline -- only the benchmark script and its
tests; matplotlib comes from the `report` dependency group. One chart per metrics table, each
taking `results_by_run: dict[str, TextMetricSuiteResult | None]` -- one bar group per run.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from raster_parser.evaluation.evaluate.metrics import TEXT_TYPES, TextMetricSuiteResult  # noqa: E402
from raster_parser.evaluation.evaluate.timing import leaf_rows  # noqa: E402

_BAR_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#d97706")


def _grouped_bar(
    categories: list[str],
    series: dict[str, list[float]],
    *,
    title: str,
    ylabel: str,
    path: Path,
    ylim: "tuple[float, float] | None" = None,
) -> None:
    """`series`: `{run_name: [value_per_category]}`, `nan` -> gap."""
    runs = list(series)
    n = len(runs)
    width = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(categories)), 4))
    for ri, run in enumerate(runs):
        offset = (ri - (n - 1) / 2) * width
        xs = [i + offset for i, v in enumerate(series[run]) if v == v]
        ys = [v for v in series[run] if v == v]
        ax.bar(xs, ys, width=width, label=run, color=_BAR_COLORS[ri % len(_BAR_COLORS)])
    ax.set_xticks(list(range(len(categories))))
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


def _per_type(results_by_run, value) -> dict[str, list[float]]:
    return {
        run: [float("nan")] * len(TEXT_TYPES)
        if result is None
        else [value(result.by_type[t]) for t in TEXT_TYPES]
        for run, result in results_by_run.items()
    }


def label_description_chart(
    results_by_run: "dict[str, TextMetricSuiteResult | None]", *, title: str, path: Path
) -> None:
    series = _per_type(results_by_run, lambda r: float(r.label_stats.label_count))
    _grouped_bar(list(TEXT_TYPES), series, title=title, ylabel="label count", path=path)


def char_word_overlap_chart(
    results_by_run: "dict[str, TextMetricSuiteResult | None]",
    *,
    field: str,
    title: str,
    path: Path,
) -> None:
    """`field`: `"char_overlap"` or `"word_overlap"`; plots recall per text type."""
    series = _per_type(results_by_run, lambda r: getattr(r, field).recall.value)
    _grouped_bar(list(TEXT_TYPES), series, title=title, ylabel="recall", path=path, ylim=(0, 1))


def bbox_accuracy_chart(
    results_by_run: "dict[str, TextMetricSuiteResult | None]", *, title: str, path: Path
) -> None:
    series = _per_type(results_by_run, lambda r: r.bbox_accuracy.mean_iou.value)
    _grouped_bar(list(TEXT_TYPES), series, title=title, ylabel="mean IoU", path=path, ylim=(0, 1))


def rotation_chart(
    results_by_run: "dict[str, TextMetricSuiteResult | None]", *, title: str, path: Path
) -> None:
    def _correct(r) -> float:
        n = r.rotation.n_localized
        return r.rotation.buckets.correct / n if n else float("nan")

    series = _per_type(results_by_run, _correct)
    _grouped_bar(
        list(TEXT_TYPES),
        series,
        title=title,
        ylabel="rotation-correct rate",
        path=path,
        ylim=(0, 1),
    )


def timing_chart(summaries_by_run: "dict[str, dict]", *, title: str, path: Path) -> None:
    """One stacked bar per run: mean seconds per page for each phase (`timing.leaf_rows`).
    Means, not medians, so the segments add up to the bar's mean total."""
    runs = list(summaries_by_run)
    rows: list[str] = []
    for summary in summaries_by_run.values():
        rows += [r for r in leaf_rows(summary) if r not in rows]

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
