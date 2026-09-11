"""Matplotlib PNG charts for the benchmark folder-diff
(`scripts/pipeline_report_benchmark.py`).

Headless (`Agg` backend). Not imported by the pipeline -- only the benchmark
script and its tests. Two chart kinds:

- `metric_comparison_chart` -- horizontal grouped bars, every metric in
  `METRIC_GROUPS` order (dimension labels between groups), one bar per run.
- `confusion_heatmap` -- the `{auto,manual} x {auto,manual,none}` GT-recall
  confusion matrix, one small heatmap per run.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from rastervec.Evaluation.Evaluate.metrics import (  # noqa: E402
    DERIVED_F1_FIELDS,
    LOWER_IS_BETTER,
    METRIC_GROUPS,
    MetricSuiteResult,
)

_BAR_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#d97706")


def _metric_value(result: MetricSuiteResult, name: str) -> float:
    if name in DERIVED_F1_FIELDS:
        return result.get(name)
    return result.ratios[name].value


def _rows() -> list[tuple[str, str]]:
    """`(dimension, metric_name)` for every metric, in display order."""
    return [(dim, name) for dim, names in METRIC_GROUPS for name in names]


def metric_comparison_chart(
    results_by_run: dict[str, "MetricSuiteResult | None"],
    *,
    title: str,
    path: Path,
) -> None:
    """One horizontal bar per run per metric. `None` result / `nan` value =>
    no bar (gap left in place). `↓` prefix marks `LOWER_IS_BETTER` metrics."""
    rows = _rows()
    labels = [
        ("↓ " if name in LOWER_IS_BETTER else "") + name for _dim, name in rows
    ]
    runs = list(results_by_run)
    n = len(runs)
    y = list(range(len(rows)))
    height = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(9, 0.34 * len(rows) + 1.4))
    for ri, run in enumerate(runs):
        result = results_by_run[run]
        offset = (ri - (n - 1) / 2) * height
        xs, ys = [], []
        for i, (_dim, name) in enumerate(rows):
            v = _metric_value(result, name) if result is not None else float("nan")
            if v == v:  # not nan
                xs.append(v)
                ys.append(i + offset)
        ax.barh(ys, xs, height=height, label=run, color=_BAR_COLORS[ri % len(_BAR_COLORS)])

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.axvline(0, color="#999", lw=0.5)
    # dimension separators
    seen = set()
    for i, (dim, _name) in enumerate(rows):
        if dim not in seen:
            seen.add(dim)
            ax.axhline(i - 0.5, color="#ccc", lw=0.6)
            ax.text(1.01, i, dim, fontsize=6, color="#666", va="center", rotation=90)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


_CONF_ROWS = ("auto", "manual")
_CONF_COLS = ("auto", "manual", "none")


def confusion_heatmap(
    confusion_by_run: dict[str, dict[str, dict[str, int]]],
    *,
    title: str,
    path: Path,
) -> None:
    """One `{auto,manual} x {auto,manual,none}` heatmap per run, side by
    side, cell counts annotated."""
    runs = list(confusion_by_run)
    fig, axes = plt.subplots(1, max(len(runs), 1), figsize=(3.2 * max(len(runs), 1), 3))
    if len(runs) == 1:
        axes = [axes]
    for ax, run in zip(axes, runs):
        conf = confusion_by_run[run]
        grid = [[conf.get(r, {}).get(c, 0) for c in _CONF_COLS] for r in _CONF_ROWS]
        ax.imshow(grid, cmap="Blues", vmin=0)
        ax.set_xticks(range(len(_CONF_COLS)), _CONF_COLS, fontsize=8)
        ax.set_yticks(range(len(_CONF_ROWS)), _CONF_ROWS, fontsize=8)
        ax.set_xlabel("detected as", fontsize=8)
        ax.set_ylabel("actual", fontsize=8)
        ax.set_title(run, fontsize=8)
        for ri in range(len(_CONF_ROWS)):
            for ci in range(len(_CONF_COLS)):
                ax.text(ci, ri, str(grid[ri][ci]), ha="center", va="center", fontsize=9)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
