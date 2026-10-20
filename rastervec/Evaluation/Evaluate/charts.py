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
from rastervec.Evaluation.Evaluate.liu_dori import FRAG_BUCKETS  # noqa: E402
from rastervec.Evaluation.Evaluate.vector_metrics import VectorMetricSuiteResult  # noqa: E402

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
    if ax.get_legend_handles_labels()[0]:
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


def vector_headline_chart(
    results_by_run: "dict[str, VectorMetricSuiteResult | None]", *, title: str, path: Path,
) -> None:
    """Liu-Dori D_v, 1 - F_v, VRI, pixel D_p, 1 - F_p, PRI and SSIM -- each
    1.0 for a perfect detection."""
    categories = ["D_v", "1 - F_v", "VRI", "D_p", "1 - F_p", "PRI", "SSIM"]
    series = {}
    for run, result in results_by_run.items():
        if result is None:
            series[run] = [float("nan")] * len(categories)
            continue
        ld, px = result.liu_dori, result.pixel
        series[run] = [
            ld.d_v, 1.0 - ld.f_v, ld.vri, px.d_p.value, 1.0 - px.f_p.value, px.pri, result.image.ssim,
        ]
    _grouped_bar(categories, series, title=title, ylabel="score (1 = match)", path=path, ylim=(0, 1))


def vector_components_chart(
    results_by_run: "dict[str, VectorMetricSuiteResult | None]", *, title: str, path: Path,
) -> None:
    """Every component of Q_v: the per-overlap factors, then GT-side and
    pred-side basic / fragmentation / total quality."""
    categories = ["Q_pt", "Q_od", "Q_w", "Q_v(c)", "Q_b(g)", "Q_fr(g)", "Q_v(g)", "Q_b(k)", "Q_fr(k)", "Q_v(k)"]
    series = {}
    for run, result in results_by_run.items():
        if result is None:
            series[run] = [float("nan")] * len(categories)
            continue
        ld = result.liu_dori
        series[run] = [
            ld.q_pt.value, ld.q_od.value, ld.q_w.value, ld.q_v_overlap.value,
            ld.gt.q_b.value, ld.gt.q_fr.value, ld.gt.q_v.value,
            ld.pred.q_b.value, ld.pred.q_fr.value, ld.pred.q_v.value,
        ]
    _grouped_bar(categories, series, title=title, ylabel="quality (1 = perfect)", path=path, ylim=(0, 1))


def fragmentation_chart(
    results_by_run: "dict[str, VectorMetricSuiteResult | None]", *, side: str, title: str, path: Path,
) -> None:
    """Share of GT lines by |D(g)| (`side="gt"`) or of predicted lines by
    |G(k)| (`side="pred"`), bucketed `liu_dori.FRAG_BUCKETS`."""
    categories = list(FRAG_BUCKETS)
    series = {}
    for run, result in results_by_run.items():
        if result is None:
            series[run] = [float("nan")] * len(categories)
            continue
        hist = getattr(result.liu_dori, side).histogram
        total = sum(hist.values())
        series[run] = [hist.get(b, 0) / total if total else float("nan") for b in categories]
    label = "predicted lines per GT line" if side == "gt" else "GT lines per predicted line"
    _grouped_bar(categories, series, title=title, ylabel=f"share of lines ({label})", path=path, ylim=(0, 1))


def vector_quality_chart(
    results_by_run: "dict[str, VectorMetricSuiteResult | None]", *, title: str, path: Path,
) -> None:
    """SSIM, length ratio (pred / GT) and direction-histogram intersection
    -- each 1.0 when the prediction matches GT."""
    categories = ["SSIM", "length ratio", "direction intersection"]
    series = {}
    for run, result in results_by_run.items():
        if result is None:
            series[run] = [float("nan")] * len(categories)
            continue
        series[run] = [result.image.ssim, result.length.ratio.value, result.direction.intersection]
    _grouped_bar(categories, series, title=title, ylabel="score (1 = match)", path=path)


def direction_histogram_chart(
    results_by_run: "dict[str, VectorMetricSuiteResult | None]", *, title: str, path: Path,
) -> None:
    """Length-weighted direction histograms (each normalised to 1): GT once,
    then one line per run."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    gt_drawn = False
    for ri, (run, result) in enumerate(results_by_run.items()):
        if result is None:
            continue
        bins = len(result.direction.gt_hist)
        centers = [(i + 0.5) * 180.0 / bins for i in range(bins)]
        if not gt_drawn:
            gt = list(result.direction.gt_hist)
            total = sum(gt)
            if total > 0:
                ax.bar(centers, [v / total for v in gt], width=180.0 / bins, color="#9ca3af", label="GT")
            gt_drawn = True
        pred = list(result.direction.pred_hist)
        total = sum(pred)
        if total > 0:
            ax.step(centers, [v / total for v in pred], where="mid", label=run,
                    color=_BAR_COLORS[ri % len(_BAR_COLORS)])
    ax.set_xlim(0, 180)
    ax.set_xlabel("direction (deg, mod 180)", fontsize=8)
    ax.set_ylabel("length share", fontsize=8)
    ax.set_title(title, fontsize=9)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
