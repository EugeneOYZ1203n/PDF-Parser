"""Category 6 (vector classification funnel), the pred-vs-GT box overlay
data, and the top-level orchestration (`evaluate_text_metrics`/
`aggregate_text_metrics`/`combine_text_metrics_by_type`) that assembles
every other `metrics_*` module's per-category stats into one
`TextMetricSuiteResult` per page (or merged across pages)."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from rastervec.Evaluation.Evaluate.metrics_core import (
    TEXT_TYPES,
    Bbox,
    GtRegion,
    MetricConfig,
    OverlapGraph,
    Prediction,
    Ratio,
    _NA,
    build_overlap_graphs_by_type,
)
from rastervec.Evaluation.Evaluate.metrics_distributions import (
    BboxAccuracyStats,
    FontSizeDistribution,
    RotationBucketCounts,
    RotationStats,
    bbox_accuracy_stats,
    bbox_accuracy_unclassified,
    font_size_distribution,
    rotation_stats,
    truly_unclassified_pred_indices,
)
from rastervec.Evaluation.Evaluate.metrics_text_overlap import (
    CharOverlapStats,
    TextLabelStats,
    WordOverlapStats,
    char_overlap_stats,
    text_label_stats,
    word_overlap_stats,
)


# --------------------------------------------------------------------------
# Category 6 -- vector classification funnel (native_to_vector, original_vector only)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassificationFunnelStats:
    text_type: str
    n_gt_vectors: int
    n_survived: int
    survival_rate: Ratio


FUNNEL_TEXT_TYPES: tuple[str, ...] = ("native_to_vector", "original_vector")


def classification_funnel_stats(
    gt_vector_signatures: "set[str]", survived_signatures: "set[str]", text_type: str,
) -> ClassificationFunnelStats:
    n_gt = len(gt_vector_signatures)
    if n_gt == 0:
        return ClassificationFunnelStats(
            text_type=text_type, n_gt_vectors=0, n_survived=0, survival_rate=_NA,
        )
    survived = len(gt_vector_signatures & survived_signatures)
    return ClassificationFunnelStats(
        text_type=text_type, n_gt_vectors=n_gt, n_survived=survived,
        survival_rate=Ratio(float(survived), float(n_gt)),
    )


# --------------------------------------------------------------------------
# Pred-vs-GT box overlay (for a visual diff PDF -- data only, no rendering)
# --------------------------------------------------------------------------
MATCH_BOX_COLOR = (0.0, 0.7, 0.0)  # green -- gt and pred that overlap
MISSED_GT_BOX_COLOR = (0.85, 0.0, 0.0)  # red -- gt no prediction reached
SPURIOUS_PRED_BOX_COLOR = (0.95, 0.75, 0.0)  # yellow -- pred over no gt


def overlay_boxes(
    graph: OverlapGraph,
) -> list[tuple[Bbox, tuple[float, float, float]]]:
    """`(bbox, rgb)` pairs for a pred-vs-ground-truth visual diff, straight
    off the same overlap graph the metrics score."""
    boxes: list[tuple[Bbox, tuple[float, float, float]]] = []
    for gi, g in enumerate(graph.gt):
        color = MATCH_BOX_COLOR if graph.gt_has_overlap[gi] else MISSED_GT_BOX_COLOR
        boxes.append((g.bbox, color))
    preds_with_overlap = {e.pred_idx for e in graph.edges}
    for pj, p in enumerate(graph.preds):
        color = MATCH_BOX_COLOR if pj in preds_with_overlap else SPURIOUS_PRED_BOX_COLOR
        boxes.append((p.bbox, color))
    return boxes


# One dash style per text type (PyMuPDF dash strings; None = solid).
_TEXT_TYPE_DASHES: dict[str, "str | None"] = {
    "native_to_vector": "[4 3] 0",
    "original_vector": None,
    "vector_to_raster": "[2 2] 0",
    "original_raster": "[6 2 2 2] 0",
    "native_to_raster": "[1 1] 0",
}
_PRED_DASHES = "[1 2] 0"


def overlay_boxes_by_type(
    graphs_by_type: dict[str, OverlapGraph],
) -> list[tuple[Bbox, tuple[float, float, float], "str | None"]]:
    """`(bbox, rgb, dashes)` for the combined benchmark box overlay -- one
    dash style per text type's GT boxes, predictions drawn once per type
    (dotted), green if they overlap that type's own GT, yellow otherwise."""
    boxes: list[tuple[Bbox, tuple[float, float, float], "str | None"]] = []
    for text_type, graph in graphs_by_type.items():
        gt_dashes = _TEXT_TYPE_DASHES.get(text_type)
        for gi, g in enumerate(graph.gt):
            color = MATCH_BOX_COLOR if graph.gt_has_overlap[gi] else MISSED_GT_BOX_COLOR
            boxes.append((g.bbox, color, gt_dashes))
        preds_with_overlap = {e.pred_idx for e in graph.edges}
        for pj, p in enumerate(graph.preds):
            color = (
                MATCH_BOX_COLOR if pj in preds_with_overlap
                else SPURIOUS_PRED_BOX_COLOR
            )
            boxes.append((p.bbox, color, _PRED_DASHES))
    return boxes


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
@dataclass
class PerTypeTextResult:
    label_stats: TextLabelStats
    char_overlap: CharOverlapStats
    word_overlap: WordOverlapStats
    bbox_accuracy: BboxAccuracyStats
    rotation: RotationStats
    reading_order: "object"  # confusion_metrics.ReadingOrderStats
    confusion: "dict[str, Counter[str]]"
    font_size: FontSizeDistribution
    funnel: "ClassificationFunnelStats | None" = None


@dataclass
class TextMetricSuiteResult:
    by_type: dict[str, PerTypeTextResult]
    bbox_unclassified: BboxAccuracyStats
    # chars of predictions with zero overlap across EVERY text type's graph
    # -- merged once here (not per type) to avoid quadruple-counting the
    # same truly-unclassified prediction across 4 type graphs.
    extra_chars: "Counter[str]" = field(default_factory=Counter)


def combine_text_metrics_by_type(
    results_by_type: "dict[str, TextMetricSuiteResult]",
) -> TextMetricSuiteResult:
    """Combines several `TextMetricSuiteResult`s -- each produced by scoring
    a distinct partition of `TEXT_TYPES` against its own prediction source
    (e.g. a vectorised-PDF pipeline run for `native_to_vector`/
    `original_vector`, a separate rasterised-PDF run for `vector_to_raster`/
    `original_raster`/`native_to_raster`) -- into one. `results_by_type`
    maps each text type to the `TextMetricSuiteResult` that scored it (the
    SAME result object repeated for every type it scored); each type's
    `PerTypeTextResult` is taken from its owning result, and `bbox_
    unclassified`/`extra_chars` (both run-scoped: "predictions with zero
    overlap across every type THAT RUN scored") are summed once per
    distinct result object referenced, not once per type. `spurious_pred_
    area_frac` is summed too (as a fraction of the SAME page area each
    contributing result was computed against -- true whenever every result
    scored the same page, which is the only case this is called for)."""
    by_type = {t: results_by_type[t].by_type[t] for t in TEXT_TYPES if t in results_by_type}
    seen: "set[int]" = set()
    total_spurious = 0
    total_area_frac = 0.0
    any_area_frac = False
    extra_chars: "Counter[str]" = Counter()
    for r in results_by_type.values():
        if id(r) in seen:
            continue
        seen.add(id(r))
        total_spurious += r.bbox_unclassified.spurious_pred_count
        frac = r.bbox_unclassified.spurious_pred_area_frac
        if not math.isnan(frac):
            total_area_frac += frac
            any_area_frac = True
        extra_chars.update(r.extra_chars)
    return TextMetricSuiteResult(
        by_type=by_type,
        bbox_unclassified=BboxAccuracyStats(
            text_type="unclassified", spurious_pred_count=total_spurious,
            spurious_pred_area_frac=total_area_frac if any_area_frac else math.nan,
        ),
        extra_chars=extra_chars,
    )


def evaluate_text_metrics(
    gt_by_type: dict[str, list[GtRegion]],
    entries_by_type: dict[str, list],
    predictions: list[Prediction],
    *,
    gt_vector_signatures_by_type: "dict[str, set[str]] | None" = None,
    survived_signatures: "set[str] | None" = None,
    page_area: float | None = None,
    dpi: float = 300.0,
    cfg: MetricConfig = MetricConfig(),
) -> TextMetricSuiteResult:
    # Local import to avoid a metrics_suite.py <-> confusion_metrics.py
    # import cycle (confusion_metrics imports OverlapGraph/GtRegion from
    # metrics.py, which re-exports from metrics_core.py -- not from here).
    from rastervec.Evaluation.Evaluate.confusion_metrics import (
        confusion_table,
        extra_predicted_chars_for_indices,
        reading_order_stats,
    )

    graphs_by_type = build_overlap_graphs_by_type(gt_by_type, predictions, cfg)
    label_stats_by_type = {s.text_type: s for s in text_label_stats(entries_by_type)}
    gt_vector_signatures_by_type = gt_vector_signatures_by_type or {}
    survived_signatures = survived_signatures or set()

    by_type: dict[str, PerTypeTextResult] = {}
    for text_type in TEXT_TYPES:
        graph = graphs_by_type[text_type]
        funnel = None
        if text_type in FUNNEL_TEXT_TYPES:
            funnel = classification_funnel_stats(
                gt_vector_signatures_by_type.get(text_type, set()),
                survived_signatures, text_type,
            )
        by_type[text_type] = PerTypeTextResult(
            label_stats=label_stats_by_type[text_type],
            char_overlap=char_overlap_stats(graph, text_type),
            word_overlap=word_overlap_stats(graph, text_type),
            bbox_accuracy=bbox_accuracy_stats(graph, text_type),
            rotation=rotation_stats(graph, text_type),
            reading_order=reading_order_stats(graph, text_type),
            confusion=confusion_table(graph),
            font_size=font_size_distribution(graph, text_type, dpi=dpi),
            funnel=funnel,
        )

    extra_chars = extra_predicted_chars_for_indices(
        predictions, truly_unclassified_pred_indices(graphs_by_type),
    )
    return TextMetricSuiteResult(
        by_type=by_type,
        bbox_unclassified=bbox_accuracy_unclassified(graphs_by_type, page_area=page_area),
        extra_chars=extra_chars,
    )


def _agg_ratio(ratios: list[Ratio]) -> Ratio:
    num = den = 0.0
    any_applicable = False
    for r in ratios:
        if not r.applicable:
            continue
        any_applicable = True
        num += r.numerator
        den += r.denominator
    return Ratio(num, den) if any_applicable else _NA


def aggregate_text_metrics(results: list[TextMetricSuiteResult]) -> "TextMetricSuiteResult | None":
    """Micro-averages every table across pages: counts sum, `Ratio`s
    re-derived from summed numerator/denominator (never averaged directly),
    confusion counters merged, rotation buckets summed and mean/RMSE
    recomputed from the pooled per-page diffs (approximated here by
    weighting each page's mean/RMSE by its own `n_localized`, since only
    the aggregates -- not the raw per-gt diffs -- survive into this
    result)."""
    if not results:
        return None

    by_type: dict[str, PerTypeTextResult] = {}
    for text_type in TEXT_TYPES:
        per_type = [r.by_type[text_type] for r in results]

        label_count = sum(p.label_stats.label_count for p in per_type)
        char_count = sum(p.label_stats.char_count for p in per_type)
        word_count = sum(p.label_stats.word_count for p in per_type)
        vector_count = sum(p.label_stats.vector_count for p in per_type)
        label_stats = TextLabelStats(text_type, label_count, char_count, word_count, vector_count)

        char_overlap = CharOverlapStats(
            text_type=text_type,
            matched=sum(p.char_overlap.matched for p in per_type),
            total_gt=sum(p.char_overlap.total_gt for p in per_type),
            unclassified=sum(p.char_overlap.unclassified for p in per_type),
            missing=sum(p.char_overlap.missing for p in per_type),
            precision=_agg_ratio([p.char_overlap.precision for p in per_type]),
            recall=_agg_ratio([p.char_overlap.recall for p in per_type]),
        )
        word_overlap = WordOverlapStats(
            text_type=text_type,
            matched=sum(p.word_overlap.matched for p in per_type),
            total_gt=sum(p.word_overlap.total_gt for p in per_type),
            unclassified=sum(p.word_overlap.unclassified for p in per_type),
            missing=sum(p.word_overlap.missing for p in per_type),
            precision=_agg_ratio([p.word_overlap.precision for p in per_type]),
            recall=_agg_ratio([p.word_overlap.recall for p in per_type]),
        )
        bbox_accuracy = BboxAccuracyStats(
            text_type=text_type,
            mean_iou=_agg_ratio([p.bbox_accuracy.mean_iou for p in per_type]),
            n_gt=sum(p.bbox_accuracy.n_gt for p in per_type),
            n_localized=sum(p.bbox_accuracy.n_localized for p in per_type),
        )

        n_loc_total = sum(p.rotation.n_localized for p in per_type)
        buckets = RotationBucketCounts(
            correct=sum(p.rotation.buckets.correct for p in per_type),
            off_90=sum(p.rotation.buckets.off_90 for p in per_type),
            off_180=sum(p.rotation.buckets.off_180 for p in per_type),
        )
        if n_loc_total:
            mean_error = sum(
                p.rotation.mean_error_deg * p.rotation.n_localized
                for p in per_type if p.rotation.n_localized
            ) / n_loc_total
            mean_sq = sum(
                (p.rotation.rmse_deg ** 2) * p.rotation.n_localized
                for p in per_type if p.rotation.n_localized
            ) / n_loc_total
            rmse = math.sqrt(mean_sq)
        else:
            mean_error = math.nan
            rmse = math.nan
        rotation = RotationStats(
            text_type=text_type, buckets=buckets, n_localized=n_loc_total,
            mean_error_deg=mean_error, rmse_deg=rmse,
        )

        from rastervec.Evaluation.Evaluate.confusion_metrics import (
            aggregate_reading_order_stats,
        )
        reading_order = aggregate_reading_order_stats([p.reading_order for p in per_type])

        merged: dict[str, "Counter[str]"] = {}
        for p in per_type:
            for ch, counter in p.confusion.items():
                merged.setdefault(ch, Counter()).update(counter)

        unit = per_type[0].font_size.unit if per_type else "pt"
        font_size = FontSizeDistribution(
            text_type=text_type, unit=unit,
            all_sizes=tuple(s for p in per_type for s in p.font_size.all_sizes),
            detected_sizes=tuple(s for p in per_type for s in p.font_size.detected_sizes),
        )

        funnel = None
        if text_type in FUNNEL_TEXT_TYPES:
            n_gt_vectors = sum(p.funnel.n_gt_vectors for p in per_type if p.funnel)
            n_survived = sum(p.funnel.n_survived for p in per_type if p.funnel)
            funnel = ClassificationFunnelStats(
                text_type=text_type, n_gt_vectors=n_gt_vectors, n_survived=n_survived,
                survival_rate=(
                    Ratio(float(n_survived), float(n_gt_vectors)) if n_gt_vectors else _NA
                ),
            )

        by_type[text_type] = PerTypeTextResult(
            label_stats=label_stats, char_overlap=char_overlap, word_overlap=word_overlap,
            bbox_accuracy=bbox_accuracy, rotation=rotation, reading_order=reading_order,
            confusion=merged, font_size=font_size, funnel=funnel,
        )

    _area_fracs = [
        r.bbox_unclassified.spurious_pred_area_frac for r in results
        if not math.isnan(r.bbox_unclassified.spurious_pred_area_frac)
    ]
    bbox_unclassified = BboxAccuracyStats(
        text_type="unclassified",
        spurious_pred_count=sum(r.bbox_unclassified.spurious_pred_count for r in results),
        # mean across pages -- not true area-weighted (per-page area isn't
        # carried, only each page's own fraction), an acceptable
        # approximation for the common near-uniform-page-size case.
        spurious_pred_area_frac=(sum(_area_fracs) / len(_area_fracs)) if _area_fracs else math.nan,
    )
    extra_chars: "Counter[str]" = Counter()
    for r in results:
        extra_chars.update(r.extra_chars)
    return TextMetricSuiteResult(
        by_type=by_type, bbox_unclassified=bbox_unclassified, extra_chars=extra_chars,
    )
