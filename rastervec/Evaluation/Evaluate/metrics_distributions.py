"""Font-size distribution, category 4 (text bbox accuracy), and category 5
(rotation accuracy) -- three independent per-`OverlapGraph` reductions."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from rastervec.commons.helpers.geometry import bbox_area, bbox_iou, union_bbox
from rastervec.Evaluation.Evaluate.metrics_core import Bbox, OverlapGraph, Ratio, _NA

# --------------------------------------------------------------------------
# Font-size distribution -- GT size histogram vs. detected-GT size histogram
# --------------------------------------------------------------------------
# LabelEntry carries no font_size field (native/vector/raster labels all
# reduce to a bbox + text), so bbox height is used as the size proxy for
# every text type -- the same measurement `native_label_pdf`'s own GT bbox
# already carries, no label-schema change needed. `native_to_vector`/
# `original_vector`/`vector_to_raster` report it in PDF points (their bbox
# is already page-space pt); `original_raster` converts pt -> px via
# `dpi / 72` (raster labelling's own default dpi, see `master_label.py`).
_TEXT_TYPE_SIZE_UNIT: dict[str, str] = {
    "native_to_vector": "pt",
    "original_vector": "pt",
    "vector_to_raster": "pt",
    "original_raster": "px",
    "native_to_raster": "pt",  # native label bbox, still page-space pt
}


@dataclass(frozen=True)
class FontSizeDistribution:
    text_type: str
    unit: str  # "pt" | "px"
    all_sizes: "tuple[float, ...]" = ()
    detected_sizes: "tuple[float, ...]" = ()  # sizes of gt that were localized


def _gt_size(bbox: Bbox) -> float:
    return bbox[3] - bbox[1]  # bbox height, the font-size proxy


def font_size_distribution(
    graph: OverlapGraph, text_type: str, *, dpi: float = 300.0,
) -> FontSizeDistribution:
    unit = _TEXT_TYPE_SIZE_UNIT[text_type]
    scale = (dpi / 72.0) if unit == "px" else 1.0
    all_sizes = tuple(_gt_size(g.bbox) * scale for g in graph.gt)
    detected_sizes = tuple(
        _gt_size(graph.gt[gi].bbox) * scale for gi in graph.localized_gt_idxs
    )
    return FontSizeDistribution(
        text_type=text_type, unit=unit, all_sizes=all_sizes, detected_sizes=detected_sizes,
    )


# --------------------------------------------------------------------------
# Category 4 -- text bbox accuracy
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BboxAccuracyStats:
    text_type: str
    mean_iou: Ratio = _NA
    n_gt: int = 0
    n_localized: int = 0
    spurious_pred_count: int = 0
    spurious_pred_area_frac: float = math.nan


def per_gt_union_pred_iou_mean(graph: OverlapGraph) -> Ratio:
    if not graph.gt:
        return _NA
    total = 0.0
    for gi, g in enumerate(graph.gt):
        assigned = graph.assigned_preds_by_gt[gi]
        if not assigned:
            continue
        union = union_bbox([graph.preds[pj].bbox for pj in assigned])
        total += bbox_iou(g.bbox, union)
    return Ratio(total, float(len(graph.gt)))


def bbox_accuracy_stats(graph: OverlapGraph, text_type: str) -> BboxAccuracyStats:
    return BboxAccuracyStats(
        text_type=text_type,
        mean_iou=per_gt_union_pred_iou_mean(graph),
        n_gt=len(graph.gt),
        n_localized=len(graph.localized_gt_idxs),
    )


def truly_unclassified_pred_indices(graphs_by_type: dict[str, OverlapGraph]) -> list[int]:
    """Indices (into any one graph's `.preds`, since `build_overlap_graphs_
    by_type` shares one predication filter across all 4 type graphs, so
    their `.preds` lists line up 1:1) of predictions with zero overlap
    across EVERY text type's graph -- i.e. genuinely unclassified, not just
    unclassified with respect to one type. Shared by `bbox_accuracy_
    unclassified` and `evaluate_text_metrics`'s top-level `extra_chars`."""
    graphs = list(graphs_by_type.values())
    if not graphs:
        return []
    n_preds = len(graphs[0].preds)
    return [pj for pj in range(n_preds) if all(not g.edges_by_pred[pj] for g in graphs)]


def bbox_accuracy_unclassified(
    graphs_by_type: dict[str, OverlapGraph], *, page_area: float | None = None,
) -> BboxAccuracyStats:
    """Predictions with zero overlap across EVERY text type's graph.
    `graphs_by_type` must come from `build_overlap_graphs_by_type` so
    `graph.preds` indices line up across types."""
    graphs = list(graphs_by_type.values())
    spurious_idxs = truly_unclassified_pred_indices(graphs_by_type)
    if not graphs:
        return BboxAccuracyStats(text_type="unclassified")
    total_area = sum(bbox_area(graphs[0].preds[pj].bbox) for pj in spurious_idxs)
    area_frac = (total_area / page_area) if page_area else math.nan
    return BboxAccuracyStats(
        text_type="unclassified",
        spurious_pred_count=len(spurious_idxs),
        spurious_pred_area_frac=area_frac,
    )


# --------------------------------------------------------------------------
# Category 5 -- rotation accuracy
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RotationBucketCounts:
    correct: int = 0
    off_90: int = 0
    off_180: int = 0


@dataclass(frozen=True)
class RotationStats:
    text_type: str
    buckets: RotationBucketCounts = field(default_factory=RotationBucketCounts)
    n_localized: int = 0
    mean_error_deg: float = math.nan
    rmse_deg: float = math.nan


def _rotation_vote(graph: OverlapGraph, gi: int) -> int:
    votes: "Counter[int]" = Counter()
    best_iou: dict[int, float] = {}
    for pj in graph.assigned_preds_by_gt[gi]:
        rot = graph.preds[pj].rotation
        votes[rot] += 1
        best_iou[rot] = max(best_iou.get(rot, -1.0), graph.iou(gi, pj))
    top = max(votes.values())
    winners = [r for r, c in votes.items() if c == top]
    return max(winners, key=lambda r: best_iou[r])


def _circular_rotation_diff(a: int, b: int) -> float:
    raw = abs(a - b) % 360
    return raw if raw <= 180 else 360 - raw


def rotation_stats(graph: OverlapGraph, text_type: str) -> RotationStats:
    localized = graph.localized_gt_idxs
    if not localized:
        return RotationStats(text_type=text_type)
    diffs = []
    for gi in localized:
        predicted = _rotation_vote(graph, gi)
        diffs.append(_circular_rotation_diff(predicted, graph.gt[gi].expected_rotation))
    buckets = RotationBucketCounts(
        correct=sum(1 for d in diffs if round(d / 90) * 90 == 0),
        off_90=sum(1 for d in diffs if round(d / 90) * 90 == 90),
        off_180=sum(1 for d in diffs if round(d / 90) * 90 == 180),
    )
    mean_error = sum(diffs) / len(diffs)
    rmse = math.sqrt(sum(d * d for d in diffs) / len(diffs))
    return RotationStats(
        text_type=text_type, buckets=buckets, n_localized=len(localized),
        mean_error_deg=mean_error, rmse_deg=rmse,
    )
