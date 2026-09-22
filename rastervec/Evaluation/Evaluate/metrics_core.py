"""Shared types + the overlap graph every other `metrics_*` module scores
over: `MetricConfig`/`Ratio`/`GtRegion`/`Prediction`/`OverlapEdge`/
`OverlapGraph`, `build_overlap_graph(s)`, and the handful of small helpers
(`f1_from`, `_region_concat_hyp`, ...) that don't belong to any one metric
category. See `metrics.py`'s own module docstring for the overall design
rationale (why an independent-reduction `OverlapGraph` instead of a greedy
1:1 match)."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from rastervec.commons.helpers.geometry import bbox_area, bbox_intersection_area

Bbox = tuple[float, float, float, float]

TEXT_TYPES: tuple[str, ...] = (
    "native_to_vector", "original_vector", "vector_to_raster", "original_raster",
    "native_to_raster",
)


# --------------------------------------------------------------------------
# Config + value types
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MetricConfig:
    """Thresholds for the overlap graph. Tune per benchmark run if a
    dataset's granularity needs it."""

    iou_edge_min: float = 0.10
    """Minimum IoU for a (gt, prediction) edge to count as a localisation
    (fallback assignment)."""
    coverage_tau: float = 0.5
    """A prediction is *assigned* to a gt (N:1) when this fraction or more
    of the prediction's area lies inside that gt."""


@dataclass(frozen=True)
class Ratio:
    """An absolute page-level count pair. `value` is the metric; aggregation
    sums numerators and denominators separately (see module docstring)."""

    numerator: float
    denominator: float

    @property
    def value(self) -> float:
        if self.denominator == 0 or math.isnan(self.denominator):
            return math.nan
        return self.numerator / self.denominator

    @property
    def applicable(self) -> bool:
        return not (self.denominator == 0 or math.isnan(self.denominator))


_NA = Ratio(0.0, math.nan)  # "not applicable" -- excluded from aggregates


@dataclass(frozen=True)
class GtRegion:
    page_index: int
    bbox: Bbox
    text: str
    expected_rotation: int = 0
    text_type: str = ""  # one of TEXT_TYPES, set by adapters.gt_regions_by_text_type


@dataclass(frozen=True)
class Prediction:
    text: str
    bbox: Bbox
    rotation: int
    reached_ocr: bool = True
    ocr_blank: bool = False
    source_cluster_id: int = 0


# --------------------------------------------------------------------------
# The overlap graph
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class OverlapEdge:
    gt_idx: int
    pred_idx: int  # index into OverlapGraph.preds (non-blank predictions)
    inter_area: float
    iou: float
    gt_coverage: float  # inter / area(gt)
    pred_coverage: float  # inter / area(pred)


@dataclass
class OverlapGraph:
    gt: list[GtRegion]
    preds: list[Prediction]  # non-blank predictions only
    n_pred_total: int
    edges: list[OverlapEdge]
    edges_by_gt: list[list[OverlapEdge]]
    edges_by_pred: list[list[OverlapEdge]]
    assigned_preds_by_gt: list[list[int]]
    overlapping_preds_by_gt: list[list[int]]
    gt_has_overlap: list[bool]
    localized_gt_idxs: list[int]
    missed_gt_idxs: list[int]
    _iou_by_gt_pred: dict[tuple[int, int], float]

    def iou(self, gt_idx: int, pred_idx: int) -> float:
        return self._iou_by_gt_pred.get((gt_idx, pred_idx), 0.0)


def build_overlap_graph(
    gt_regions: list[GtRegion],
    predictions: list[Prediction],
    cfg: MetricConfig = MetricConfig(),
) -> OverlapGraph:
    preds = [p for p in predictions if not p.ocr_blank and p.text.strip()]

    edges: list[OverlapEdge] = []
    iou_by_gt_pred: dict[tuple[int, int], float] = {}
    for gi, g in enumerate(gt_regions):
        g_area = bbox_area(g.bbox)
        for pj, p in enumerate(preds):
            inter = bbox_intersection_area(g.bbox, p.bbox)
            if inter <= 0.0:
                continue
            p_area = bbox_area(p.bbox)
            union = g_area + p_area - inter
            iou = inter / union if union > 0 else 0.0
            edge = OverlapEdge(
                gt_idx=gi,
                pred_idx=pj,
                inter_area=inter,
                iou=iou,
                gt_coverage=(inter / g_area) if g_area > 0 else 0.0,
                pred_coverage=(inter / p_area) if p_area > 0 else 0.0,
            )
            edges.append(edge)
            iou_by_gt_pred[(gi, pj)] = iou

    edges_by_gt: list[list[OverlapEdge]] = [[] for _ in gt_regions]
    edges_by_pred: list[list[OverlapEdge]] = [[] for _ in preds]
    for e in edges:
        edges_by_gt[e.gt_idx].append(e)
        edges_by_pred[e.pred_idx].append(e)
    for lst in edges_by_gt:
        lst.sort(key=lambda e: e.iou, reverse=True)
    for lst in edges_by_pred:
        lst.sort(key=lambda e: e.iou, reverse=True)

    assigned_preds_by_gt: list[list[int]] = []
    overlapping_preds_by_gt: list[list[int]] = []
    for gi in range(len(gt_regions)):
        gt_edges = edges_by_gt[gi]
        overlapping_preds_by_gt.append([e.pred_idx for e in gt_edges])
        assigned = [e.pred_idx for e in gt_edges if e.pred_coverage >= cfg.coverage_tau]
        if not assigned and gt_edges and gt_edges[0].iou >= cfg.iou_edge_min:
            assigned = [gt_edges[0].pred_idx]
        assigned_preds_by_gt.append(assigned)

    gt_has_overlap = [bool(edges_by_gt[gi]) for gi in range(len(gt_regions))]
    localized_gt_idxs = [gi for gi, a in enumerate(assigned_preds_by_gt) if a]
    missed_gt_idxs = [gi for gi, a in enumerate(assigned_preds_by_gt) if not a]

    return OverlapGraph(
        gt=list(gt_regions),
        preds=preds,
        n_pred_total=len(predictions),
        edges=edges,
        edges_by_gt=edges_by_gt,
        edges_by_pred=edges_by_pred,
        assigned_preds_by_gt=assigned_preds_by_gt,
        overlapping_preds_by_gt=overlapping_preds_by_gt,
        gt_has_overlap=gt_has_overlap,
        localized_gt_idxs=localized_gt_idxs,
        missed_gt_idxs=missed_gt_idxs,
        _iou_by_gt_pred=iou_by_gt_pred,
    )


def build_overlap_graphs_by_type(
    gt_by_type: dict[str, list[GtRegion]],
    predictions: list[Prediction],
    cfg: MetricConfig = MetricConfig(),
) -> dict[str, OverlapGraph]:
    """One `OverlapGraph` per text type, all built from the SAME
    `predictions` list -- `build_overlap_graph`'s non-blank filter is the
    same predicate regardless of gt, so `graph.preds` indices line up
    across every type's graph (used by `bbox_accuracy_unclassified` to spot
    predictions unclaimed by ANY text type)."""
    return {
        text_type: build_overlap_graph(gt_by_type.get(text_type, []), predictions, cfg)
        for text_type in TEXT_TYPES
    }


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def _multiset_overlap(a: "Counter[str]", b: "Counter[str]") -> int:
    return sum((a & b).values())


def f1_from(recall: Ratio, precision: Ratio) -> float:
    r, p = recall.value, precision.value
    if math.isnan(r) or math.isnan(p) or (r + p) == 0:
        return math.nan
    return 2 * r * p / (r + p)


def _reading_order_key(bbox: Bbox) -> tuple[float, float]:
    """Top-to-bottom, then left-to-right -- default/legacy key, kept for
    `label_overlays.py`. `confusion_metrics._reading_order_sort_key` is the
    rotation-aware key used by category 7's word-order metric."""
    return (bbox[1], bbox[0])


def _region_concat_hyp(graph: OverlapGraph, pred_idxs: list[int]) -> str:
    """The overlapping predictions' text concatenated in reading order."""
    ordered = sorted(pred_idxs, key=lambda pj: _reading_order_key(graph.preds[pj].bbox))
    return " ".join(graph.preds[pj].text for pj in ordered)
