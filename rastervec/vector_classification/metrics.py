"""Metric registry for the classification engine.

A metric is a small, named, self-registered function -- adding a new one
means adding one dict entry with a compute function, here in this file;
nothing in engine.py/filter_step.py/cluster_step.py/group_step.py or the
debug-app UI needs to change (dropdowns read these dicts' keys directly).

Two kinds of metric:
- SCALAR_METRICS: `compute(unit, ctx, params) -> float`, `unit` is a single
  `VectorPath` (item scope) or `list[VectorPath]` (cluster scope). Used by
  Filter steps directly, and by Cluster steps in "pairwise" mode (split a
  sorted-by-metric sequence wherever the gap exceeds a threshold).
- PAIRWISE_METRICS: `compute(a, b, ctx, params) -> float`, a distance
  between two units of the same kind (both VectorPath, or both groups) --
  smaller means closer/more similar, merged when <= a threshold. Used by
  Cluster steps in "global" mode and by all Group steps.

`resolve_value` implements the outlier/relative-to-aggregate wrapper once,
generically, for every scalar metric: any metric can be requested as its
raw value, or as `raw / aggregate(population)` (aggregate = mean/median),
so no separate ">2x mean"-style metric needs to be written by hand.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from rastervec.geometry import rect_gap
from rastervec.helpers.clustering import _bbox_fully_contains
from rastervec.vector_classification.base import _bbox_of

if TYPE_CHECKING:
    from rastervec.models import Page, VectorPath

Unit = "VectorPath | list[VectorPath]"


@dataclass
class MetricContext:
    page: "Page"
    item_counts: dict[int, int] = field(default_factory=dict)
    item_bboxes: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    label: str
    compute: Callable
    applies_to: tuple[str, ...] = ("item", "cluster")
    extra_params: tuple[str, ...] = ()


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(x1 - x0, 0.0) * max(y1 - y0, 0.0)


def _bbox_dims(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return max(x1 - x0, 0.0), max(y1 - y0, 0.0)


# ------------------------------------------------------------------
# Scalar metrics
# ------------------------------------------------------------------

def _bbox_area_fraction(unit, ctx: MetricContext, params: dict) -> float:
    page_area = ctx.page.meta.width * ctx.page.meta.height
    if page_area <= 0:
        return 0.0
    return _bbox_area(_bbox_of(unit)) / page_area


def _dimension_fraction(unit, ctx: MetricContext, params: dict) -> float:
    page_min = min(ctx.page.meta.width, ctx.page.meta.height)
    if page_min <= 0:
        return 0.0
    w, h = _bbox_dims(_bbox_of(unit))
    return max(w, h) / page_min


def _aspect_ratio(unit, ctx: MetricContext, params: dict) -> float:
    w, h = _bbox_dims(_bbox_of(unit))
    w, h = max(w, 1e-6), max(h, 1e-6)
    return max(w, h) / min(w, h)


def _item_path_count(unit, ctx: MetricContext, params: dict) -> float:
    return float(ctx.item_counts.get(unit.seq, 1))


def _item_bbox_min_side(unit, ctx: MetricContext, params: dict) -> float:
    w, h = _bbox_dims(ctx.item_bboxes.get(unit.seq, unit.bbox))
    return min(w, h)


def _item_bbox_max_side(unit, ctx: MetricContext, params: dict) -> float:
    w, h = _bbox_dims(ctx.item_bboxes.get(unit.seq, unit.bbox))
    return max(w, h)


def _layout_panel_singleton(unit, ctx: MetricContext, params: dict) -> float:
    is_panel = ctx.item_counts.get(unit.seq, 1) == 1 and unit.kind in ("re", "qu")
    return 1.0 if is_panel else 0.0


def _seq_metric(unit, ctx: MetricContext, params: dict) -> float:
    return float(unit.seq)


SCALAR_METRICS: dict[str, MetricSpec] = {
    "bbox_area_fraction": MetricSpec(
        "bbox_area_fraction", "Bbox area fraction of page", _bbox_area_fraction,
    ),
    "dimension_fraction": MetricSpec(
        "dimension_fraction", "Max dimension fraction of page", _dimension_fraction,
    ),
    "aspect_ratio": MetricSpec("aspect_ratio", "Aspect ratio", _aspect_ratio),
    "item_path_count": MetricSpec(
        "item_path_count", "Original item's path count", _item_path_count, applies_to=("item",),
    ),
    "item_bbox_min_side": MetricSpec(
        "item_bbox_min_side", "Original item's bbox min side", _item_bbox_min_side, applies_to=("item",),
    ),
    "item_bbox_max_side": MetricSpec(
        "item_bbox_max_side", "Original item's bbox max side", _item_bbox_max_side, applies_to=("item",),
    ),
    "layout_panel_singleton": MetricSpec(
        "layout_panel_singleton", "Is a lone-item re/qu panel", _layout_panel_singleton, applies_to=("item",),
    ),
    "seq": MetricSpec("seq", "Drawing sequence number", _seq_metric, applies_to=("item",)),
}


# ------------------------------------------------------------------
# Pairwise (distance) metrics
# ------------------------------------------------------------------

def _spatial_gap(a, b, ctx: MetricContext, params: dict) -> float:
    return rect_gap(_bbox_of(a), _bbox_of(b))


def _dimension_similarity(a, b, ctx: MetricContext, params: dict) -> float:
    if params.get("bbox_source") == "item":
        bbox_a = ctx.item_bboxes.get(a.seq, a.bbox)
        bbox_b = ctx.item_bboxes.get(b.seq, b.bbox)
    else:
        bbox_a, bbox_b = _bbox_of(a), _bbox_of(b)
    aw, ah = _bbox_dims(bbox_a)
    bw, bh = _bbox_dims(bbox_b)
    dw = abs(aw - bw) / max(aw, bw, 1e-6)
    dh = abs(ah - bh) / max(ah, bh, 1e-6)
    return max(dw, dh)


def _overlap_tolerance(a, b, ctx: MetricContext, params: dict) -> float:
    bbox_a, bbox_b = _bbox_of(a), _bbox_of(b)
    if _bbox_fully_contains(bbox_a, bbox_b):
        return float("inf")
    return rect_gap(bbox_a, bbox_b)


PAIRWISE_METRICS: dict[str, MetricSpec] = {
    "spatial_gap": MetricSpec("spatial_gap", "Spatial bbox gap", _spatial_gap),
    "dimension_similarity": MetricSpec(
        "dimension_similarity", "Relative dimension difference", _dimension_similarity,
        extra_params=("bbox_source",),
    ),
    "overlap_tolerance": MetricSpec(
        "overlap_tolerance", "Overlap/gap (never merges full containment)", _overlap_tolerance,
    ),
}


# ------------------------------------------------------------------
# Condition evaluation + relative-aggregate wrapper
# ------------------------------------------------------------------

def check_condition(condition: str, value: float, threshold) -> bool:
    if condition == ">":
        return value > threshold
    if condition == ">=":
        return value >= threshold
    if condition == "<":
        return value < threshold
    if condition == "<=":
        return value <= threshold
    if condition == "==":
        return value == threshold
    if condition == "!=":
        return value != threshold
    if condition == "within":
        lo, hi = threshold
        return lo <= value <= hi
    if condition == "outside":
        lo, hi = threshold
        return value < lo or value > hi
    raise ValueError(f"unknown condition: {condition!r}")


def resolve_value(step, unit, population: list, ctx: MetricContext) -> float:
    """`step.metric`'s raw value for `unit`, or (if `step.aggregate` is
    set) that raw value divided by the mean/median of the same metric
    computed over `population` -- the generic outlier/relative-to-mean
    mechanism shared by every metric, so e.g. ">2x mean" is just
    `aggregate="mean", condition="<=", threshold=2.0` on any metric,
    never a metric written specifically for outliers."""
    spec = SCALAR_METRICS[step.metric]
    raw = spec.compute(unit, ctx, step.params)
    if not step.aggregate:
        return raw
    values = [spec.compute(u, ctx, step.params) for u in population]
    agg = statistics.mean(values) if step.aggregate == "mean" else statistics.median(values)
    return raw / agg if agg else float("inf")
