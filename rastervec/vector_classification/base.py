"""Shared helpers for the metric-driven classification engine.

`_bbox_of` is the one place that decides "what bbox does this unit have" --
a single `VectorPath` uses its own `.bbox`, a group (`list[VectorPath]`)
uses the union of its members' bboxes -- so every metric and every engine
module (filter_step.py/cluster_step.py/group_step.py) can operate on either
an item or a whole cluster without caring which. `_reconcile_items`/
`_reconcile_groups` are the generalized versions of the old `Vector.
_filter_step`/`_group_filter_step` -- id()-based kept/dropped bookkeeping,
reused by filter_step.py.
"""
from __future__ import annotations

from rastervec.geometry import union_bbox
from rastervec.models import VectorPath


def _bbox_of(unit: "VectorPath | list[VectorPath]") -> tuple[float, float, float, float]:
    if isinstance(unit, VectorPath):
        return unit.bbox
    return union_bbox([p.bbox for p in unit])


def _reconcile_items(
    groups: list[list[VectorPath]], kept_ids: set[int],
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    """A group keeps its still-kept members (dropping one member doesn't
    remove its groupmates); each dropped path becomes its own singleton
    group in the drop bucket."""
    kept_groups: list[list[VectorPath]] = []
    dropped_groups: list[list[VectorPath]] = []
    for g in groups:
        keep_members = [p for p in g if id(p) in kept_ids]
        if keep_members:
            kept_groups.append(keep_members)
        dropped_groups.extend([p] for p in g if id(p) not in kept_ids)
    return kept_groups, dropped_groups


def _reconcile_groups(
    groups: list[list[VectorPath]], kept_groups: list[list[VectorPath]],
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    """`kept_groups` is already the same list objects the caller decided to
    keep, so identity comparison recovers what was dropped."""
    kept_ids = {id(g) for g in kept_groups}
    dropped_groups = [g for g in groups if id(g) not in kept_ids]
    return kept_groups, dropped_groups
