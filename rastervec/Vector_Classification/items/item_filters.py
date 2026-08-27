"""Item-level steps of the Vector Classification pipeline (see
rastervec/Vector_Classification/classification.py for the fixed step
order these are run in):

1. `filter_large_items` -- drop items whose own bbox's max dimension
   exceeds a fraction of the page's smaller side.
2. `compute_vector_signatures` -- informational, pure pass-through: builds
   a per-signature occurrence count (`vector_signature`) over the current
   population, consumed by later group/cluster steps (and the debug app's
   "color by vector type" view) -- never drops anything itself.

Also home to `_bbox_of`/`_max_dimension`/`_dims`, the shared bbox-geometry
helpers reused by the group- and cluster-level filter modules -- these
operate on the same `list[VectorPath]` unit those higher-level modules
call a "group"/"cluster", but the math itself has no group/cluster
semantics of its own, so it lives at this, the lowest level.
"""
from __future__ import annotations

from collections import Counter

from rastervec.helpers.geometry import union_bbox
from rastervec.models import Page, VectorPath

VectorSignature = tuple[str, tuple[tuple[int, int], ...]]


def _bbox_of(group: list[VectorPath]) -> tuple[float, float, float, float]:
    return union_bbox([p.bbox for p in group])


def _max_dimension(bbox: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(x1 - x0, 0.0, y1 - y0)


def _dims(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (x1 - x0, y1 - y0)


def vector_signature(path: VectorPath, round_px: float) -> VectorSignature:
    """A translation-normalized shape signature: `path.kind` plus every
    point translated so the first point sits at the origin, each
    coordinate rounded to the nearest `round_px` grid cell. Two paths that
    are pure translations of each other (same shape, same size, same
    rotation) round to the exact same signature; a rotated, rescaled, or
    differently-shaped path does not -- this captures shape+size+rotation
    together without a separate angle computation."""
    if not path.points:
        return (path.kind, ())
    x0, y0 = path.points[0]
    rel = tuple(
        (round((x - x0) / round_px), round((y - y0) / round_px))
        for x, y in path.points
    )
    return (path.kind, rel)


def filter_large_items(
    groups: list[list[VectorPath]], page: Page, max_dimension_fraction: float,
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    """Drops individual paths whose own bbox's max dimension exceeds
    `max_dimension_fraction` of the page's smaller side -- border/frame
    geometry caught by size instead of item-count."""
    page_min = min(page.meta.width, page.meta.height)
    threshold = max_dimension_fraction * page_min if page_min > 0 else float("inf")

    kept: list[list[VectorPath]] = []
    dropped: list[list[VectorPath]] = []
    for g in groups:
        keep = [p for p in g if _max_dimension(p.bbox) <= threshold]
        keep_ids = {id(p) for p in keep}
        if keep:
            kept.append(keep)
        dropped.extend([p] for p in g if id(p) not in keep_ids)
    return kept, dropped


def compute_vector_signatures(
    groups: list[list[VectorPath]], round_px: float,
) -> tuple[list[list[VectorPath]], dict[VectorSignature, int]]:
    """Informational pass-through: never drops or regroups anything,
    just tallies each path's `vector_signature` occurrence count over the
    current population. The returned counts dict is threaded through to
    `remove_duplicate_runs`' run-length check and `compute_group_stats`,
    plus the debug app's "color by vector type" view -- all three read the
    exact same counts, computed once here."""
    counts: Counter[VectorSignature] = Counter()
    for g in groups:
        for p in g:
            counts[vector_signature(p, round_px)] += 1
    return groups, dict(counts)
