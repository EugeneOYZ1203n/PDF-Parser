"""Item-level steps of the Vector Classification pipeline (see
rastervec/Vector_Classification/classification.py for the fixed step
order these are run in):

1. `filter_large_items` -- drop items whose own bbox's max dimension
   exceeds a fraction of the page's smaller side.
2. `compute_vector_signatures` -- informational, pure pass-through: builds
   a per-signature occurrence count (`vector_signature`) over the current
   population, consumed by later group/cluster steps (and the debug app's
   "color by vector type" view) -- never drops anything itself.

Also home to `bbox_of`, the "union bbox of a group's member paths" helper
reused by the group- and cluster-level filter modules. The pure bbox math
(`max_dimension`, `dims`) lives in `helpers/geometry.py`.
"""
from __future__ import annotations

from collections import Counter

from rastervec.helpers.geometry import BBox, max_dimension, union_bbox
from rastervec.models import Page, VectorPath

VectorSignature = tuple[str, tuple[tuple[int, int], ...]]


def bbox_of(group: list[VectorPath]) -> BBox:
    """Union bbox of every member path's own bbox."""
    return union_bbox([p.bbox for p in group])


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
        keep = [p for p in g if max_dimension(p.bbox) <= threshold]
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
