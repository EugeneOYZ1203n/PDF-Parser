"""Item-level steps of the Vector Classification pipeline (see
rastervec/Vector_Classification/classification.py for the fixed step
order these are run in):

1. `filter_large_items` -- drop vectors whose own bbox's max dimension
   exceeds a fraction of the page's smaller side.
2. `compute_vector_signatures` -- informational, pure pass-through: builds
   a per-signature occurrence count (`vector_signature`) over the current
   population, consumed by later group/cluster steps (and the debug app's
   "color by vector type" view) -- never drops anything itself.

Also home to `bbox_of`, the "union bbox of a group's member vectors" helper
reused by the group- and cluster-level filter modules, and `item_signature`,
a per-Vector-*item* shape signature (distinct from `vector_signature`'s
whole-Vector one) used by `cluster_filters.filter_constant_spacing_clusters`
for hatching/tick-mark pattern detection -- a Vector's own items stay
inspectable internally for scoring even though a Vector is never decomposed
into standalone items anywhere in this pipeline's output. The pure bbox math
(`max_dimension`, `dims`) lives in `helpers/geometry.py`.
"""
from __future__ import annotations

from collections import Counter

from rastervec.helpers.geometry import BBox, item_bbox, item_points, max_dimension, union_bbox
from rastervec.models import Page, Vector

# A whole-Vector signature: one per-item signature (kind + points, translated
# so the *Vector's own bbox origin* sits at the origin) per item, in order --
# a strict generalization of the single-item case (a 1-item Vector's
# signature is a 1-tuple), so two translated copies of the same multi-item
# drawing produce equal signatures.
VectorSignature = tuple[tuple[str, tuple[tuple[int, int], ...]], ...]

# A single Vector-item's own signature, translation-normalized to the item's
# own first point -- used only for item-level pattern detection (constant
# spacing), never as a standalone output unit.
ItemSignature = tuple[str, tuple[tuple[int, int], ...]]


def bbox_of(group: list[Vector]) -> BBox:
    """Union bbox of every member Vector's own bbox."""
    return union_bbox([v.bbox for v in group])


def item_signature(item: tuple, round_px: float) -> ItemSignature:
    """A translation-normalized shape signature for one Vector.items entry:
    `item[0]` (kind) plus every point translated so the item's own first
    point sits at the origin, rounded to the nearest `round_px` grid cell."""
    points = item_points(item)
    if not points:
        return (item[0], ())
    x0, y0 = points[0]
    rel = tuple((round((x - x0) / round_px), round((y - y0) / round_px)) for x, y in points)
    return (item[0], rel)


def vector_signature(v: Vector, round_px: float) -> VectorSignature:
    """A translation-normalized shape signature for a whole Vector: every
    item's own signature, all normalized to the *Vector's* own bbox origin
    (not each item's own first point) so a multi-item drawing's internal
    layout is captured as one comparable unit. Two Vectors that are pure
    translations of each other (same items, same relative layout) round to
    the exact same signature."""
    x0, y0 = v.rect[0], v.rect[1]
    sigs = []
    for item in v.items:
        points = item_points(item)
        rel = tuple((round((x - x0) / round_px), round((y - y0) / round_px)) for x, y in points)
        sigs.append((item[0], rel))
    return tuple(sigs)


def filter_large_items(
    groups: list[list[Vector]], page: Page, max_dimension_fraction: float,
) -> tuple[list[list[Vector]], list[list[Vector]]]:
    """Drops individual vectors whose own bbox's max dimension exceeds
    `max_dimension_fraction` of the page's smaller side -- border/frame
    geometry caught by size instead of item-count."""
    page_min = min(page.meta.width, page.meta.height)
    threshold = max_dimension_fraction * page_min if page_min > 0 else float("inf")

    kept: list[list[Vector]] = []
    dropped: list[list[Vector]] = []
    for g in groups:
        keep = [v for v in g if max_dimension(v.bbox) <= threshold]
        keep_ids = {id(v) for v in keep}
        if keep:
            kept.append(keep)
        dropped.extend([v] for v in g if id(v) not in keep_ids)
    return kept, dropped


def compute_vector_signatures(
    groups: list[list[Vector]], round_px: float,
) -> tuple[list[list[Vector]], dict[VectorSignature, int]]:
    """Informational pass-through: never drops or regroups anything,
    just tallies each Vector's `vector_signature` occurrence count over the
    current population. The returned counts dict is threaded through to
    `remove_duplicate_runs`' run-length check, plus the debug app's "color
    by vector type" view -- both read the exact same counts, computed once
    here."""
    counts: Counter[VectorSignature] = Counter()
    for g in groups:
        for v in g:
            counts[vector_signature(v, round_px)] += 1
    return groups, dict(counts)
