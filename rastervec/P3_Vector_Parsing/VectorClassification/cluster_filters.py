"""Cluster-level step of the Vector Classification pipeline (see
`classify_vectors.py` for the fixed step order). A "cluster" is the final
classification output for one (layer, color) bucket -- see
docs/Glossary.md for the group/cluster distinction.

`cluster_spatial_groups` -- single-linkage spatial merge of the incoming
groups (by bbox gap), constrained: two groups only merge if one has a
"valid" side (see below) within `SPATIAL_SIZE_TOLERANCE` of a valid side
of the other, AND those two sides are roughly parallel (lie along the
same axis -- see below). A group's own short side ("width") always counts
as valid alongside its long side ("length"). "Parallel" means: a group's
length lies along whichever axis (x or y) its bbox's larger extent is on,
and its width along the other axis; two sides are only compared if their
axes match. Every resulting cluster continues downstream as one ordinary
cluster -- member groups are kept *nested* (`list[list[Vector]]` per
cluster) rather than flattened, so a cluster's group/vector tiering is
real structure, not a side-channel `id()`-keyed lineage dict.

Whole-page similarity grouping of text-candidate clusters used to live
here (`group_similar_clusters`) but has moved downstream, past Radon
segmentation -- it now operates on post-Radon `Segment`s (using each
segment's own known precise skew angle for rotation-exact normalization)
instead of pre-Radon clusters via a PCA-based rotation search. See
`OCR/radon.py` / `pipelines/_steps.py`'s segment-similarity step.

All thresholds are passed in by the caller (`classify_vectors.py`'s own
constants), so nothing here is hardcoded.
"""
from __future__ import annotations

from rastervec.commons.helpers.clustering import cluster_spatial
from rastervec.commons.helpers.geometry import dims
from rastervec.commons.models import Vector
from rastervec.P3_Vector_Parsing.VectorClassification.item_filters import bbox_of


def _group_sides(
    bbox: tuple[float, float, float, float],
) -> list[tuple[float, str]]:
    """A group's comparable sides as `(length, axis)` pairs -- `axis` is
    "x" if that side is the bbox's horizontal extent, "y" if vertical.
    Both the bbox's longer extent ("length") and shorter extent ("width")
    are included whenever the group has any extent at all."""
    w, h = dims(bbox)
    if w >= h:
        length, length_axis, width, width_axis = w, "x", h, "y"
    else:
        length, length_axis, width, width_axis = h, "y", w, "x"
    sides = [(length, length_axis)]
    if length > 0:
        sides.append((width, width_axis))
    return sides


def _matched_side_value(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
    size_tolerance: float,
    require_parallel: bool,
) -> float | None:
    """The representative length (average of the two matched sides) of the
    first valid-side pair (see `_group_sides`) of `bbox_a` and `bbox_b`
    found within `size_tolerance` of each other, or `None` if no such pair
    exists. When `require_parallel` is set, a candidate pair only counts if
    the two sides also lie on the same axis. Takes each group's bbox
    directly (not the group itself) so a caller examining many pairs can
    compute each group's bbox once and reuse it, instead of this function
    re-deriving it (`bbox_of` is `union_bbox` over the group's own member
    vectors, O(group size)) on every pair it's asked about."""
    sides_a = _group_sides(bbox_a)
    sides_b = _group_sides(bbox_b)
    for sa, axis_a in sides_a:
        for sb, axis_b in sides_b:
            if require_parallel and axis_a != axis_b:
                continue
            if sa <= 0 or sb <= 0:
                if sa == sb:
                    return 0.0
                continue
            if abs(sa - sb) / max(sa, sb) <= size_tolerance:
                return (sa + sb) / 2.0
    return None


def _any_side_close(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
    size_tolerance: float,
    require_parallel: bool,
) -> bool:
    """Bool-only view of `_matched_side_value` -- True if some valid side
    of `bbox_a` is within `size_tolerance` of some valid side of `bbox_b`
    (subject to `require_parallel`, see `_matched_side_value`)."""
    return _matched_side_value(bbox_a, bbox_b, size_tolerance, require_parallel) is not None


def cluster_spatial_groups(
    groups: list[list[Vector]],
    threshold: float,
    size_tolerance: float,
) -> list[list[list[Vector]]]:
    """Single-linkage spatial merge of the incoming groups (by each
    group's own aggregate bbox), via `helpers.clustering.cluster_spatial`
    reused at the group level, constrained by `_any_side_close` (with
    `require_parallel=True`). Each group's bbox (`bbox_of`, O(group size))
    is computed exactly once up front into `bbox_by_id` and reused both as
    `cluster_spatial`'s own `get_bbox` and inside the `extra_close`
    callback -- `cluster_spatial`'s neighbor search can examine the same
    group against several candidates, and without this cache each of those
    examinations would re-derive both groups' bboxes from scratch. Returns
    `kept`, preserving real group/vector tiering -- each entry is
    `list[list[Vector]]` (this cluster's member groups, unflattened) rather
    than a flat `list[Vector]`."""
    bbox_by_id = {id(g): bbox_of(g) for g in groups}

    def _close_parallel(a: list[Vector], b: list[Vector]) -> bool:
        return _any_side_close(
            bbox_by_id[id(a)], bbox_by_id[id(b)], size_tolerance, require_parallel=True,
        )

    constrained = cluster_spatial(
        groups, get_bbox=lambda g: bbox_by_id[id(g)],
        threshold=threshold, extra_close=_close_parallel,
    )
    return [list(cluster) for cluster in constrained]
