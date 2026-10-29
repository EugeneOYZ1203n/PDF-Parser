"""Cluster-level step of the v2_vector_class classification chain (see
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

All thresholds are passed in by the caller (`classify_vectors.py`'s own
constants), so nothing here is hardcoded. `max_area` (the caller's
`MAX_CLUSTER_PAGE_AREA_FRAC` of the page) caps a cluster's bbox area --
see `helpers.clustering.cluster_spatial`'s `max_union_area`.
"""

from __future__ import annotations

from raster_parser.commons.helpers.clustering import cluster_spatial
from raster_parser.commons.helpers.geometry import dims, union_bbox
from raster_parser.models.vector import Vector


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


def _any_side_close(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
    size_tolerance: float,
    require_parallel: bool,
) -> bool:
    """True if some valid side (see `_group_sides`) of `bbox_a` is within `size_tolerance`
    (relative) of some valid side of `bbox_b`; with `require_parallel`, only sides on the same
    axis count. Takes each group's bbox (not the group) so a caller examining many pairs computes
    each bbox once."""
    for sa, axis_a in _group_sides(bbox_a):
        for sb, axis_b in _group_sides(bbox_b):
            if require_parallel and axis_a != axis_b:
                continue
            if sa <= 0 or sb <= 0:
                if sa == sb:
                    return True
                continue
            if abs(sa - sb) / max(sa, sb) <= size_tolerance:
                return True
    return False


def cluster_spatial_groups(
    groups: list[list[Vector]],
    threshold: float,
    size_tolerance: float,
    max_area: float | None = None,
) -> list[list[list[Vector]]]:
    """Single-linkage spatial merge of the incoming groups (by each
    group's own aggregate bbox), via `helpers.clustering.cluster_spatial`
    reused at the group level, constrained by `_any_side_close` (with
    `require_parallel=True`). Each group's bbox (union of its members', O(group size))
    is computed exactly once up front into `bbox_by_id` and reused both as
    `cluster_spatial`'s own `get_bbox` and inside the `extra_close`
    callback -- `cluster_spatial`'s neighbor search can examine the same
    group against several candidates, and without this cache each of those
    examinations would re-derive both groups' bboxes from scratch. Returns
    `kept`, preserving real group/vector tiering -- each entry is
    `list[list[Vector]]` (this cluster's member groups, unflattened) rather
    than a flat `list[Vector]`."""
    bbox_by_id = {id(g): union_bbox([v.bbox for v in g]) for g in groups}

    def _close_parallel(a: list[Vector], b: list[Vector]) -> bool:
        return _any_side_close(
            bbox_by_id[id(a)],
            bbox_by_id[id(b)],
            size_tolerance,
            require_parallel=True,
        )

    constrained = cluster_spatial(
        groups,
        get_bbox=lambda g: bbox_by_id[id(g)],
        threshold=threshold,
        extra_close=_close_parallel,
        max_union_area=max_area,
    )
    return [list(cluster) for cluster in constrained]
