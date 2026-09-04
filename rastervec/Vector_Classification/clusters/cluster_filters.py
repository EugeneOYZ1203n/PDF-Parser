"""Cluster-level steps of the Vector Classification pipeline (see
rastervec/Vector_Classification/classification.py for the fixed step
order these are run in). A "cluster" is the final classification output
for one (layer, color) bucket -- see Glossary.md for the group/cluster
distinction.

6. `cluster_spatial_groups` -- single-linkage spatial merge of the
   remaining groups (by bbox gap), constrained: two groups only merge if
   one has a "valid" side (see below) within `SPATIAL_SIZE_TOLERANCE` of a
   valid side of the other, AND those two sides are roughly parallel (lie
   along the same axis -- see below). A group's own short side ("width")
   always counts as valid alongside its long side ("length").
   "Parallel" means: a group's length lies along whichever axis (x or y)
   its bbox's larger extent is on, and its width along the other axis;
   two sides are only compared if their axes match. Every resulting
   cluster continues downstream as one ordinary cluster, its member groups
   flattened together -- nothing is dropped or split here. Also returns
   two debug-only categories, and `lineage` -- for every merged cluster,
   which of this step's own input groups it's composed of.
7. `filter_mixed_fill_rule_clusters` -- drop a whole cluster if its
   members don't all share the same `fill_rule` ("f"/"fs"/"s") -- a
   cluster mixing paint styles is drawing content, not text.
9. `filter_perimeter_only_clusters` -- drop whole clusters whose members
   never reach into the cluster's own shrunk-in center region, i.e. every
   member sits in the bbox's perimeter margin (border/ring geometry, not
   text).
10. `filter_density_clusters` -- splits each cluster's own bbox into a
    `DENSITY_DEFAULT_GRID_SIZE`-per-axis grid, clamped so each cell's
    side stays within `[DENSITY_MIN_CELL_PX, DENSITY_MAX_CELL_PX]`; drops
    the whole cluster if more than `DENSITY_MAX_EMPTY_FRACTION` of the
    cells have no member touching them -- too sparse to be text.
11. `filter_constant_spacing_clusters` -- splits each cluster into
    same-shape sub-groups and checks spacing *within* each sub-group
    separately; the whole cluster is dropped if the patterned sub-groups'
    members together make up at least `PATTERN_FRACTION_THRESHOLD` of the
    cluster's total members -- a regular repeated pattern (hatching, tick
    marks), not text.
12. `filter_low_variety_clusters` -- drop a whole cluster if it contains
    fewer distinct shape signatures (from `compute_group_stats`) than a
    log-scale ramp requires for its member count -- real text has a
    variety of glyph shapes; a cluster built from only a handful of
    repeated shapes is drawing content, not a text candidate.

`group_similar_clusters` (not one of the numbered filter steps; called
separately by pipeline.py's unique_clusters stage) does whole-page
similarity grouping of the final text-candidate clusters -- see
Glossary.md's "similarity group" entry.

All thresholds are passed in by the caller (`classification.py`'s own
constants), so nothing here is hardcoded.
"""
from __future__ import annotations

from collections import defaultdict
from math import atan2, ceil, cos, hypot, log, sin

from rastervec.helpers.clustering import cluster_spatial
from rastervec.helpers.geometry import bboxes_intersect, dims, union_bbox
from rastervec.helpers.iterutils import partition
from rastervec.models import VectorPath
from rastervec.Vector_Classification.groups.group_filters import GroupStats
from rastervec.Vector_Classification.items.item_filters import (
    VectorSignature,
    bbox_of,
    vector_signature,
)


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
    a: list[VectorPath],
    b: list[VectorPath],
    size_tolerance: float,
    require_parallel: bool,
) -> float | None:
    """The representative length (average of the two matched sides) of the
    first valid-side pair (see `_group_sides`) of `a` and `b` found within
    `size_tolerance` of each other, or `None` if no such pair exists. When
    `require_parallel` is set, a candidate pair only counts if the two
    sides also lie on the same axis."""
    sides_a = _group_sides(bbox_of(a))
    sides_b = _group_sides(bbox_of(b))
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
    a: list[VectorPath],
    b: list[VectorPath],
    size_tolerance: float,
    require_parallel: bool,
) -> bool:
    """Bool-only view of `_matched_side_value` -- True if some valid side
    of `a` is within `size_tolerance` of some valid side of `b` (subject
    to `require_parallel`, see `_matched_side_value`)."""
    return _matched_side_value(a, b, size_tolerance, require_parallel) is not None


def cluster_spatial_groups(
    groups: list[list[VectorPath]],
    threshold: float,
    size_tolerance: float,
) -> tuple[
    list[list[VectorPath]], list[list[VectorPath]], list[list[VectorPath]],
    dict[int, list[list[VectorPath]]],
]:
    """Single-linkage spatial merge of the incoming groups (by each
    group's own aggregate bbox), via `helpers.clustering.cluster_spatial`
    reused at the group level, constrained by `_any_side_close` (with
    `require_parallel=True`). Returns `(kept, debug_unconstrained,
    debug_no_parallel, lineage)` -- see this module's docstring."""

    def _close_parallel(a: list[VectorPath], b: list[VectorPath]) -> bool:
        return _any_side_close(a, b, size_tolerance, require_parallel=True)

    constrained = cluster_spatial(
        groups, get_bbox=bbox_of, threshold=threshold, extra_close=_close_parallel,
    )

    kept: list[list[VectorPath]] = []
    lineage: dict[int, list[list[VectorPath]]] = {}
    for cluster in constrained:
        piece = [p for sub in cluster for p in sub]
        kept.append(piece)
        lineage[id(piece)] = list(cluster)

    unconstrained = cluster_spatial(groups, get_bbox=bbox_of, threshold=threshold)
    debug_unconstrained = [[p for sub in cluster for p in sub] for cluster in unconstrained]

    def _close_no_parallel(a: list[VectorPath], b: list[VectorPath]) -> bool:
        return _any_side_close(a, b, size_tolerance, require_parallel=False)

    no_parallel = cluster_spatial(
        groups, get_bbox=bbox_of, threshold=threshold, extra_close=_close_no_parallel,
    )
    debug_no_parallel = [[p for sub in cluster for p in sub] for cluster in no_parallel]

    return kept, debug_unconstrained, debug_no_parallel, lineage


def filter_mixed_fill_rule_clusters(
    groups: list[list[VectorPath]],
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    """Drops a whole cluster if its members don't all share the same
    `fill_rule` ("f"/"fs"/"s", PyMuPDF's per-drawing paint-style field) --
    real text glyphs are painted consistently one way; a cluster mixing
    fill-only, stroke-only, and/or fill+stroke members is drawing content,
    not text."""
    return partition(groups, lambda g: len({p.fill_rule for p in g}) <= 1)


def filter_perimeter_only_clusters(
    groups: list[list[VectorPath]], group_stats: dict[int, GroupStats], margin_fraction: float,
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    """Drops whole clusters whose members all sit in the `margin_fraction`
    perimeter band of the cluster's own bbox (reused from `group_stats`,
    via `compute_group_stats`), never touching its shrunk-in center
    region -- e.g. a rectangle/ring made of border strokes with nothing
    drawn in the middle."""
    kept: list[list[VectorPath]] = []
    dropped: list[list[VectorPath]] = []
    for g in groups:
        x0, y0, x1, y1 = group_stats[id(g)].bbox
        width, height = x1 - x0, y1 - y0
        dx, dy = width * margin_fraction, height * margin_fraction
        center = (x0 + dx, y0 + dy, x1 - dx, y1 - dy)
        if center[0] >= center[2] or center[1] >= center[3]:
            # Margin swallows the whole bbox (tiny/thin cluster) -- nothing
            # meaningful to compare against, keep as-is.
            kept.append(g)
            continue
        if any(bboxes_intersect(p.bbox, center) for p in g):
            kept.append(g)
        else:
            dropped.append(g)
    return kept, dropped


def _density_axis_cell_count(
    extent: float, default_cols: int, min_cell_px: float, max_cell_px: float,
) -> int:
    """How many grid cells to split one axis into: `default_cols` (the
    usual, fixed grid density), clamped down if that would make a cell
    smaller than `min_cell_px` or up if it would make a cell bigger than
    `max_cell_px`."""
    if extent <= 0:
        return 1
    max_cols = max(1, int(extent // min_cell_px))
    min_cols = max(1, ceil(extent / max_cell_px))
    return min(max(default_cols, min_cols), max_cols)


def filter_density_clusters(
    groups: list[list[VectorPath]],
    group_stats: dict[int, GroupStats],
    default_grid_size: int,
    min_cell_px: float,
    max_cell_px: float,
    max_empty_fraction: float,
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    """Splits each cluster's own bbox (reused from `group_stats`, via
    `compute_group_stats`) into a grid sized so each cell's side is
    between `min_cell_px` and `max_cell_px`; if more than
    `max_empty_fraction` of the resulting cells have no member touching
    them, the cluster is too sparse to be text and is dropped whole."""
    kept: list[list[VectorPath]] = []
    dropped: list[list[VectorPath]] = []
    for g in groups:
        x0, y0, x1, y1 = group_stats[id(g)].bbox
        cols = _density_axis_cell_count(x1 - x0, default_grid_size, min_cell_px, max_cell_px)
        rows = _density_axis_cell_count(y1 - y0, default_grid_size, min_cell_px, max_cell_px)
        cell_w, cell_h = (x1 - x0) / cols, (y1 - y0) / rows
        if cell_w <= 0 or cell_h <= 0:
            kept.append(g)
            continue
        empty = 0
        for row in range(rows):
            for col in range(cols):
                cell = (
                    x0 + col * cell_w, y0 + row * cell_h,
                    x0 + (col + 1) * cell_w, y0 + (row + 1) * cell_h,
                )
                if not any(bboxes_intersect(p.bbox, cell) for p in g):
                    empty += 1
        (dropped if empty / (rows * cols) > max_empty_fraction else kept).append(g)
    return kept, dropped


def _centroid(path: VectorPath) -> tuple[float, float]:
    x0, y0, x1, y1 = path.bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _is_constant_spacing(paths: list[VectorPath], tolerance: float) -> bool:
    """True if consecutive members, ordered along whichever axis (x or y)
    has the larger spread, sit at a near-constant distance from each
    other (max deviation from the mean gap, relative to the mean gap,
    within `tolerance`)."""
    if len(paths) < 2:
        return False
    centroids = [_centroid(p) for p in paths]
    xs, ys = [c[0] for c in centroids], [c[1] for c in centroids]
    key = (lambda c: c[0]) if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else (lambda c: c[1])
    ordered = sorted(centroids, key=key)
    gaps = [hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(ordered, ordered[1:])]
    mean_gap = sum(gaps) / len(gaps)
    if mean_gap <= 0:
        return False
    return max(abs(g - mean_gap) for g in gaps) / mean_gap <= tolerance


def filter_constant_spacing_clusters(
    groups: list[list[VectorPath]],
    round_px: float,
    spacing_tolerance: float,
    min_repeat_count: int,
    pattern_fraction_threshold: float,
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    """Splits each cluster into same-shape sub-groups (by `vector_
    signature`) and judges spacing *within* each sub-group separately. The
    whole cluster is dropped if the members belonging to patterned
    sub-groups together make up at least `pattern_fraction_threshold` of
    the cluster's total member count."""

    def _mostly_patterned(g: list[VectorPath]) -> bool:
        if not g:
            return False
        by_sig: dict[VectorSignature, list[VectorPath]] = defaultdict(list)
        for p in g:
            by_sig[vector_signature(p, round_px)].append(p)
        patterned_count = sum(
            len(members)
            for members in by_sig.values()
            if len(members) >= min_repeat_count
            and _is_constant_spacing(members, spacing_tolerance)
        )
        return (patterned_count / len(g)) >= pattern_fraction_threshold

    return partition(groups, lambda g: not _mostly_patterned(g))


def _required_unique_signature_count(
    member_count: int,
    min_member_count: int,
    min_required: int,
    max_member_count: int,
    max_required: int,
) -> int:
    """How many distinct shape signatures a cluster with `member_count`
    members must have to survive `filter_low_variety_clusters` -- a
    log-scale ramp from `min_required` (at `member_count <=
    min_member_count`) up to `max_required` (at `member_count >=
    max_member_count`)."""
    if member_count <= min_member_count:
        return min_required
    ratio = log(member_count / min_member_count) / log(max_member_count / min_member_count)
    value = min_required + ratio * (max_required - min_required)
    return min(max_required, max(min_required, int(value)))


def _normalized_point_cloud(
    paths: list[VectorPath],
) -> tuple[list[tuple[float, float]], float]:
    """Translation+rotation-normalized point cloud for one cluster: every
    path's points, translated so the cluster's own centroid sits at the
    origin, then rotated by the negative of its 2D-PCA principal-axis angle
    so that axis aligns to x. Also returns `scale`, the cluster's own bbox
    max dimension -- the reference size for a caller's relative tolerance."""
    pts = [pt for p in paths for pt in p.points]
    if not pts:
        return [], 1.0
    cx = sum(x for x, y in pts) / len(pts)
    cy = sum(y for x, y in pts) / len(pts)
    centered = [(x - cx, y - cy) for x, y in pts]
    sxx = sum(x * x for x, y in centered)
    syy = sum(y * y for x, y in centered)
    sxy = sum(x * y for x, y in centered)
    theta = 0.5 * atan2(2 * sxy, sxx - syy)
    cos_t, sin_t = cos(-theta), sin(-theta)
    rotated = [(x * cos_t - y * sin_t, x * sin_t + y * cos_t) for x, y in centered]
    x0, y0, x1, y1 = union_bbox([p.bbox for p in paths])
    scale = max(x1 - x0, y1 - y0, 1e-6)
    return sorted(rotated), scale


def _point_clouds_close(
    a: list[tuple[float, float]], b: list[tuple[float, float]], tolerance: float,
) -> bool:
    return len(a) == len(b) and all(
        hypot(ax - bx, ay - by) <= tolerance for (ax, ay), (bx, by) in zip(a, b)
    )


def _clusters_similar(
    a: list[VectorPath], b: list[VectorPath], tolerance: float,
) -> bool:
    """True if `a` and `b` are the same shapes (same path count and same
    multiset of `kind`s) and, once both are translation+rotation-normalized
    (`_normalized_point_cloud`), every corresponding point pair sits within
    `tolerance * max(scale_a, scale_b)` of each other -- checked against
    both the direct normalization and its 180-degree-flipped mirror."""
    if len(a) != len(b) or tuple(sorted(p.kind for p in a)) != tuple(sorted(p.kind for p in b)):
        return False
    pts_a, scale_a = _normalized_point_cloud(a)
    pts_b, scale_b = _normalized_point_cloud(b)
    if len(pts_a) != len(pts_b):
        return False
    tol = tolerance * max(scale_a, scale_b)
    if _point_clouds_close(pts_a, pts_b, tol):
        return True
    flipped_b = sorted((-x, -y) for x, y in pts_b)
    return _point_clouds_close(pts_a, flipped_b, tol)


def group_similar_clusters(
    clusters: list[list[VectorPath]], tolerance: float,
) -> list[list[list[VectorPath]]]:
    """Whole-page, translation+rotation-tolerant grouping of geometrically
    equivalent clusters (see Glossary.md's "similarity group" entry) --
    e.g. repeated instances of the same label/symbol at different
    positions/orientations on the page. Greedy O(n^2): each cluster joins
    the first existing group whose representative (its first member) it's
    `_clusters_similar` to, else starts a new group."""
    groups: list[list[list[VectorPath]]] = []
    for cluster in clusters:
        matched = False
        for group in groups:
            if _clusters_similar(group[0], cluster, tolerance):
                group.append(cluster)
                matched = True
                break
        if not matched:
            groups.append([cluster])
    return groups


def filter_low_variety_clusters(
    groups: list[list[VectorPath]],
    group_stats: dict[int, GroupStats],
    min_member_count: int,
    min_required: int,
    max_member_count: int,
    max_required: int,
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    """Drops a whole cluster if it contains fewer distinct shape
    signatures (from `compute_group_stats`'s `unique_signature_count`,
    keyed by `id(group)`) than `_required_unique_signature_count` demands
    for its own member count."""
    kept: list[list[VectorPath]] = []
    dropped: list[list[VectorPath]] = []
    for g in groups:
        stats = group_stats[id(g)]
        required = _required_unique_signature_count(
            stats.member_count, min_member_count, min_required, max_member_count, max_required,
        )
        (kept if stats.unique_signature_count >= required else dropped).append(g)
    return kept, dropped
