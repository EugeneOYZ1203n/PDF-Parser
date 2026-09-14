"""Cluster-level steps of the Vector Classification pipeline (see
rastervec/Vector_Classification/classification.py for the fixed step
order these are run in). A "cluster" is the final classification output
for one (layer, color) bucket -- see docs/Glossary.md for the group/cluster
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
   cluster continues downstream as one ordinary cluster -- unlike the
   pre-refactor version, member groups are kept *nested*
   (`list[list[Vector]]` per cluster) rather than flattened, so a
   cluster's group/vector tiering is real structure, not a side-channel
   `id()`-keyed lineage dict. Also returns two debug-only categories.
7. `filter_mixed_fill_rule_clusters` -- drop a whole cluster if its
   members don't all share the same `type` ("f"/"fs"/"s", PyMuPDF's
   per-drawing paint-style field) -- a cluster mixing paint styles is
   drawing content, not text.
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
    same-shape sub-groups (at the Vector-*item* level -- a repeated tick
    mark is often one item within a larger multi-item Vector, not a whole
    Vector on its own) and checks spacing *within* each sub-group
    separately; the whole cluster is dropped if the patterned sub-groups'
    members together make up at least `PATTERN_FRACTION_THRESHOLD` of the
    cluster's total item count -- a regular repeated pattern (hatching,
    tick marks), not text.
12. `filter_low_variety_clusters` -- drop a whole cluster if it contains
    fewer distinct shape signatures (from `compute_group_stats`) than a
    log-scale ramp requires for its member count -- real text has a
    variety of glyph shapes; a cluster built from only a handful of
    repeated shapes is drawing content, not a text candidate.

Whole-page similarity grouping of text-candidate clusters used to live here
(`group_similar_clusters`) but has moved downstream, past Radon segmentation
-- it now operates on post-Radon `Segment`s (using each segment's own known
precise skew angle for rotation-exact normalization) instead of pre-Radon
clusters via a PCA-based rotation search. See `OCR/radon.py` /
`pipelines/_steps.py`'s segment-similarity step.

All thresholds are passed in by the caller (`classification.py`'s own
constants), so nothing here is hardcoded.
"""
from __future__ import annotations

from collections import defaultdict
from math import ceil, hypot, log

from rastervec.helpers.clustering import cluster_spatial
from rastervec.helpers.geometry import bboxes_intersect, dims, item_bbox
from rastervec.helpers.iterutils import partition
from rastervec.models import Vector
from rastervec.Vector_Classification.group_filters import GroupStats
from rastervec.Vector_Classification.item_filters import (
    ItemSignature,
    bbox_of,
    item_signature,
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
    a: list[Vector],
    b: list[Vector],
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
    a: list[Vector],
    b: list[Vector],
    size_tolerance: float,
    require_parallel: bool,
) -> bool:
    """Bool-only view of `_matched_side_value` -- True if some valid side
    of `a` is within `size_tolerance` of some valid side of `b` (subject
    to `require_parallel`, see `_matched_side_value`)."""
    return _matched_side_value(a, b, size_tolerance, require_parallel) is not None


def cluster_spatial_groups(
    groups: list[list[Vector]],
    threshold: float,
    size_tolerance: float,
) -> tuple[
    list[list[list[Vector]]], list[list[Vector]], list[list[Vector]],
]:
    """Single-linkage spatial merge of the incoming groups (by each
    group's own aggregate bbox), via `helpers.clustering.cluster_spatial`
    reused at the group level, constrained by `_any_side_close` (with
    `require_parallel=True`). Returns `(kept, debug_unconstrained,
    debug_no_parallel)`. `kept` preserves real group/vector tiering --
    each entry is `list[list[Vector]]` (this cluster's member groups,
    unflattened) rather than a flat `list[Vector]`; the two debug
    categories stay flattened since they're display-only."""

    def _close_parallel(a: list[Vector], b: list[Vector]) -> bool:
        return _any_side_close(a, b, size_tolerance, require_parallel=True)

    constrained = cluster_spatial(
        groups, get_bbox=bbox_of, threshold=threshold, extra_close=_close_parallel,
    )
    kept: list[list[list[Vector]]] = [list(cluster) for cluster in constrained]

    unconstrained = cluster_spatial(groups, get_bbox=bbox_of, threshold=threshold)
    debug_unconstrained = [[v for sub in cluster for v in sub] for cluster in unconstrained]

    def _close_no_parallel(a: list[Vector], b: list[Vector]) -> bool:
        return _any_side_close(a, b, size_tolerance, require_parallel=False)

    no_parallel = cluster_spatial(
        groups, get_bbox=bbox_of, threshold=threshold, extra_close=_close_no_parallel,
    )
    debug_no_parallel = [[v for sub in cluster for v in sub] for cluster in no_parallel]

    return kept, debug_unconstrained, debug_no_parallel


def _flatten(cluster: list[list[Vector]]) -> list[Vector]:
    return [v for group in cluster for v in group]


def filter_mixed_fill_rule_clusters(
    clusters: list[list[list[Vector]]],
) -> tuple[list[list[list[Vector]]], list[list[Vector]]]:
    """Drops a whole cluster if its members don't all share the same
    `type` ("f"/"fs"/"s", PyMuPDF's per-drawing paint-style field) --
    real text glyphs are painted consistently one way; a cluster mixing
    fill-only, stroke-only, and/or fill+stroke members is drawing content,
    not text. Dropped clusters are reported flattened (debug-only)."""
    kept: list[list[list[Vector]]] = []
    dropped: list[list[Vector]] = []
    for cluster in clusters:
        flat = _flatten(cluster)
        if len({v.type for v in flat}) <= 1:
            kept.append(cluster)
        else:
            dropped.append(flat)
    return kept, dropped


def filter_perimeter_only_clusters(
    clusters: list[list[list[Vector]]], group_stats: dict[int, GroupStats], margin_fraction: float,
) -> tuple[list[list[list[Vector]]], list[list[Vector]]]:
    """Drops whole clusters whose members all sit in the `margin_fraction`
    perimeter band of the cluster's own bbox (reused from `group_stats`,
    via `compute_group_stats`), never touching its shrunk-in center
    region -- e.g. a rectangle/ring made of border strokes with nothing
    drawn in the middle. Checked per-item (a Vector spanning the border
    might still have one item reaching into the center)."""
    kept: list[list[list[Vector]]] = []
    dropped: list[list[Vector]] = []
    for cluster in clusters:
        flat = _flatten(cluster)
        x0, y0, x1, y1 = group_stats[id(cluster)].bbox
        width, height = x1 - x0, y1 - y0
        dx, dy = width * margin_fraction, height * margin_fraction
        center = (x0 + dx, y0 + dy, x1 - dx, y1 - dy)
        if center[0] >= center[2] or center[1] >= center[3]:
            # Margin swallows the whole bbox (tiny/thin cluster) -- nothing
            # meaningful to compare against, keep as-is.
            kept.append(cluster)
            continue
        touches_center = any(
            bboxes_intersect(item_bbox(item), center)
            for v in flat
            for item in v.items
        )
        if touches_center:
            kept.append(cluster)
        else:
            dropped.append(flat)
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
    clusters: list[list[list[Vector]]],
    group_stats: dict[int, GroupStats],
    default_grid_size: int,
    min_cell_px: float,
    max_cell_px: float,
    max_empty_fraction: float,
) -> tuple[list[list[list[Vector]]], list[list[Vector]]]:
    """Splits each cluster's own bbox (reused from `group_stats`, via
    `compute_group_stats`) into a grid sized so each cell's side is
    between `min_cell_px` and `max_cell_px`; if more than
    `max_empty_fraction` of the resulting cells have no member touching
    them, the cluster is too sparse to be text and is dropped whole.
    Checked per-item, same rationale as `filter_perimeter_only_clusters`."""
    kept: list[list[list[Vector]]] = []
    dropped: list[list[Vector]] = []
    for cluster in clusters:
        flat = _flatten(cluster)
        x0, y0, x1, y1 = group_stats[id(cluster)].bbox
        cols = _density_axis_cell_count(x1 - x0, default_grid_size, min_cell_px, max_cell_px)
        rows = _density_axis_cell_count(y1 - y0, default_grid_size, min_cell_px, max_cell_px)
        cell_w, cell_h = (x1 - x0) / cols, (y1 - y0) / rows
        if cell_w <= 0 or cell_h <= 0:
            kept.append(cluster)
            continue
        item_bboxes = [item_bbox(item) for v in flat for item in v.items]
        empty = 0
        for row in range(rows):
            for col in range(cols):
                cell = (
                    x0 + col * cell_w, y0 + row * cell_h,
                    x0 + (col + 1) * cell_w, y0 + (row + 1) * cell_h,
                )
                if not any(bboxes_intersect(b, cell) for b in item_bboxes):
                    empty += 1
        if empty / (rows * cols) > max_empty_fraction:
            dropped.append(flat)
        else:
            kept.append(cluster)
    return kept, dropped


def _item_centroid(item: tuple) -> tuple[float, float]:
    x0, y0, x1, y1 = item_bbox(item)
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _is_constant_spacing(items: list[tuple], tolerance: float) -> bool:
    """True if consecutive members, ordered along whichever axis (x or y)
    has the larger spread, sit at a near-constant distance from each
    other (max deviation from the mean gap, relative to the mean gap,
    within `tolerance`)."""
    if len(items) < 2:
        return False
    centroids = [_item_centroid(it) for it in items]
    xs, ys = [c[0] for c in centroids], [c[1] for c in centroids]
    key = (lambda c: c[0]) if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else (lambda c: c[1])
    ordered = sorted(centroids, key=key)
    gaps = [hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(ordered, ordered[1:])]
    mean_gap = sum(gaps) / len(gaps)
    if mean_gap <= 0:
        return False
    return max(abs(g - mean_gap) for g in gaps) / mean_gap <= tolerance


def filter_constant_spacing_clusters(
    clusters: list[list[list[Vector]]],
    round_px: float,
    spacing_tolerance: float,
    min_repeat_count: int,
    pattern_fraction_threshold: float,
) -> tuple[list[list[list[Vector]]], list[list[Vector]]]:
    """Splits each cluster's items (across every member Vector) into
    same-shape sub-groups (by `item_signature`) and judges spacing
    *within* each sub-group separately -- a repeated tick mark or hatch
    line is often one item within a larger multi-item Vector, so pattern
    detection operates at item granularity even though a whole cluster is
    kept or dropped as one unit. The whole cluster is dropped if the
    items belonging to patterned sub-groups together make up at least
    `pattern_fraction_threshold` of the cluster's total item count."""

    def _mostly_patterned(flat: list[Vector]) -> bool:
        items = [item for v in flat for item in v.items]
        if not items:
            return False
        by_sig: dict[ItemSignature, list[tuple]] = defaultdict(list)
        for item in items:
            by_sig[item_signature(item, round_px)].append(item)
        patterned_count = sum(
            len(members)
            for members in by_sig.values()
            if len(members) >= min_repeat_count
            and _is_constant_spacing(members, spacing_tolerance)
        )
        return (patterned_count / len(items)) >= pattern_fraction_threshold

    kept: list[list[list[Vector]]] = []
    dropped: list[list[Vector]] = []
    for cluster in clusters:
        flat = _flatten(cluster)
        if _mostly_patterned(flat):
            dropped.append(flat)
        else:
            kept.append(cluster)
    return kept, dropped


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


def filter_low_variety_clusters(
    clusters: list[list[list[Vector]]],
    group_stats: dict[int, GroupStats],
    min_member_count: int,
    min_required: int,
    max_member_count: int,
    max_required: int,
) -> tuple[list[list[list[Vector]]], list[list[Vector]]]:
    """Drops a whole cluster if it contains fewer distinct shape
    signatures (from `compute_group_stats`'s `unique_signature_count`,
    keyed by `id(cluster)`) than `_required_unique_signature_count`
    demands for its own member count."""
    kept: list[list[list[Vector]]] = []
    dropped: list[list[Vector]] = []
    for cluster in clusters:
        stats = group_stats[id(cluster)]
        required = _required_unique_signature_count(
            stats.member_count, min_member_count, min_required, max_member_count, max_required,
        )
        if stats.unique_signature_count >= required:
            kept.append(cluster)
        else:
            dropped.append(_flatten(cluster))
    return kept, dropped
