"""Fixed vector-classification pipeline steps for the Vector stage.

Each function is one step of the fixed pipeline `Vector.cluster()` runs,
in order:

1. `filter_large_items` -- drop items whose own bbox's max dimension
   exceeds a fraction of the page's smaller side.
2. `compute_vector_signatures` -- informational, pure pass-through: builds
   a per-signature occurrence count (`vector_signature`) over the current
   population, consumed by step 3, `compute_group_stats`,
   `filter_high_frequency_clusters`, and `filter_constant_spacing_clusters`
   below (and by the debug app's "color by vector type" view) -- never
   drops anything itself.
3. `remove_duplicate_runs` + `combine_overlapping_seq` -- sorts by `seq`,
   drops maximal runs of `DUPLICATE_RUN_MIN_LENGTH`-or-more consecutive
   items sharing the exact same shape signature (e.g. a long strip of
   identical tick marks), then chain-merges whatever's left into groups by
   bbox-gap tolerance, same rule as the old `group_by_seq_overlap`.
4. `filter_tiny_groups` -- drop a whole group if its aggregate bbox's max
   dimension is under `MIN_GROUP_SIZE_PX` -- a leftover speck too small
   to be a real glyph.
5. `filter_large_groups` -- same dimension check as step 1, applied to
   each group's aggregate bbox instead of one item's.
6. `cluster_spatial_groups` -- single-linkage spatial merge of the
   remaining groups (by bbox gap), constrained: two groups only merge if
   one has a "valid" side (see below) within `SPATIAL_SIZE_TOLERANCE` of a
   valid side of the other, AND those two sides are roughly parallel (lie
   along the same axis -- see below). A group's own short side ("width")
   only counts as valid if it's at least `SPATIAL_MIN_SIDE_FRACTION` of
   that same group's long side ("length") -- this excludes a thin
   stroke's near-zero short side from ever being used as a comparand.
   "Parallel" means: a group's length lies along whichever axis (x or y)
   its bbox's larger extent is on, and its width along the other axis;
   two sides are only compared if their axes match -- e.g. group A's
   length can match group B's width, but only if A's long side and B's
   short side both happen to lie on the same axis (both horizontal or
   both vertical). Right after merging, every resulting cluster is split
   by whichever matched side-length value (rounded to `SPATIAL_TAG_
   ROUND_PX`) caused the most of its internal merges: the sub-groups that
   merged via that dominant length become one piece, everything else
   (merged via a less-common length, or not the cause of any merge)
   becomes a second piece -- both continue downstream as ordinary
   clusters, nothing dropped, just separated so a locally-repeated
   same-length pattern doesn't stay mixed in with whatever else got
   spatially swept into the same cluster. Also returns two debug-only
   categories: the same input spatially clustered with *no* size
   constraint at all (plain bbox-gap rule, unsplit), and with the
   size/validity constraint but *no* parallel requirement (also unsplit)
   -- for visually comparing all of these; none of the debug categories
   are ever folded into `drawing_vectors` or fed to the next step.
7. `filter_mixed_fill_rule_clusters` -- drop a whole cluster if its
   members don't all share the same `fill_rule` ("f"/"fs"/"s") -- a
   cluster mixing paint styles is drawing content, not text.
8. `compute_group_stats` -- informational, pure pass-through: computes and
   returns per-group `GroupStats` (member count, distinct shape-signature
   count, dominant signature's global frequency, bbox) keyed by `id(group)`
   -- consumed by `filter_low_variety_clusters` below and available to any
   future downstream consumer that wants per-group shape stats without
   recomputing them. Relies on group *object identity* staying stable
   through every later step that only filters (never rebuilds) the group
   lists it keeps, up to where it's read.
9. `filter_perimeter_only_clusters` -- drop whole clusters whose members
   never reach into the cluster's own shrunk-in center region, i.e. every
   member sits in the bbox's perimeter margin (border/ring geometry, not
   text).
10. `filter_density_clusters` -- splits each cluster's own bbox into a
    `DENSITY_DEFAULT_GRID_SIZE`-per-axis grid, clamped so each cell's
    side stays within `[DENSITY_MIN_CELL_PX, DENSITY_MAX_CELL_PX]` (fewer
    cells for a small cluster, more for a large one); drops the whole
    cluster if more than `DENSITY_MAX_EMPTY_FRACTION` of the cells have
    no member touching them -- too sparse to be text.
11. `filter_high_frequency_clusters` -- drop a whole cluster only if
    *every* member's shape signature is among the top
    `HIGH_FREQUENCY_TOP_FRACTION` highest-occurring signatures in this
    bucket (a percentile cutoff over step 2's counts, not an absolute
    count) -- a cluster made entirely of the page's most repeated shapes is
    almost certainly a texture/pattern fill, not text.
12. `filter_constant_spacing_clusters` -- splits each cluster into
    same-shape sub-groups and checks spacing *within* each sub-group
    separately (a cluster with 2 shape types gets 2 independent spacing
    checks); a sub-group counts as "patterned" if it has at least
    `PATTERN_MIN_REPEAT_COUNT` members with near-constant spacing between
    consecutive members (within `PATTERN_SPACING_TOLERANCE`). The whole
    cluster is dropped if the patterned sub-groups' members together make
    up at least `PATTERN_FRACTION_THRESHOLD` of the cluster's total
    members -- not requiring every sub-group to be patterned, just enough
    of the cluster's actual content -- a regular repeated pattern
    (hatching, tick marks), not text.
13. `filter_low_variety_clusters` -- drop a whole cluster if it contains
    fewer distinct shape signatures (from step 8's `GroupStats`) than a
    log-scale ramp requires for its member count -- `LOW_VARIETY_MIN_
    REQUIRED` (1) for clusters at or under `LOW_VARIETY_MIN_MEMBER_COUNT`
    (5) members, up to `LOW_VARIETY_MAX_REQUIRED` (10) for clusters at or
    over `LOW_VARIETY_MAX_MEMBER_COUNT` (300) members -- real text has a
    variety of glyph shapes; a cluster built from only a handful of
    repeated shapes is drawing content, and a tiny cluster can't
    realistically clear the same bar as a hundreds-strong one.

All thresholds are passed in by the caller (`Vector`'s own constants), so
nothing here is hardcoded -- see `rastervec/vector.py`'s module-level
constants to tune the pipeline.

Each "filter" step returns `(kept_groups, dropped_groups)`; pure
"cluster/group" or informational steps return `(kept_groups, [])` /
`(groups, extra_payload)` as documented per function -- see
`Vector.cluster()` for how each step's return value is wrapped into a
named `CategoryResult` for the debug UI.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import ceil, hypot, log

from rastervec.geometry import rect_gap, union_bbox
from rastervec.helpers.clustering import Clustering
from rastervec.models import Page, VectorPath

VectorSignature = tuple[str, tuple[tuple[int, int], ...]]


@dataclass
class GroupStats:
    """Per-group data computed by `compute_group_stats`, for downstream
    filter steps and any future consumer that wants per-group shape stats
    without recomputing them from scratch."""

    member_count: int
    unique_signature_count: int
    bbox: tuple[float, float, float, float]
    max_dimension: float
    dominant_signature_count: int


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
    `remove_duplicate_runs`' run-length check, `compute_group_stats`, and
    `filter_high_frequency_clusters`, plus the debug app's "color by
    vector type" view -- all four read the exact same counts, computed
    once here."""
    counts: Counter[VectorSignature] = Counter()
    for g in groups:
        for p in g:
            counts[vector_signature(p, round_px)] += 1
    return groups, dict(counts)


def remove_duplicate_runs(
    groups: list[list[VectorPath]], round_px: float, min_run_length: int,
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    """Sorts flat by `seq`, then walks the sequence collecting maximal runs
    of consecutive items sharing the exact same shape signature. A run
    with `min_run_length` or more members is dropped whole (as one group
    per run, for the debug UI); shorter runs are left untouched and kept
    as individual single-item groups, ready for `combine_overlapping_seq`.
    A run longer than `min_run_length` (e.g. 12 when the threshold is 5)
    is still dropped in its entirety -- there's no partial keep within one
    run -- and whatever non-matching item follows starts a fresh run,
    proceeding normally."""
    flat = sorted((p for g in groups for p in g), key=lambda p: p.seq)
    if not flat:
        return [], []

    kept: list[VectorPath] = []
    dropped_runs: list[list[VectorPath]] = []

    def _flush(run: list[VectorPath]) -> None:
        if len(run) >= min_run_length:
            dropped_runs.append(run)
        else:
            kept.extend(run)

    run = [flat[0]]
    run_sig = vector_signature(flat[0], round_px)
    for path in flat[1:]:
        sig = vector_signature(path, round_px)
        if sig == run_sig:
            run.append(path)
        else:
            _flush(run)
            run = [path]
            run_sig = sig
    _flush(run)

    return [[p] for p in kept], dropped_runs


def combine_overlapping_seq(
    groups: list[list[VectorPath]], tolerance: float,
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    """Sorts every incoming path by `seq`, then sweeps in that order:
    a path joins the current group if its bbox is within `tolerance` of
    the group's aggregate bbox so far, otherwise it starts a new group.
    Never drops anything -- pure regrouping."""
    flat = sorted((p for g in groups for p in g), key=lambda p: p.seq)
    if not flat:
        return [], []

    result: list[list[VectorPath]] = []
    current = [flat[0]]
    current_bbox = flat[0].bbox
    for path in flat[1:]:
        if rect_gap(current_bbox, path.bbox) <= tolerance:
            current.append(path)
            current_bbox = union_bbox([current_bbox, path.bbox])
        else:
            result.append(current)
            current = [path]
            current_bbox = path.bbox
    result.append(current)
    return result, []


def filter_tiny_groups(
    groups: list[list[VectorPath]], min_size_px: float,
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    """Drops a whole group if its aggregate bbox's max dimension is under
    `min_size_px` -- a leftover speck too small to be a real glyph."""
    kept: list[list[VectorPath]] = []
    dropped: list[list[VectorPath]] = []
    for g in groups:
        if _max_dimension(_bbox_of(g)) < min_size_px:
            dropped.append(g)
        else:
            kept.append(g)
    return kept, dropped


def filter_large_groups(
    groups: list[list[VectorPath]], page: Page, max_dimension_fraction: float,
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    """Same dimension check as `filter_large_items`, applied to each
    group's own aggregate bbox instead of one path's -- an all-or-nothing
    drop per group."""
    page_min = min(page.meta.width, page.meta.height)
    threshold = max_dimension_fraction * page_min if page_min > 0 else float("inf")

    kept = [g for g in groups if _max_dimension(_bbox_of(g)) <= threshold]
    kept_ids = {id(g) for g in kept}
    dropped = [g for g in groups if id(g) not in kept_ids]
    return kept, dropped


def _group_sides(
    bbox: tuple[float, float, float, float], min_side_fraction: float,
) -> list[tuple[float, str]]:
    """A group's comparable sides as `(length, axis)` pairs -- `axis` is
    "x" if that side is the bbox's horizontal extent, "y" if vertical.
    The bbox's longer extent ("length") is always included; the shorter
    extent ("width") is only included if it's at least `min_side_fraction`
    of the length -- a thin stroke's near-zero short side never becomes a
    comparand."""
    w, h = _dims(bbox)
    if w >= h:
        length, length_axis, width, width_axis = w, "x", h, "y"
    else:
        length, length_axis, width, width_axis = h, "y", w, "x"
    sides = [(length, length_axis)]
    if length > 0 and width >= min_side_fraction * length:
        sides.append((width, width_axis))
    return sides


def _matched_side_value(
    a: list[VectorPath],
    b: list[VectorPath],
    size_tolerance: float,
    min_side_fraction: float,
    require_parallel: bool,
) -> float | None:
    """The representative length (average of the two matched sides) of the
    first valid-side pair (see `_group_sides`) of `a` and `b` found within
    `size_tolerance` of each other, or `None` if no such pair exists. When
    `require_parallel` is set, a candidate pair only counts if the two
    sides also lie on the same axis (both horizontal or both vertical) --
    e.g. group A's length matching group B's width only merges them if
    A's long side and B's short side are actually parallel, not
    perpendicular, on the page. The returned value is what actually
    "caused" the match -- used both as the bool-equivalent "did this
    merge" signal (non-`None`) and, via `cluster_spatial_groups`'s
    dominant-length-tag split, to remember *which* length caused it."""
    sides_a = _group_sides(_bbox_of(a), min_side_fraction)
    sides_b = _group_sides(_bbox_of(b), min_side_fraction)
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
    min_side_fraction: float,
    require_parallel: bool,
) -> bool:
    """Bool-only view of `_matched_side_value` -- True if some valid side
    of `a` is within `size_tolerance` of some valid side of `b` (subject
    to `require_parallel`, see `_matched_side_value`)."""
    return _matched_side_value(a, b, size_tolerance, min_side_fraction, require_parallel) is not None


def _split_cluster_by_dominant_tag(
    cluster: list[list[VectorPath]],
    edges: list[tuple[int, int, float]],
    id_to_index: dict[int, int],
    tag_round_px: float,
) -> tuple[list[VectorPath], list[VectorPath] | None]:
    """Splits one spatially-merged cluster (a list of its original
    sub-groups) into `(dominant_piece, minority_piece)` using the union
    edges that actually formed it: every union edge's matched-side value
    (see `_matched_side_value`) is rounded to `tag_round_px` and tallied;
    the most frequent rounded value is the "dominant tag". Sub-groups
    touching at least one dominant-tag edge become `dominant_piece`;
    every other sub-group (touched only by less-frequent-tag edges, or
    not touched by any internal edge at all) becomes `minority_piece`.
    `minority_piece` is `None` (nothing to split) if the cluster never
    merged (no internal edges) or every sub-group touches the dominant
    tag."""
    indices = {id_to_index[id(sub)] for sub in cluster}
    internal_edges = [(i, j, tag) for i, j, tag in edges if i in indices and j in indices]

    if not internal_edges:
        return [p for sub in cluster for p in sub], None

    tag_counts: dict[float, int] = defaultdict(int)
    rounded_edges: list[tuple[int, int, float]] = []
    for i, j, tag in internal_edges:
        rounded = round(tag / tag_round_px) * tag_round_px if tag_round_px > 0 else tag
        tag_counts[rounded] += 1
        rounded_edges.append((i, j, rounded))
    dominant_tag = max(tag_counts.items(), key=lambda kv: kv[1])[0]

    dominant_indices: set[int] = set()
    for i, j, rounded in rounded_edges:
        if rounded == dominant_tag:
            dominant_indices.add(i)
            dominant_indices.add(j)

    dominant_sub = [sub for sub in cluster if id_to_index[id(sub)] in dominant_indices]
    minority_sub = [sub for sub in cluster if id_to_index[id(sub)] not in dominant_indices]

    dominant_piece = [p for sub in dominant_sub for p in sub]
    minority_piece = [p for sub in minority_sub for p in sub] if minority_sub else None
    return dominant_piece, minority_piece


def cluster_spatial_groups(
    groups: list[list[VectorPath]],
    clustering: Clustering,
    threshold: float,
    size_tolerance: float,
    min_side_fraction: float,
    tag_round_px: float,
) -> tuple[
    list[list[VectorPath]], list[list[VectorPath]], list[list[VectorPath]],
    list[list[VectorPath]], list[list[VectorPath]],
]:
    """Single-linkage spatial merge of the incoming groups (by each
    group's own aggregate bbox), via `Clustering.cluster_spatial_with_
    tags` reused at the group level, constrained by `_matched_side_value`
    (with `require_parallel=True`): two groups only merge if the bbox-gap
    rule passes AND some valid side of one is within `size_tolerance` of
    some valid side of the other AND those two sides are parallel (same
    axis). See `_group_sides`/`_matched_side_value` for exactly what
    "valid" and "parallel" mean.

    Right after that merge, every resulting cluster is split by
    `_split_cluster_by_dominant_tag` into the piece that merged via the
    cluster's single most common matched side-length (rounded to
    `tag_round_px`) and the piece that didn't (e.g. leftover members that
    only reached the cluster through a differently-sized match, or
    weren't the cause of any merge at all) -- a repeated hatch/tick
    pattern tends to concentrate on one dominant length, so separating it
    from whatever else got spatially swept in gives later steps
    (perimeter/density/frequency/variety) a cleaner, more homogeneous
    population to judge. Nothing is dropped here -- both pieces continue
    downstream as ordinary clusters.

    Returns `(kept, dominant_tag_groups, minority_tag_groups, debug_
    unconstrained, debug_no_parallel)`: `kept` is every dominant/minority
    piece from every cluster, flattened together (feeds the next step);
    `dominant_tag_groups`/`minority_tag_groups` are that same population
    partitioned by which half of the split each piece is, for the debug
    UI (every piece in `kept` appears in exactly one of the two).
    `debug_unconstrained` is the same input clustered with the plain
    bbox-gap rule only (no size constraint, no split); `debug_no_parallel`
    uses the same size/validity constraint as `kept` but without the
    parallel requirement (also unsplit). Both debug variants are purely
    for the debug app's opt-in comparison checkboxes -- neither is ever
    folded into `drawing_vectors` or fed downstream."""

    def _match_parallel(a: list[VectorPath], b: list[VectorPath]) -> float | None:
        return _matched_side_value(a, b, size_tolerance, min_side_fraction, require_parallel=True)

    constrained, edges = clustering.cluster_spatial_with_tags(
        groups, get_bbox=_bbox_of, threshold=threshold, match_fn=_match_parallel,
    )
    id_to_index = {id(g): i for i, g in enumerate(groups)}

    kept: list[list[VectorPath]] = []
    dominant_tag_groups: list[list[VectorPath]] = []
    minority_tag_groups: list[list[VectorPath]] = []
    for cluster in constrained:
        dominant_piece, minority_piece = _split_cluster_by_dominant_tag(
            cluster, edges, id_to_index, tag_round_px,
        )
        kept.append(dominant_piece)
        dominant_tag_groups.append(dominant_piece)
        if minority_piece is not None:
            kept.append(minority_piece)
            minority_tag_groups.append(minority_piece)

    unconstrained = clustering.cluster_spatial(groups, get_bbox=_bbox_of, threshold=threshold)
    debug_unconstrained = [[p for sub in cluster for p in sub] for cluster in unconstrained]

    def _close_no_parallel(a: list[VectorPath], b: list[VectorPath]) -> bool:
        return _any_side_close(a, b, size_tolerance, min_side_fraction, require_parallel=False)

    no_parallel = clustering.cluster_spatial(
        groups, get_bbox=_bbox_of, threshold=threshold, extra_close=_close_no_parallel,
    )
    debug_no_parallel = [[p for sub in cluster for p in sub] for cluster in no_parallel]

    return kept, dominant_tag_groups, minority_tag_groups, debug_unconstrained, debug_no_parallel


def filter_mixed_fill_rule_clusters(
    groups: list[list[VectorPath]],
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    """Drops a whole cluster if its members don't all share the same
    `fill_rule` ("f"/"fs"/"s", PyMuPDF's per-drawing paint-style field) --
    real text glyphs are painted consistently one way; a cluster mixing
    fill-only, stroke-only, and/or fill+stroke members is drawing content
    (e.g. a stroked outline plus a separately-filled body), not text."""
    kept: list[list[VectorPath]] = []
    dropped: list[list[VectorPath]] = []
    for g in groups:
        if len({p.fill_rule for p in g}) > 1:
            dropped.append(g)
        else:
            kept.append(g)
    return kept, dropped


def compute_group_stats(
    groups: list[list[VectorPath]],
    signature_counts: dict[VectorSignature, int],
    round_px: float,
) -> tuple[list[list[VectorPath]], dict[int, GroupStats]]:
    """Informational pass-through, like `compute_vector_signatures`: never
    drops or regroups anything. For every group (keyed by `id(group)` --
    stable through this step and every later step that keeps the same
    group objects, i.e. `filter_perimeter_only_clusters`/
    `filter_high_frequency_clusters`/`filter_low_variety_clusters`),
    records member count, the number of distinct shape signatures among
    its members, its bbox/max dimension, and the highest global
    occurrence count (from `compute_vector_signatures`) among those
    signatures -- consumed by `filter_low_variety_clusters` and available
    to any future downstream consumer wanting per-group shape stats
    without recomputing them."""
    stats: dict[int, GroupStats] = {}
    for g in groups:
        sigs = {vector_signature(p, round_px) for p in g}
        bbox = _bbox_of(g)
        stats[id(g)] = GroupStats(
            member_count=len(g),
            unique_signature_count=len(sigs),
            bbox=bbox,
            max_dimension=_max_dimension(bbox),
            dominant_signature_count=max(
                (signature_counts.get(s, 0) for s in sigs), default=0,
            ),
        )
    return groups, stats


def _bboxes_intersect(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float],
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 <= bx1 and bx0 <= ax1 and ay0 <= by1 and by0 <= ay1


def filter_perimeter_only_clusters(
    groups: list[list[VectorPath]], margin_fraction: float,
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    """Drops whole clusters whose members all sit in the `margin_fraction`
    perimeter band of the cluster's own bbox, never touching its shrunk-in
    center region -- e.g. a rectangle/ring made of border strokes with
    nothing drawn in the middle. Real text glyphs, even sparse ones, tend
    to have at least one member reaching into the center; a cluster where
    every member avoids it is drawing content, not a text candidate."""
    kept: list[list[VectorPath]] = []
    dropped: list[list[VectorPath]] = []
    for g in groups:
        x0, y0, x1, y1 = _bbox_of(g)
        width, height = x1 - x0, y1 - y0
        dx, dy = width * margin_fraction, height * margin_fraction
        center = (x0 + dx, y0 + dy, x1 - dx, y1 - dy)
        if center[0] >= center[2] or center[1] >= center[3]:
            # Margin swallows the whole bbox (tiny/thin cluster) -- nothing
            # meaningful to compare against, keep as-is.
            kept.append(g)
            continue
        if any(_bboxes_intersect(p.bbox, center) for p in g):
            kept.append(g)
        else:
            dropped.append(g)
    return kept, dropped


def _density_axis_cell_count(
    extent: float, default_cols: int, min_cell_px: float, max_cell_px: float,
) -> int:
    """How many grid cells to split one axis into: `default_cols` (the
    usual, fixed grid density), clamped down if that would make a cell
    smaller than `min_cell_px` ("use less grids if necessary") or up if
    it would make a cell bigger than `max_cell_px` -- so a small cluster
    gets fewer, bigger cells and a large one gets more, smaller ones,
    while a typically-sized cluster just gets the default grid unchanged."""
    if extent <= 0:
        return 1
    max_cols = max(1, int(extent // min_cell_px))  # cols such that cell >= min_cell_px
    min_cols = max(1, ceil(extent / max_cell_px))  # cols such that cell <= max_cell_px
    return min(max(default_cols, min_cols), max_cols)


def filter_density_clusters(
    groups: list[list[VectorPath]],
    default_grid_size: int,
    min_cell_px: float,
    max_cell_px: float,
    max_empty_fraction: float,
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    """Splits each cluster's own bbox into a grid sized so each cell's
    side is between `min_cell_px` and `max_cell_px` (independently per
    axis, via `_density_axis_cell_count` -- `default_grid_size` cells per
    axis for a typically-sized cluster, fewer for a small one, more for a
    large one); if more than `max_empty_fraction` of the resulting cells
    have no member touching them, the cluster is too sparse to be text
    (drawing content -- e.g. a dimension line with a few isolated ticks)
    and is dropped whole."""
    kept: list[list[VectorPath]] = []
    dropped: list[list[VectorPath]] = []
    for g in groups:
        x0, y0, x1, y1 = _bbox_of(g)
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
                if not any(_bboxes_intersect(p.bbox, cell) for p in g):
                    empty += 1
        (dropped if empty / (rows * cols) > max_empty_fraction else kept).append(g)
    return kept, dropped


def _high_frequency_threshold(
    signature_counts: dict[VectorSignature, int], top_fraction: float,
) -> float:
    """The occurrence count at the `top_fraction` cutoff over the
    distinct-signature occurrence distribution (a percentile, not an
    absolute count) -- a signature at or above this count is "high
    frequency". Sorts each distinct signature's own count descending and
    picks the value `top_fraction` of the way into that list, so the
    cutoff scales with however many distinct shapes are actually present
    in this bucket."""
    if not signature_counts:
        return float("inf")
    counts = sorted(signature_counts.values(), reverse=True)
    n = len(counts)
    idx = min(max(int(n * top_fraction) - 1, 0), n - 1)
    return counts[idx]


def filter_high_frequency_clusters(
    groups: list[list[VectorPath]],
    signature_counts: dict[VectorSignature, int],
    round_px: float,
    top_fraction: float,
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    """Drops a whole cluster only if *every* member's shape signature is
    among the top `top_fraction` highest-occurring signatures in
    `signature_counts` (from `compute_vector_signatures`, computed once
    over this bucket's whole population) -- a cluster made entirely of the
    bucket's most repeated shapes is almost certainly a texture/pattern
    fill, not text. A cluster with even one member outside that top slice
    is kept."""
    threshold = _high_frequency_threshold(signature_counts, top_fraction)
    kept: list[list[VectorPath]] = []
    dropped: list[list[VectorPath]] = []
    for g in groups:
        if all(
            signature_counts.get(vector_signature(p, round_px), 0) >= threshold
            for p in g
        ):
            dropped.append(g)
        else:
            kept.append(g)
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
    signature`) and judges spacing *within* each sub-group separately --
    a cluster containing 2 shape types gets 2 independent spacing checks,
    not one check over the whole mixed population. A sub-group counts as
    "patterned" if it has at least `min_repeat_count` members with
    near-constant spacing between consecutive members (see
    `_is_constant_spacing`). The whole cluster is dropped if the members
    belonging to patterned sub-groups together make up at least
    `pattern_fraction_threshold` of the cluster's total member count --
    not every sub-group needs to be patterned, just enough of the
    cluster's actual content -- a regular repeated pattern (hatching,
    tick marks), not text."""
    kept: list[list[VectorPath]] = []
    dropped: list[list[VectorPath]] = []
    for g in groups:
        by_sig: dict[VectorSignature, list[VectorPath]] = defaultdict(list)
        for p in g:
            by_sig[vector_signature(p, round_px)].append(p)
        patterned_count = sum(
            len(members)
            for members in by_sig.values()
            if len(members) >= min_repeat_count and _is_constant_spacing(members, spacing_tolerance)
        )
        if g and (patterned_count / len(g)) >= pattern_fraction_threshold:
            dropped.append(g)
        else:
            kept.append(g)
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
    max_member_count`), growing fastest over the first few members and
    flattening out over the long tail -- e.g. requiring 5 distinct shapes
    was unconditionally unpassable for any cluster with under 5 members
    in the first place; a tiny cluster shouldn't be held to the same bar
    as a hundreds-strong one."""
    if member_count <= min_member_count:
        return min_required
    ratio = log(member_count / min_member_count) / log(max_member_count / min_member_count)
    value = min_required + ratio * (max_required - min_required)
    return min(max_required, max(min_required, int(value)))


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
    for its own member count -- real text has a variety of glyph shapes;
    a cluster built from only a handful of repeated shapes (ticks, hatch
    strokes) is drawing content, not a text candidate."""
    kept: list[list[VectorPath]] = []
    dropped: list[list[VectorPath]] = []
    for g in groups:
        stats = group_stats[id(g)]
        required = _required_unique_signature_count(
            stats.member_count, min_member_count, min_required, max_member_count, max_required,
        )
        (kept if stats.unique_signature_count >= required else dropped).append(g)
    return kept, dropped
