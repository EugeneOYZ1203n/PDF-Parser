"""Pure tuple-math geometry for CAD-vector character graph construction --
step 2 of `docs/cad_font_vector_recognition.md`.

No pymupdf import (same convention as `commons/helpers/geometry.py`). The
cubic-bezier sampler here (`cubic_bezier_points`) is a deliberate, fresh
local reimplementation -- this package must never import
`scripts/label/_common.py::bezier_points` or any `P2_Raster_To_Vec`/
`P3_Vector_Parsing` backend's own bezier helper (e.g. the two `radon.py`
modules' private `_bezier`), per this feature's hard "commons-only" import
constraint. Everything in this module operates on plain `(x, y)` points and
`(p1, p2)` line segments built from `Vector.items` primitives -- see
`rastervec/commons/models/vector.py` for the `items` tuple shapes.
"""
from __future__ import annotations

import math

from rastervec.commons.helpers.clustering import cluster_spatial

Point = tuple[float, float]
Segment = tuple[Point, Point]

# Fallback/default sample count for `cubic_bezier_points` when called
# directly (e.g. by `estimate_curve_length`'s own coarse pre-pass, or a
# caller not going through `flatten_item_to_segments`). The real pipeline
# no longer flattens beziers to a fixed count -- `flatten_item_to_segments`
# picks an adaptive count from `curve_spacing` instead (see that function).
CURVE_SAMPLE_COUNT = 8


def cubic_bezier_points(
    p0: Point, p1: Point, p2: Point, p3: Point, n: int = CURVE_SAMPLE_COUNT,
) -> list[Point]:
    """`n + 1` points sampled along the cubic Bernstein-basis curve
    `p0 -> p1 -> p2 -> p3` at evenly spaced `t` in `[0, 1]`."""
    points: list[Point] = []
    for i in range(n + 1):
        t = i / n
        mt = 1.0 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
        points.append((x, y))
    return points


def estimate_curve_length(p0: Point, p1: Point, p2: Point, p3: Point, samples: int = 16) -> float:
    """Coarse arc-length estimate for one cubic bezier: sum of consecutive
    chord distances over a `samples`-point sampling. Only used to pick an
    adaptive final sample count in `flatten_item_to_segments` -- not exact,
    just good enough to size the real, final sampling."""
    pts = cubic_bezier_points(p0, p1, p2, p3, n=samples)
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def flatten_item_to_segments(
    item: tuple, curve_spacing: float = 1.0,
) -> tuple[list[Segment], set[Point]]:
    """One `Vector.items` entry -> `(segments, original_vertices)`.

    `"l"` -> one segment, both endpoints are original vertices. `"c"` ->
    adaptively sampled segments (roughly `curve_spacing` page units apart,
    via `estimate_curve_length`), where only the curve's own true start/end
    (`p0`/`p3`) are original vertices -- every interior sample point is
    synthetic. `"re"`/`"qu"` -> 4 edges closing the polygon (all 4 sides --
    not the 2-perpendicular-edge shortcut
    `similarity_single_line_cad_text.ipynb` uses; planarity/graph
    construction needs the real closed shape), all 4 corners are original
    vertices.

    `original_vertices` feeds `graph.build_char_graph`'s protected-node
    rule: an original vertex is never dropped by the area-based
    simplification pass, unlike a synthetic curve-interior sample."""
    kind = item[0]
    if kind == "l":
        p0, p1 = item[1], item[2]
        return [(p0, p1)], {p0, p1}
    if kind == "c":
        p0, p1, p2, p3 = item[1], item[2], item[3], item[4]
        spacing = max(curve_spacing, 1e-6)
        length = estimate_curve_length(p0, p1, p2, p3)
        n = max(2, math.ceil(length / spacing))
        pts = cubic_bezier_points(p0, p1, p2, p3, n=n)
        return list(zip(pts, pts[1:])), {p0, p3}
    if kind == "re":
        x0, y0, x1, y1 = item[1]
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        return list(zip(corners, corners[1:] + corners[:1])), set(corners)
    if kind == "qu":
        corners = list(item[1])
        return list(zip(corners, corners[1:] + corners[:1])), set(corners)
    return [], set()


def vector_to_segments(v, curve_spacing: float = 1.0) -> tuple[list[Segment], list[frozenset[Point]]]:
    """Every item of one `Vector`, flattened and concatenated in item
    order, alongside `vertex_groups`: one `frozenset[Point]` per item
    holding *that item's own* original vertices -- kept as separate,
    per-item groups (rather than flattened into one set) so a caller can
    tell "these 2 (or 4) points came from the same item" from "these came
    from different items", which `merge_close_points`'s `forbidden_groups`
    and `graph.build_char_graph`'s protected-node computation both need."""
    segments: list[Segment] = []
    vertex_groups: list[frozenset[Point]] = []
    for item in v.items:
        segs, verts = flatten_item_to_segments(item, curve_spacing=curve_spacing)
        segments.extend(segs)
        if verts:
            vertex_groups.append(frozenset(verts))
    return segments, vertex_groups


def vectors_to_segments(
    vectors, curve_spacing: float = 1.0,
) -> tuple[list[Segment], list[frozenset[Point]]]:
    """`vector_to_segments` over every vector, concatenated in order;
    `vertex_groups` extended (not unioned) across every vector, so
    per-item grouping survives across vectors too."""
    out: list[Segment] = []
    vertex_groups: list[frozenset[Point]] = []
    for v in vectors:
        segs, groups = vector_to_segments(v, curve_spacing=curve_spacing)
        out.extend(segs)
        vertex_groups.extend(groups)
    return out, vertex_groups


def _segment_intersection(a: Segment, b: Segment, tol: float = 0.5) -> Point | None:
    """Standard 2D parametric segment-intersection (determinant solve).
    Returns the intersection point if the segments cross at a single point
    with both parameters within `tol` (converted to a per-segment
    fractional epsilon, since `t`/`u` are dimensionless proportions of
    each segment's own length) of `[0, 1]`; `None` for parallel/colinear
    segments or segments that don't cross.

    The tolerance matters in practice: an upstream rigid rotation (e.g.
    `character_bank.to_baseline_relative_vectors`) composed with a PDF's
    own limited on-disk coordinate precision can leave a point that should
    land exactly on another segment (a genuine T-touch) off by ~1e-5 to
    1e-6 page units -- tiny in absolute terms, but enough to push `t`/`u`
    just outside a hardcoded `[0, 1]` and silently miss the intersection.
    Reusing `tol` (the caller's own `point_tol`, the same "how close is
    close enough" tolerance `split_at_intersections` uses for its endpoint
    check) keeps this a single tunable knob instead of a second magic
    constant."""
    (x1, y1), (x2, y2) = a
    (x3, y3), (x4, y4) = b
    d1x, d1y = x2 - x1, y2 - y1
    d2x, d2y = x4 - x3, y4 - y3
    denom = d1x * d2y - d1y * d2x
    if abs(denom) < 1e-12:
        return None
    t = ((x3 - x1) * d2y - (y3 - y1) * d2x) / denom
    u = ((x3 - x1) * d1y - (y3 - y1) * d1x) / denom
    len_a = math.hypot(d1x, d1y)
    len_b = math.hypot(d2x, d2y)
    eps_t = tol / len_a if len_a > 1e-9 else 0.0
    eps_u = tol / len_b if len_b > 1e-9 else 0.0
    if not (-eps_t <= t <= 1.0 + eps_t and -eps_u <= u <= 1.0 + eps_u):
        return None
    return (x1 + t * d1x, y1 + t * d1y)


def _dist(a: Point, b: Point) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def split_at_intersections(segments: list[Segment], *, point_tol: float = 0.5) -> list[Segment]:
    """All-pairs (n is small: one character's segment count) intersection
    test; for every crossing found (not already within `point_tol` of one
    of that segment's own endpoints), record it as a split point on both
    segments. Re-emits each original segment as its split sub-segments,
    ordered along the segment's own direction -- an edge set where every
    pair of edges shares at most one endpoint (planar)."""
    n = len(segments)
    split_points: list[list[Point]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            pt = _segment_intersection(segments[i], segments[j], tol=point_tol)
            if pt is None:
                continue
            for k, seg in ((i, segments[i]), (j, segments[j])):
                if _dist(pt, seg[0]) <= point_tol or _dist(pt, seg[1]) <= point_tol:
                    continue
                split_points[k].append(pt)

    out: list[Segment] = []
    for idx, seg in enumerate(segments):
        p0, p1 = seg
        extra = split_points[idx]
        if not extra:
            out.append(seg)
            continue
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]

        def proj(p: Point) -> float:
            return (p[0] - p0[0]) * dx + (p[1] - p0[1]) * dy

        ordered = sorted({p0, p1, *extra}, key=proj)
        out.extend(zip(ordered, ordered[1:]))
    return out


def merge_close_points(
    segments: list[Segment],
    *, point_merge_tol: float = 1e-3,
    forbidden_groups: list[frozenset[Point]] | None = None,
) -> tuple[list[Point], list[tuple[int, int]], dict[Point, int]]:
    """Reduces every segment endpoint to a canonical node index, merging
    coincident points (within `point_merge_tol`) into one node.

    Pure floating-point-duplicate dedup now, not a visual-simplification
    step -- the default tolerance is small on purpose. (Visual point
    reduction is `graph.py`'s degree-aware simplification pass, which runs
    after this.)

    `forbidden_groups`, when given, is `vectors_to_segments`'s per-item
    vertex groups -- used only to protect the **2-member** groups (a
    single `"l"`/`"c"` item's own 2 original vertices): two points that
    are both members of the same 2-member group are never allowed to
    merge directly, even within `point_merge_tol`, so a single non-closed
    item's own endpoints can never collapse to fewer than 2 distinct
    nodes. 4-member groups (`"re"`/`"qu"` corners) get no such protection
    and may still merge normally. This is single-linkage clustering, so a
    third, unrelated nearby point could in principle still transitively
    bridge two forbidden points into one cluster -- a known, accepted,
    low-probability edge case, not solved here.

    Reuses `commons.helpers.clustering.cluster_spatial` over every segment
    endpoint, each treated as a zero-size bbox, rather than writing a
    second spatial clusterer. Each returned cluster becomes one node (its
    position the centroid of its member points); segment endpoints are
    re-mapped through the point -> node-index map to build deduped edges
    (self-loop edges -- both endpoints collapsing to the same node -- are
    dropped). Also returns the raw-point -> node-index map itself, so a
    caller can look up which final node a known original-data point ended
    up as (`graph.py`'s protected-node computation)."""
    all_points: list[Point] = [p for seg in segments for p in seg]
    if not all_points:
        return [], [], {}

    extra_close = None
    if forbidden_groups:
        point_group_ids: dict[Point, list[int]] = {}
        for gid, group in enumerate(forbidden_groups):
            if len(group) != 2:
                continue
            for p in group:
                point_group_ids.setdefault(p, []).append(gid)
        if point_group_ids:
            def extra_close(p: Point, q: Point) -> bool:
                gids_p = point_group_ids.get(p)
                gids_q = point_group_ids.get(q)
                if not gids_p or not gids_q:
                    return True
                return not (set(gids_p) & set(gids_q))

    clusters = cluster_spatial(
        all_points,
        get_bbox=lambda p: (p[0], p[1], p[0], p[1]),
        threshold=point_merge_tol,
        extra_close=extra_close,
    )

    nodes: list[Point] = []
    point_to_node: dict[Point, int] = {}
    for node_idx, cluster in enumerate(clusters):
        cx = sum(p[0] for p in cluster) / len(cluster)
        cy = sum(p[1] for p in cluster) / len(cluster)
        nodes.append((cx, cy))
        for p in cluster:
            point_to_node[p] = node_idx

    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for seg in segments:
        a = point_to_node[seg[0]]
        b = point_to_node[seg[1]]
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        edges.append(key)
    return nodes, edges, point_to_node
