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

from scipy.spatial import cKDTree

Point = tuple[float, float]
Segment = tuple[Point, Point]

# Default sample count for `cubic_bezier_points` when called directly, and
# the fixed number of points `flatten_item_to_segments` always samples a
# "c" item into (2 real endpoints + 3 generated interior points) --
# deliberately a fixed count now, not adaptive: the old `curve_spacing`-
# driven adaptive sampling (and the `_character_scale`-derived fraction that
# fed it) is gone, per this project's move to a fully data-derived
# `epsilon` (see `graph.py::build_char_graph`) for both connectivity and
# simplification.
CURVE_SAMPLE_COUNT = 8
BEZIER_SAMPLE_COUNT = 5


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


def flatten_item_to_segments(
    item: tuple, bezier_sample_count: int = BEZIER_SAMPLE_COUNT,
) -> tuple[list[Segment], set[Point]]:
    """One `Vector.items` entry -> `(segments, original_vertices)`.

    `"l"` -> one segment, both endpoints are original vertices. `"c"` ->
    always sampled into a fixed `bezier_sample_count` points (2 real
    endpoints + `bezier_sample_count - 2` generated interior points, e.g.
    3 interior points for the default 5), where only the curve's own true
    start/end (`p0`/`p3`) are original vertices -- every interior sample
    point is synthetic. `"re"`/`"qu"` -> 4 edges closing the polygon (all 4
    sides -- not the 2-perpendicular-edge shortcut
    `similarity_single_line_cad_text.ipynb` uses; planarity/graph
    construction needs the real closed shape), all 4 corners are original
    vertices.

    `original_vertices` feeds `graph.build_char_graph`'s protected-node
    rule: an original vertex is never dropped by the RDP-based
    simplification pass, unlike a synthetic curve-interior sample."""
    kind = item[0]
    if kind == "l":
        p0, p1 = item[1], item[2]
        return [(p0, p1)], {p0, p1}
    if kind == "c":
        p0, p1, p2, p3 = item[1], item[2], item[3], item[4]
        pts = cubic_bezier_points(p0, p1, p2, p3, n=bezier_sample_count - 1)
        return list(zip(pts, pts[1:])), {p0, p3}
    if kind == "re":
        x0, y0, x1, y1 = item[1]
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        return list(zip(corners, corners[1:] + corners[:1])), set(corners)
    if kind == "qu":
        corners = list(item[1])
        return list(zip(corners, corners[1:] + corners[:1])), set(corners)
    return [], set()


def vector_to_segments(
    v, bezier_sample_count: int = BEZIER_SAMPLE_COUNT,
) -> tuple[list[Segment], list[frozenset[Point]]]:
    """Every item of one `Vector`, flattened and concatenated in item
    order, alongside `vertex_groups`: one `frozenset[Point]` per item
    holding *that item's own* original vertices -- kept as separate,
    per-item groups (rather than flattened into one set) so a caller can
    tell "these 2 (or 4) points came from the same item" from "these came
    from different items"."""
    segments: list[Segment] = []
    vertex_groups: list[frozenset[Point]] = []
    for item in v.items:
        segs, verts = flatten_item_to_segments(item, bezier_sample_count=bezier_sample_count)
        segments.extend(segs)
        if verts:
            vertex_groups.append(frozenset(verts))
    return segments, vertex_groups


def vectors_to_segments(
    vectors, bezier_sample_count: int = BEZIER_SAMPLE_COUNT,
) -> tuple[list[Segment], list[frozenset[Point]]]:
    """`vector_to_segments` over every vector, concatenated in order;
    `vertex_groups` extended (not unioned) across every vector, so
    per-item grouping survives across vectors too."""
    out: list[Segment] = []
    vertex_groups: list[frozenset[Point]] = []
    for v in vectors:
        segs, groups = vector_to_segments(v, bezier_sample_count=bezier_sample_count)
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


def dedupe_exact_points(
    segments: list[Segment],
) -> tuple[list[Point], list[tuple[int, int]], dict[Point, int]]:
    """Reduces every segment endpoint to a canonical node index, collapsing
    only bit-identical points (a plain `dict[Point, int]` keyed by literal
    `(x, y)` value) -- never a tolerance merge. This is the minimum
    dedup needed for basic connectivity: two segments sharing one item's
    own point (consecutive bezier samples, a rect/quad's own shared
    corner) must land on the same node, or even one item's own polyline
    wouldn't be connected. Any two points that are merely *close* (not
    identical) are left as distinct nodes -- see `connect_nearby_points`
    for how those get wired together instead of merged.

    Also returns the raw-point -> node-index map itself, so a caller can
    look up which final node a known original-data point ended up as
    (`graph.py`'s protected-node computation)."""
    nodes: list[Point] = []
    point_to_node: dict[Point, int] = {}

    def node_for(p: Point) -> int:
        idx = point_to_node.get(p)
        if idx is None:
            idx = len(nodes)
            nodes.append(p)
            point_to_node[p] = idx
        return idx

    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for seg in segments:
        a = node_for(seg[0])
        b = node_for(seg[1])
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        edges.append(key)
    return nodes, edges, point_to_node


def connect_nearby_points(
    nodes: list[Point], edges: list[tuple[int, int]], epsilon: float,
) -> list[tuple[int, int]]:
    """`edges` plus one new edge per pair of DISTINCT nodes within
    `epsilon` of each other that isn't already connected -- the
    "connect, don't merge" replacement for tolerance-based point merging.
    Node positions are never touched; this only ever adds edges. Uses
    `scipy.spatial.cKDTree.query_pairs` for the within-`epsilon` pair
    search (`O(n log n)`-ish rather than the all-pairs `O(n^2)` a naive
    scan would need)."""
    if len(nodes) < 2:
        return list(edges)
    existing = {(a, b) if a < b else (b, a) for a, b in edges}
    tree = cKDTree(nodes)
    out = list(edges)
    for i, j in tree.query_pairs(epsilon):
        key = (i, j) if i < j else (j, i)
        if key in existing:
            continue
        existing.add(key)
        out.append(key)
    return out
