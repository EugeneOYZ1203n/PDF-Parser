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

# Beziers are approximated by 8 straight lines, per the algorithm spec
# (docs/cad_font_vector_recognition.md, step 2).
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


def flatten_item_to_segments(item: tuple, curve_samples: int = CURVE_SAMPLE_COUNT) -> list[Segment]:
    """One `Vector.items` entry -> ordered straight-line segments.

    `"l"` -> one segment. `"c"` -> `curve_samples` segments via
    `cubic_bezier_points`. `"re"`/`"qu"` -> 4 edges closing the polygon (all
    4 sides -- not the 2-perpendicular-edge shortcut
    `similarity_single_line_cad_text.ipynb` uses; planarity/graph
    construction needs the real closed shape)."""
    kind = item[0]
    if kind == "l":
        return [(item[1], item[2])]
    if kind == "c":
        pts = cubic_bezier_points(item[1], item[2], item[3], item[4], n=curve_samples)
        return list(zip(pts, pts[1:]))
    if kind == "re":
        x0, y0, x1, y1 = item[1]
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        return list(zip(corners, corners[1:] + corners[:1]))
    if kind == "qu":
        corners = list(item[1])
        return list(zip(corners, corners[1:] + corners[:1]))
    return []


def vector_to_segments(v, curve_samples: int = CURVE_SAMPLE_COUNT) -> list[Segment]:
    """Every item of one `Vector`, flattened and concatenated in item order."""
    segments: list[Segment] = []
    for item in v.items:
        segments.extend(flatten_item_to_segments(item, curve_samples=curve_samples))
    return segments


def vectors_to_segments(vectors, curve_samples: int = CURVE_SAMPLE_COUNT) -> list[Segment]:
    """`vector_to_segments` over every vector, concatenated in order."""
    out: list[Segment] = []
    for v in vectors:
        out.extend(vector_to_segments(v, curve_samples=curve_samples))
    return out


class _UnionFind:
    """Tiny local union-find -- deliberately not importing
    `commons/helpers/clustering.py`'s private `_UnionFind`, which isn't
    exported for reuse outside that module."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _seg_vector(seg: Segment) -> Point:
    (x0, y0), (x1, y1) = seg
    return (x1 - x0, y1 - y0)


def _seg_length(seg: Segment) -> float:
    dx, dy = _seg_vector(seg)
    return (dx * dx + dy * dy) ** 0.5


def _angle_diff_mod_180(a: float, b: float) -> float:
    """Smallest difference between two direction angles (degrees) treating
    a line's direction as sign-agnostic (mod 180)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _point_to_line_distance(p: Point, line_origin: Point, unit_dir: Point) -> float:
    px, py = p[0] - line_origin[0], p[1] - line_origin[1]
    dx, dy = unit_dir
    return abs(px * dy - py * dx)


def merge_colinear_segments(
    segments: list[Segment],
    *, angle_tol_deg: float = 2.0, perp_tol: float = 0.75, gap_tol: float = 1.0,
) -> list[Segment]:
    """Merges segments lying on (approximately) the same infinite line and
    touching/overlapping along it into single longer segments -- a
    touching/overlapping merge only, never a "collinear anywhere on the
    page" merge.

    Union-find over all segment pairs `(i, j)` where: (a) their direction
    angles differ by at most `angle_tol_deg` (mod 180 -- a line has no
    intrinsic direction sign), (b) both of j's endpoints are within
    `perp_tol` of i's own infinite line, and (c) projecting both segments
    onto i's direction, their projected intervals overlap or are within
    `gap_tol` of each other. Each final group's merged segment is the two
    most extreme points when every member's endpoints are projected onto
    the group's own (first-member) direction."""
    n = len(segments)
    if n <= 1:
        return list(segments)

    angles = []
    units = []
    for seg in segments:
        dx, dy = _seg_vector(seg)
        length = math.hypot(dx, dy)
        if length < 1e-9:
            angles.append(0.0)
            units.append((1.0, 0.0))
        else:
            angles.append(math.degrees(math.atan2(dy, dx)))
            units.append((dx / length, dy / length))

    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if _angle_diff_mod_180(angles[i], angles[j]) > angle_tol_deg:
                continue
            origin_i = segments[i][0]
            unit_i = units[i]
            if _point_to_line_distance(segments[j][0], origin_i, unit_i) > perp_tol:
                continue
            if _point_to_line_distance(segments[j][1], origin_i, unit_i) > perp_tol:
                continue
            # projections onto segment i's own direction
            len_i = _seg_length(segments[i])

            def proj(p: Point) -> float:
                return (p[0] - origin_i[0]) * unit_i[0] + (p[1] - origin_i[1]) * unit_i[1]

            t0, t1 = proj(segments[j][0]), proj(segments[j][1])
            j_lo, j_hi = min(t0, t1), max(t0, t1)
            # segment i's own projected range is [0, len_i]
            if j_lo > len_i + gap_tol or j_hi < -gap_tol:
                continue
            uf.union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(n):
        groups.setdefault(uf.find(idx), []).append(idx)

    merged: list[Segment] = []
    for member_idxs in groups.values():
        if len(member_idxs) == 1:
            merged.append(segments[member_idxs[0]])
            continue
        origin = segments[member_idxs[0]][0]
        unit = units[member_idxs[0]]

        def proj(p: Point) -> float:
            return (p[0] - origin[0]) * unit[0] + (p[1] - origin[1]) * unit[1]

        all_pts = [p for idx in member_idxs for p in segments[idx]]
        projections = [(proj(p), p) for p in all_pts]
        p_min = min(projections, key=lambda pr: pr[0])[1]
        p_max = max(projections, key=lambda pr: pr[0])[1]
        merged.append((p_min, p_max))
    return merged


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
    segments: list[Segment], *, point_merge_tol: float = 0.5,
) -> tuple[list[Point], list[tuple[int, int]]]:
    """Reduces every segment endpoint to a canonical node index, merging
    near-duplicate points (within `point_merge_tol`) into one node.

    Reuses `commons.helpers.clustering.cluster_spatial` over every segment
    endpoint, each treated as a zero-size bbox, rather than writing a
    second spatial clusterer. Each returned cluster becomes one node (its
    position the centroid of its member points); segment endpoints are
    re-mapped through the point -> node-index map to build deduped edges
    (self-loop edges -- both endpoints collapsing to the same node -- are
    dropped)."""
    all_points: list[Point] = [p for seg in segments for p in seg]
    if not all_points:
        return [], []

    clusters = cluster_spatial(
        all_points,
        get_bbox=lambda p: (p[0], p[1], p[0], p[1]),
        threshold=point_merge_tol,
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
    return nodes, edges
