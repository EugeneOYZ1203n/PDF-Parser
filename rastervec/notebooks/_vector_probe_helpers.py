"""Shared helpers for the three vector-geometry probe notebooks
(`vector_intersection_lab.ipynb`, `dashed_line_collinear_lab.ipynb`,
`parallel_groups_lab.ipynb`) -- experimental vector-classification signals,
kept out of `P3_Vector_Parsing/` until one of them proves useful.

The unit everywhere is one whole `Vector` (one `get_drawings()` drawing),
never an individual item:

- **crossings** -- `crossing_segment_counts(cluster)`: per Vector, how many
  segments of *other* Vectors in the same cluster *properly* (interior,
  X-shaped) cross at least one of its own segments
  (`crossing_partner_counts` is the older per-partner variant). Items are flattened to straight segments
  for the test (`vector_segments`: "l" as-is, "re"/"qu" as their 4 edges,
  "c" as a `curve_steps`-segment polyline); touching endpoints, corners and
  T-junctions don't count, and a Vector crossing its own items doesn't
  count.
- **straight Vectors** -- `straight_line(v)`: a Vector whose items are all
  "l" and whose points all lie within `tol` of one line. Only these take
  part in collinear/parallel grouping; everything else is "excluded".
- **grouping** -- `group_parallel` (single-linkage on the folded [0, 180)
  angle, wrap-aware -- see `_chain_angles`) and `group_collinear`
  (`group_parallel`, then *anchored* grouping on each line's perpendicular
  offset `rho` within each angle group -- every member within `tol` of its
  group's smallest offset, so a group never spans more than `tol`; see
  `_anchored_1d` for why offsets can't be single-linkage). "Same infinite
  line", no gap limit.
- **curved chains** -- `chain_curves`: good-continuation chains of `Piece`s
  (`as_piece`: a straight Vector, or an all-"l" open polyline with every
  internal turn <= `max_turn_deg`). Greedy from the longest seeds; a link
  needs only a bounded turn (piece direction and connector) within
  `reach_factor` x the last piece's length -- no gap-consistency test.
  Optional `max_len_ratio` also caps each piece's length vs the chain's
  running median member length.
- **corner merge** -- `merge_at_corners`: union-find over >= 2-member
  collinear groups whose infinite lines cross where both have an endpoint
  (or where an L-shaped polyline bridges the corner).

Colour scales use one shared low -> high gradient, `gradient4` (green ->
yellow -> red -> purple). `summarise_groups`/`add_gradient_layers`/
`group_histograms` are the group stats, gradient layers and histograms
shared by the collinear/parallel notebooks.

`DebugReport` writes a folder `scripts/pipeline_report_viewer.py` opens
directly: one single-page PDF per layer plus a `manifest.json` in the exact
shape the viewer's `_Panel` reads.
"""
from __future__ import annotations

import colorsys
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from rastervec.commons.models import PageMeta, Vector
from rastervec.commons.paths import output_dir
from rastervec.commons.renderer.pdf import render_boxes_pdf, render_vectors_pdf
from rastervec.P1_Reading_Native.reader import Reader
from rastervec.P1_Reading_Native.vector_extract import extract_vectors
from rastervec.P3_Vector_Parsing.LatestVectorClassification.classify_vectors import classify_vectors
from rastervec.P3_Vector_Parsing.LatestVectorClassification.layer_color_separation import (
    separate_by_color,
    separate_by_layer,
    separate_by_width,
)

Point = tuple[float, float]
Segment = tuple[float, float, float, float]  # x0, y0, x1, y1
BucketKey = tuple  # (layer, color-key, width-key)


# --------------------------------------------------------------------------
# Pipeline front half: extract -> bucket -> (cluster)
# --------------------------------------------------------------------------

def load_page_vectors(pdf_path: str, page_index: int) -> tuple[PageMeta, list[Vector]]:
    """Phase-1 raw vectors for one page. `Vector`s are plain data, so the
    document is closed again before returning."""
    with Reader(str(pdf_path)) as reader:
        page = reader.get_page(page_index)
        vectors = extract_vectors(page)
        return page.meta, vectors


def bucket_vectors(vectors: list[Vector]) -> dict[BucketKey, list[Vector]]:
    """layer -> color -> width buckets, flattened to one dict -- the same
    three P3 separation functions `classify_vectors` uses."""
    out: dict[BucketKey, list[Vector]] = {}
    for layer, by_layer in separate_by_layer(vectors).items():
        for color, by_color in separate_by_color(by_layer).items():
            for width, by_width in separate_by_width(by_color).items():
                out[(layer, color, width)] = by_width
    return out


def cluster_vectors(vectors: list[Vector], page_meta: PageMeta) -> list[tuple[BucketKey, list[Vector]]]:
    """Run P3 LatestVectorClassification's own classification chain
    (layer/color/width buckets -> collinear drawing removal -> seqno-overlap
    merge -> constrained spatial clustering) and return every cluster of its
    "Spatial cluster" step -- before the length-outlier and crossings steps
    drop anything, so the labs measure the clusters those steps see -- as
    `(bucket_key, flat Vector list)`."""

    class _Page:  # classify_vectors only threads `page` through, never reads it
        meta = page_meta

    result = classify_vectors(vectors, _Page())
    clusters: list[tuple[BucketKey, list[Vector]]] = []
    for key, stage in result.clustering.items():
        if not stage.steps:
            continue
        step = next((s for s in stage.steps if s.label == "Spatial cluster"), stage.steps[-1])
        for cluster in step.categories["kept"].groups:
            clusters.append((key, [v for group in cluster for v in group]))
    return clusters


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def _cubic(p0: Point, p1: Point, p2: Point, p3: Point, steps: int) -> list[Point]:
    pts = []
    for i in range(steps + 1):
        t = i / steps
        u = 1.0 - t
        a, b, c, d = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
        pts.append((
            a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
            a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
        ))
    return pts


def _ring(points: list[Point]) -> list[Segment]:
    return [
        (*points[i], *points[(i + 1) % len(points)])
        for i in range(len(points))
    ]


def vector_segments(v: Vector, curve_steps: int = 8) -> list[Segment]:
    """Every item of `v` as straight segments: "l" as-is, "re" (x0, y0, x1,
    y1) and "qu" (ul, ur, lr, ll -- `plain_item`'s order) as their 4 edges,
    "c" flattened to `curve_steps` segments. Zero-length segments dropped."""
    segs: list[Segment] = []
    for item in v.items:
        kind = item[0]
        if kind == "l":
            segs.append((*item[1], *item[2]))
        elif kind == "re":
            x0, y0, x1, y1 = item[1]
            segs.extend(_ring([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
        elif kind == "qu":
            segs.extend(_ring(list(item[1])))
        elif kind == "c":
            pts = _cubic(item[1], item[2], item[3], item[4], curve_steps)
            segs.extend((*pts[i], *pts[i + 1]) for i in range(len(pts) - 1))
    return [s for s in segs if s[0] != s[2] or s[1] != s[3]]


def _signed_dist(seg: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Signed perpendicular distance of points (px, py) from each segment's
    supporting line; `seg` (n, 4), px/py broadcastable against (n, ...)."""
    dx = seg[..., 2] - seg[..., 0]
    dy = seg[..., 3] - seg[..., 1]
    length = np.hypot(dx, dy)
    length = np.where(length == 0, 1.0, length)
    return (dx * (py - seg[..., 1]) - dy * (px - seg[..., 0])) / length


def _crossing_matrix(a: np.ndarray, b: np.ndarray, eps: float = 0.01) -> np.ndarray:
    """Bool (n, m): `[i, j]` is True when segment `a[i]` properly crosses
    segment `b[j]` -- each segment's endpoints lie strictly on opposite
    sides of the other's line, each by more than `eps` pt. So an endpoint
    touching the other segment (T-junction, shared corner, a dash end
    resting on a line) is never a crossing, and neither is collinear
    overlap."""
    out = np.zeros((len(a), len(b)), dtype=bool)
    if len(a) == 0 or len(b) == 0:
        return out
    A = a[:, None, :]  # (n, 1, 4)
    B = b[None, :, :]  # (1, m, 4)
    # Cheap per-pair bbox reject first.
    overlap = (
        (np.minimum(A[..., 0], A[..., 2]) <= np.maximum(B[..., 0], B[..., 2]))
        & (np.minimum(B[..., 0], B[..., 2]) <= np.maximum(A[..., 0], A[..., 2]))
        & (np.minimum(A[..., 1], A[..., 3]) <= np.maximum(B[..., 1], B[..., 3]))
        & (np.minimum(B[..., 1], B[..., 3]) <= np.maximum(A[..., 1], A[..., 3]))
    )
    if not overlap.any():
        return out
    ia, ib = np.nonzero(overlap)
    sa, sb = a[ia], b[ib]
    d1 = _signed_dist(sa, sb[:, 0], sb[:, 1])
    d2 = _signed_dist(sa, sb[:, 2], sb[:, 3])
    d3 = _signed_dist(sb, sa[:, 0], sa[:, 1])
    d4 = _signed_dist(sb, sa[:, 2], sa[:, 3])
    out[ia, ib] = (
        (d1 * d2 < 0) & (np.abs(d1) > eps) & (np.abs(d2) > eps)
        & (d3 * d4 < 0) & (np.abs(d3) > eps) & (np.abs(d4) > eps)
    )
    return out


def segments_properly_cross(a: np.ndarray, b: np.ndarray, eps: float = 0.01) -> bool:
    """True if any segment of `a` (n, 4) properly crosses any segment of
    `b` (m, 4) -- see `_crossing_matrix`."""
    return bool(_crossing_matrix(a, b, eps).any())


def _overlapping_pairs(bboxes: list[tuple[float, float, float, float]]):
    """Sweep-and-prune over x: every (i, j), i < j, whose bboxes overlap."""
    order = sorted(range(len(bboxes)), key=lambda i: bboxes[i][0])
    active: list[int] = []
    for i in order:
        x0, y0, x1, y1 = bboxes[i]
        active = [j for j in active if bboxes[j][2] >= x0]
        for j in active:
            if bboxes[j][1] <= y1 and y0 <= bboxes[j][3]:
                yield (min(i, j), max(i, j))
        active.append(i)


def _cluster_segments(cluster: list[Vector], curve_steps: int) -> tuple[list[np.ndarray], list[tuple]]:
    """Per Vector: its (n, 4) segment array and that array's bbox (an
    inverted, never-overlapping bbox for a Vector with no segments)."""
    segs = [np.asarray(vector_segments(v, curve_steps), dtype=float).reshape(-1, 4) for v in cluster]
    bboxes = []
    for s in segs:
        if len(s):
            xs, ys = s[:, [0, 2]], s[:, [1, 3]]
            bboxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
        else:
            bboxes.append((math.inf, math.inf, -math.inf, -math.inf))
    return segs, bboxes


def crossing_partner_counts(
    cluster: list[Vector], *, eps: float = 0.01, curve_steps: int = 8,
) -> list[int]:
    """Per Vector of `cluster` (same order), the number of *other* Vectors
    in `cluster` it properly crosses at least once (see
    `segments_properly_cross`)."""
    segs, bboxes = _cluster_segments(cluster, curve_steps)
    counts = [0] * len(cluster)
    for i, j in _overlapping_pairs(bboxes):
        if segments_properly_cross(segs[i], segs[j], eps):
            counts[i] += 1
            counts[j] += 1
    return counts


def crossing_segment_counts(
    cluster: list[Vector], *, eps: float = 0.01, curve_steps: int = 8,
) -> list[int]:
    """Per Vector of `cluster` (same order), the number of distinct
    segments (`vector_segments` pieces -- so a curve contributes up to
    `curve_steps`) belonging to *other* Vectors in `cluster` that properly
    cross at least one of its own segments. One foreign segment crossing
    two of this Vector's segments counts once; its own segments never
    count."""
    segs, bboxes = _cluster_segments(cluster, curve_steps)
    counts = [0] * len(cluster)
    for i, j in _overlapping_pairs(bboxes):
        m = _crossing_matrix(segs[i], segs[j], eps)
        counts[i] += int(m.any(axis=0).sum())  # distinct j-segments crossing i
        counts[j] += int(m.any(axis=1).sum())  # distinct i-segments crossing j
    return counts


@dataclass(frozen=True)
class LineFit:
    """A straight Vector's line: `angle` folded to [0, 180) degrees (page
    space, y down), `midpoint` of its extent, `length` = extent along the
    line (t_max - t_min)."""

    angle: float
    midpoint: Point
    length: float


def straight_line(v: Vector, tol: float = 0.25) -> LineFit | None:
    """`LineFit` if every item of `v` is "l" and every point lies within
    `tol` pt of one line (through the two mutually farthest points), else
    `None`. A zero-extent Vector is not straight."""
    if not v.items or any(item[0] != "l" for item in v.items):
        return None
    pts = np.asarray([p for item in v.items for p in item[1:3]], dtype=float)
    first = pts[0]
    p0 = pts[np.argmax(np.hypot(*(pts - first).T))]
    p1 = pts[np.argmax(np.hypot(*(pts - p0).T))]
    span = float(np.hypot(*(p1 - p0)))
    if span == 0.0:
        return None
    angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])) % 180.0
    ux, uy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    rel = pts - p0
    if np.max(np.abs(rel[:, 0] * uy - rel[:, 1] * ux)) > tol:
        return None
    t = rel[:, 0] * ux + rel[:, 1] * uy
    t_min, t_max = float(t.min()), float(t.max())
    t_mid = (t_min + t_max) / 2.0
    return LineFit(
        angle=angle,
        midpoint=(float(p0[0] + t_mid * ux), float(p0[1] + t_mid * uy)),
        length=t_max - t_min,
    )


def split_straight(vectors: list[Vector], tol: float = 0.25) -> tuple[list[tuple[Vector, LineFit]], list[Vector]]:
    """`(straight, excluded)` -- `(Vector, LineFit)` pairs for straight
    Vectors, plain Vectors for everything else."""
    straight, excluded = [], []
    for v in vectors:
        fit = straight_line(v, tol)
        if fit is None:
            excluded.append(v)
        else:
            straight.append((v, fit))
    return straight, excluded


def _anchored_1d(values: list[float], tol: float) -> list[list[int]]:
    """Anchored grouping on a line: indices sorted by value; a group starts
    at its smallest value and takes every following value within `tol` of
    that anchor, so a group's total span is at most `tol`. Deliberately
    *not* single-linkage: on a real page (240118 p0) dense hatching put
    2227 diagonal segments' perpendicular offsets 0.06 pt apart (median),
    and single-linkage chained them into one 218 pt-wide "collinear" group."""
    order = sorted(range(len(values)), key=values.__getitem__)
    groups: list[list[int]] = []
    for i in order:
        if groups and values[i] - values[groups[-1][0]] <= tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _chain_angles(angles: list[float], tol: float) -> list[list[int]]:
    """Single-linkage on the [0, 180) circle: angles sorted, split wherever
    the gap to the next exceeds `tol`; the first and last groups merge when
    the wrap-around gap (last -> 180 -> first) is also within `tol`.
    Single-linkage (not `_anchored_1d`) on purpose: short CAD segments carry
    real angle jitter -- 240118 p0's 6 pt hatch segments spread over
    134-136 deg -- and a hard 1 deg span cap cut one hatch into 3 groups.
    The cost: a polygonised arc drawn as separate straight Vectors < `tol`
    apart chains into one group."""
    order = sorted(range(len(angles)), key=angles.__getitem__)
    groups: list[list[int]] = []
    for i in order:
        if groups and angles[i] - angles[groups[-1][-1]] <= tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    if len(groups) > 1 and angles[groups[0][0]] + 180.0 - angles[groups[-1][-1]] <= tol:
        groups[0] = groups.pop() + groups[0]
    return groups


def _mean_axis_angle(angles: list[float]) -> float:
    """Circular mean of axial angles (period 180), via doubled angles."""
    s = sum(math.sin(math.radians(2 * a)) for a in angles)
    c = sum(math.cos(math.radians(2 * a)) for a in angles)
    return (math.degrees(math.atan2(s, c)) / 2.0) % 180.0


def group_parallel(
    straight: list[tuple[Vector, LineFit]], angle_tol: float = 1.0,
) -> list[list[tuple[Vector, LineFit]]]:
    """Single-linkage groups of straight Vectors by folded angle
    (`_chain_angles` -- wrap-aware, so 179.8 deg and 0.1 deg join)."""
    angles = [fit.angle for _, fit in straight]
    return [[straight[i] for i in g] for g in _chain_angles(angles, angle_tol)]


def group_collinear(
    straight: list[tuple[Vector, LineFit]], angle_tol: float = 1.0, offset_tol: float = 0.5,
) -> list[list[tuple[Vector, LineFit]]]:
    """Same-infinite-line groups: `group_parallel`, then within each angle
    group anchored grouping on each line's perpendicular offset `rho` (its
    midpoint projected onto the group's mean-angle normal) within
    `offset_tol` pt. No along-line gap limit."""
    out = []
    for group in group_parallel(straight, angle_tol):
        theta = math.radians(_mean_axis_angle([fit.angle for _, fit in group]))
        nx, ny = -math.sin(theta), math.cos(theta)
        rhos = [fit.midpoint[0] * nx + fit.midpoint[1] * ny for _, fit in group]
        out.extend([group[i] for i in g] for g in _anchored_1d(rhos, offset_tol))
    return out


def group_stats(group: list[tuple[Vector, LineFit]]) -> tuple[int, float]:
    """`(member count, population std of member lengths in pt)`."""
    lengths = [fit.length for _, fit in group]
    return len(lengths), float(np.std(lengths)) if lengths else 0.0


# --------------------------------------------------------------------------
# Curved chains: good continuation, direction drifts slowly
# --------------------------------------------------------------------------

_JOIN_EPS = 0.01  # pt: endpoints closer than this are the same point


def _turn(a: float, b: float) -> float:
    """Signed turn from direction `a` to direction `b` (degrees, full
    circle), in [-180, 180)."""
    return (b - a + 180.0) % 360.0 - 180.0


def _direction(p: Point, q: Point) -> float:
    """Direction of travel p -> q in degrees, [0, 360)."""
    return math.degrees(math.atan2(q[1] - p[1], q[0] - p[0])) % 360.0


@dataclass(frozen=True)
class Piece:
    """One chainable Vector: a straight line or a smooth open polyline.
    `start_dir`/`end_dir` point *out of* the piece at that end (so the
    travel direction entering at `start` is `start_dir + 180`); `length` is
    the path length; `angle` the folded [0, 180) chord angle."""

    start: Point
    end: Point
    start_dir: float
    end_dir: float
    length: float
    angle: float

    def reversed(self) -> "Piece":
        return Piece(self.end, self.start, self.end_dir, self.start_dir, self.length, self.angle)


def _order_polyline(segments: list[tuple[Point, Point]]) -> list[Point] | None:
    """Vertices of one open, unbranched path through every segment (any
    item order/orientation), or `None` if the segments don't form one."""
    rest = list(segments)
    path = list(rest.pop(0))

    def same(p: Point, q: Point) -> bool:
        return math.hypot(p[0] - q[0], p[1] - q[1]) <= _JOIN_EPS

    while rest:
        for i, (p, q) in enumerate(rest):
            if same(p, path[-1]):
                path.append(q)
            elif same(q, path[-1]):
                path.append(p)
            elif same(q, path[0]):
                path.insert(0, p)
            elif same(p, path[0]):
                path.insert(0, q)
            else:
                continue
            rest.pop(i)
            break
        else:
            return None  # disconnected, or a branch off an interior vertex
    if same(path[0], path[-1]):
        return None  # closed
    return path


def as_piece(v: Vector, max_turn_deg: float = 15.0, straight_tol: float = 0.25) -> Piece | None:
    """`Piece` for a straight Vector (`straight_line`) or an all-"l" open
    polyline whose every internal turn is <= `max_turn_deg`; else `None`."""
    fit = straight_line(v, straight_tol)
    if fit is not None:
        ux, uy = math.cos(math.radians(fit.angle)), math.sin(math.radians(fit.angle))
        h = fit.length / 2.0
        start = (fit.midpoint[0] - h * ux, fit.midpoint[1] - h * uy)
        end = (fit.midpoint[0] + h * ux, fit.midpoint[1] + h * uy)
        return Piece(start, end, (fit.angle + 180.0) % 360.0, fit.angle % 360.0, fit.length, fit.angle)
    if not is_all_lines(v):
        return None
    segs = [(tuple(item[1]), tuple(item[2])) for item in v.items]
    segs = [(p, q) for p, q in segs if math.hypot(q[0] - p[0], q[1] - p[1]) > _JOIN_EPS]
    if not segs:
        return None
    path = _order_polyline(segs)
    if path is None:
        return None
    dirs = [_direction(path[i], path[i + 1]) for i in range(len(path) - 1)]
    if any(abs(_turn(a, b)) > max_turn_deg for a, b in zip(dirs, dirs[1:])):
        return None
    length = sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                 for i in range(len(path) - 1))
    return Piece(
        path[0], path[-1], (dirs[0] + 180.0) % 360.0, dirs[-1], length,
        _direction(path[0], path[-1]) % 180.0,
    )


def split_pieces(
    vectors: list[Vector], max_turn_deg: float = 15.0, straight_tol: float = 0.25,
) -> tuple[list[tuple[Vector, Piece]], list[Vector]]:
    """`(pieces, excluded)` -- the `split_straight` shape, over `as_piece`."""
    pieces, excluded = [], []
    for v in vectors:
        p = as_piece(v, max_turn_deg, straight_tol)
        if p is None:
            excluded.append(v)
        else:
            pieces.append((v, p))
    return pieces, excluded


def _link_cost(tip: Point, tip_dir: float, near: Point, b_in: float,
               max_turn_deg: float, reach: float) -> tuple[float, float] | None:
    """`(gap, turn)` cost (compared as a tuple: nearest wins, smoother
    breaks ties -- skipping a nearer valid piece would orphan it) of continuing a chain whose free end is `tip` (travelling in
    `tip_dir`) into a piece entered at `near` travelling `b_in`; `None` when
    rejected. No gap test beyond `reach`: the piece's own direction, and the
    connector tip -> near (skipped for touching ends), must each turn by at
    most `max_turn_deg` -- the connector check stops a hop sideways onto a
    parallel neighbour."""
    gap = math.hypot(near[0] - tip[0], near[1] - tip[1])
    if gap > reach:
        return None
    turn = abs(_turn(tip_dir, b_in))
    if turn > max_turn_deg:
        return None
    if gap < _JOIN_EPS:
        return (gap, turn)
    conn = _direction(tip, near)
    t1, t2 = abs(_turn(tip_dir, conn)), abs(_turn(conn, b_in))
    if t1 > max_turn_deg or t2 > max_turn_deg:
        return None
    return (gap, turn + t1)


def chain_curves(
    pieces: list[tuple[Vector, Piece]], max_turn_deg: float = 15.0, reach_factor: float = 3.0,
    *, max_len_ratio: float | None = None,
) -> list[list[tuple[Vector, Piece]]]:
    """Greedy good-continuation chains. Seeds in descending length; each
    grows from both ends, at every step taking the lowest-`_link_cost`
    (nearest, then smoothest) unclaimed piece with an end within `reach_factor` x the last piece's
    length. Each piece joins at most one chain. Returned pieces are
    oriented in walk order (so `chain_total_turn` is meaningful);
    singletons are returned too.

    `max_len_ratio` (None = off): a candidate is also skipped when its
    length and the chain's running median member length (seed included,
    both growth directions) differ by more than that factor. Seeds go
    longest-first, so a long solid line rejects short dashes and stays a
    singleton; the dashes then chain from their own seeds."""
    from scipy.spatial import cKDTree

    if not pieces:
        return []
    ends = np.asarray([pt for _, p in pieces for pt in (p.start, p.end)], dtype=float)
    tree = cKDTree(ends)
    claimed = [False] * len(pieces)

    def len_ok(length: float, lengths: list[float]) -> bool:
        if max_len_ratio is None:
            return True
        med = float(np.median(lengths))
        return max(length, med) / max(min(length, med), 1e-9) <= max_len_ratio

    def grow(last: Piece, lengths: list[float]) -> list[tuple[Vector, Piece]]:
        out = []
        while True:
            reach = reach_factor * last.length
            tip, tip_dir = last.end, last.end_dir
            best = None
            for k in tree.query_ball_point(tip, reach + _JOIN_EPS):
                j, at_end = divmod(k, 2)
                if claimed[j]:
                    continue
                cand = pieces[j][1].reversed() if at_end else pieces[j][1]
                if not len_ok(cand.length, lengths):
                    continue
                cost = _link_cost(tip, tip_dir, cand.start, (cand.start_dir + 180.0) % 360.0,
                                  max_turn_deg, reach)
                if cost is not None and (best is None or cost < best[0]):
                    best = (cost, j, cand)
            if best is None:
                return out
            _, j, last = best
            claimed[j] = True
            lengths.append(last.length)
            out.append((pieces[j][0], last))

    chains = []
    for i in sorted(range(len(pieces)), key=lambda i: -pieces[i][1].length):
        if claimed[i]:
            continue
        claimed[i] = True
        v, seed = pieces[i]
        lengths = [seed.length]
        forward = grow(seed, lengths)
        backward = grow(seed.reversed(), lengths)
        chains.append([(bv, bp.reversed()) for bv, bp in reversed(backward)] + [(v, seed)] + forward)
    return chains


def chain_total_turn(chain: list[tuple[Vector, Piece]]) -> float:
    """Signed total turn (degrees) along a walk-ordered chain: within each
    piece (entry -> exit direction) plus every link (exit -> next entry)."""
    total, prev_exit = 0.0, None
    for _, p in chain:
        entry = (p.start_dir + 180.0) % 360.0
        if prev_exit is not None:
            total += _turn(prev_exit, entry)
        total += _turn(entry, p.end_dir)
        prev_exit = p.end_dir
    return total


# --------------------------------------------------------------------------
# Corner merge: collinear groups whose lines meet at their intersection
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GroupLine:
    """A collinear group's line: folded mean `angle`, unit normal `normal`,
    offset `rho` (= p . normal for points on it), and every member
    endpoint (`ends`, (n, 2))."""

    angle: float
    normal: Point
    rho: float
    ends: np.ndarray

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (float(self.ends[:, 0].min()), float(self.ends[:, 1].min()),
                float(self.ends[:, 0].max()), float(self.ends[:, 1].max()))


def group_line(group: list[tuple[Vector, LineFit]]) -> GroupLine:
    angle = _mean_axis_angle([fit.angle for _, fit in group])
    nx, ny = -math.sin(math.radians(angle)), math.cos(math.radians(angle))
    ends = []
    for _, fit in group:
        ux, uy = math.cos(math.radians(fit.angle)), math.sin(math.radians(fit.angle))
        h = fit.length / 2.0
        ends.append((fit.midpoint[0] - h * ux, fit.midpoint[1] - h * uy))
        ends.append((fit.midpoint[0] + h * ux, fit.midpoint[1] + h * uy))
    ends_arr = np.asarray(ends, dtype=float)
    rho = float(np.mean(ends_arr @ np.asarray([nx, ny])))
    return GroupLine(angle, (nx, ny), rho, ends_arr)


def _axis_diff(a: float, b: float) -> float:
    """Unsigned difference of two folded [0, 180) angles, in [0, 90]."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _line_intersection(g1: GroupLine, g2: GroupLine, min_angle: float) -> Point | None:
    """Where the two infinite lines cross; `None` when they're within
    `min_angle` of parallel (ill-conditioned / far away)."""
    if _axis_diff(g1.angle, g2.angle) < min_angle:
        return None
    a = np.asarray([g1.normal, g2.normal], dtype=float)
    x, y = np.linalg.solve(a, np.asarray([g1.rho, g2.rho]))
    return (float(x), float(y))


def _min_dist(points: np.ndarray, p: Point) -> float:
    return float(np.min(np.hypot(points[:, 0] - p[0], points[:, 1] - p[1])))


def _point_seg_dist(points: np.ndarray, a: Point, b: Point) -> float:
    """Min distance from any of `points` (n, 2) to segment a-b."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    ll = dx * dx + dy * dy
    t = np.zeros(len(points)) if ll == 0 else np.clip(
        ((points[:, 0] - ax) * dx + (points[:, 1] - ay) * dy) / ll, 0.0, 1.0)
    return float(np.min(np.hypot(points[:, 0] - (ax + t * dx), points[:, 1] - (ay + t * dy))))


def _on_line(seg: tuple[Point, Point], g: GroupLine, angle_tol: float, offset_tol: float) -> bool:
    p, q = seg
    if _axis_diff(_direction(p, q) % 180.0, g.angle) > angle_tol:
        return False
    nx, ny = g.normal
    return all(abs(pt[0] * nx + pt[1] * ny - g.rho) <= offset_tol for pt in seg)


@dataclass
class CornerMerge:
    """`groups`: merged components (>= 2 source collinear groups), each a
    flat `(Vector, LineFit | None)` list -- bridging polylines carry
    `None`. `points`: every accepted intersection. `bridges`: the polylines
    that bridged a corner."""

    groups: list[list[tuple[Vector, LineFit | None]]]
    source_counts: list[int]
    points: list[Point]
    bridges: list[Vector]


def merge_at_corners(
    groups: list[list[tuple[Vector, LineFit]]], polylines: list[Vector], *,
    angle_tol: float = 1.0, offset_tol: float = 0.5, corner_tol: float = 1.0, min_angle: float = 10.0,
) -> CornerMerge:
    """Union-find merge of >= 2-member collinear `groups` (one bucket) whose
    lines meet, by either rule:

    - **endpoint**: each group has a member endpoint within `corner_tol` of
      the lines' intersection;
    - **corner piece**: a `polylines` Vector has a segment on each group's
      line (`angle_tol`/`offset_tol`), sharing a vertex within `corner_tol`
      of the intersection, and each group has an endpoint within
      `corner_tol` of its arm. The polyline joins the merged group.
    """
    multi = [g for g in groups if len(g) >= 2]
    lines = [group_line(g) for g in multi]
    parent = list(range(len(multi)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    points: list[Point] = []
    padded = [(x0 - corner_tol, y0 - corner_tol, x1 + corner_tol, y1 + corner_tol)
              for x0, y0, x1, y1 in (ln.bbox for ln in lines)]
    for i, j in _overlapping_pairs(padded):
        x = _line_intersection(lines[i], lines[j], min_angle)
        if x is None:
            continue
        if _min_dist(lines[i].ends, x) <= corner_tol and _min_dist(lines[j].ends, x) <= corner_tol:
            parent[find(i)] = find(j)
            points.append(x)

    bridge_of: dict[int, list[Vector]] = {}
    bridges: list[Vector] = []
    boxes = np.asarray(padded, dtype=float).reshape(-1, 4)
    for v in polylines:
        segs = [(tuple(it[1]), tuple(it[2])) for it in v.items if it[0] == "l"]
        bridged = False
        for a in range(len(segs)):
            for b in range(a + 1, len(segs)):
                sa, sb = segs[a], segs[b]
                vertex = next((p for p in sa for q in sb
                               if math.hypot(p[0] - q[0], p[1] - q[1]) <= _JOIN_EPS), None)
                if vertex is None:
                    continue

                def near(seg):
                    xs, ys = [seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]]
                    hit = ((boxes[:, 0] <= max(xs)) & (min(xs) <= boxes[:, 2])
                           & (boxes[:, 1] <= max(ys)) & (min(ys) <= boxes[:, 3]))
                    return [k for k in np.nonzero(hit)[0]
                            if _on_line(seg, lines[k], angle_tol, offset_tol)
                            and _point_seg_dist(lines[k].ends, *seg) <= corner_tol]

                for i in near(sa):
                    for j in near(sb):
                        x = _line_intersection(lines[i], lines[j], min_angle)
                        if x is None or math.hypot(vertex[0] - x[0], vertex[1] - x[1]) > corner_tol:
                            continue
                        parent[find(i)] = find(j)
                        points.append(x)
                        bridge_of.setdefault(i, []).append(v)
                        bridged = True
        if bridged:
            bridges.append(v)

    comps: dict[int, list[int]] = {}
    for i in range(len(multi)):
        comps.setdefault(find(i), []).append(i)
    out, counts = [], []
    for members in comps.values():
        if len(members) < 2:
            continue
        merged: list[tuple[Vector, LineFit | None]] = [m for i in members for m in multi[i]]
        seen = set()
        for i in members:
            for v in bridge_of.get(i, []):
                if id(v) not in seen:
                    seen.add(id(v))
                    merged.append((v, None))
        out.append(merged)
        counts.append(len(members))
    return CornerMerge(out, counts, points, bridges)


# --------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------

RGB = tuple[float, float, float]
GREY: RGB = (0.55, 0.55, 0.55)
LIGHT_GREY: RGB = (0.8, 0.8, 0.8)

_GRADIENT_STOPS: list[RGB] = [
    (0.0, 0.7, 0.0),     # green  (t = 0)
    (1.0, 0.85, 0.0),    # yellow (t = 1/3)
    (1.0, 0.0, 0.0),     # red    (t = 2/3)
    (0.55, 0.0, 0.75),   # purple (t = 1)
]


def gradient4(t: float) -> RGB:
    """The one shared low -> high colour scale: green -> yellow -> red ->
    purple, stops evenly spaced at t = 0, 1/3, 2/3, 1, linear RGB in
    between; t is clamped to [0, 1]."""
    t = min(1.0, max(0.0, t))
    pos = t * (len(_GRADIENT_STOPS) - 1)
    k = min(int(pos), len(_GRADIENT_STOPS) - 2)
    f = pos - k
    return tuple(a + (b - a) * f for a, b in zip(_GRADIENT_STOPS[k], _GRADIENT_STOPS[k + 1]))


def percentile_buckets(values: list[float], k: int = 5) -> list[tuple[float, float, list[int]]]:
    """`k` percentile buckets over `values` as `(lo, hi, indices)`, `lo`/
    `hi` the actual min/max of the members. Edges are
    `np.percentile(values, linspace(0, 100, k + 1))`; each value goes into
    the first bucket whose upper edge is >= it, so tied values never split
    across buckets. Buckets left empty by duplicate edges (heavy ties) are
    dropped, so fewer than `k` can come back."""
    if not values:
        return []
    edges = np.percentile(values, np.linspace(0, 100, k + 1))[1:]
    members: list[list[int]] = [[] for _ in range(k)]
    for i, v in enumerate(values):
        members[min(int(np.searchsorted(edges, v, side="left")), k - 1)].append(i)
    return [
        (min(values[i] for i in m), max(values[i] for i in m), m)
        for m in members if m
    ]


def exact_value_layers(values: list[int], tail_frac: float = 0.05) -> list[tuple[int, int, list[int]]]:
    """One `(lo, hi, indices)` layer per distinct value of `values`,
    ascending, except the high tail: walking down from the largest value,
    values join one merged `(lo, hi)` layer while that layer's combined
    member count stays < `tail_frac` of `len(values)`. Everything below the
    tail keeps its own `lo == hi` layer."""
    if not values:
        return []
    by_value: dict[int, list[int]] = {}
    for i, v in enumerate(values):
        by_value.setdefault(v, []).append(i)
    distinct = sorted(by_value)
    limit = tail_frac * len(values)
    tail_start, tail_count = len(distinct), 0
    while tail_start > 0 and tail_count + len(by_value[distinct[tail_start - 1]]) < limit:
        tail_start -= 1
        tail_count += len(by_value[distinct[tail_start]])
    layers = [(v, v, by_value[v]) for v in distinct[:tail_start]]
    if tail_start < len(distinct):
        tail = distinct[tail_start:]
        layers.append((tail[0], tail[-1], [i for v in tail for i in by_value[v]]))
    return layers


def is_all_lines(v: Vector) -> bool:
    """True if `v` has items and every one is a straight `"l"` (a polyline,
    not necessarily collinear -- no "re"/"qu"/"c")."""
    return bool(v.items) and all(item[0] == "l" for item in v.items)


def normalise(x: float, lo: float, hi: float) -> float:
    """`(x - lo) / (hi - lo)`, 0 when the range is empty."""
    return 0.0 if hi <= lo else (x - lo) / (hi - lo)


def golden_hues(k: int) -> list[float]:
    """`k` well-spread hues in [0, 1) (golden-ratio steps)."""
    return [(i * 0.618033988749895) % 1.0 for i in range(k)]


def angle_value(angle: float) -> float:
    """HSV value from a folded angle: triangle wave, 0/180 deg -> 0.25,
    90 deg -> 0.75, linear in between."""
    a = angle % 180.0
    return 0.25 + 0.5 * (1.0 - abs(a - 90.0) / 90.0)


def hsv_cluster_angle(hue: float, angle: float) -> RGB:
    return colorsys.hsv_to_rgb(hue, 1.0, angle_value(angle))


def to_hex(rgb: RGB) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(round(c * 255) for c in rgb))


# --------------------------------------------------------------------------
# Group summaries shared by the collinear/parallel notebooks' passes
# --------------------------------------------------------------------------

@dataclass
class GroupSummary:
    """`multi` = groups with >= 2 members (with per-group `counts`/`stds`
    and their page-wide `c_lo..c_hi` / `s_lo..s_hi` ranges), `singles` =
    the 1-member groups."""

    multi: list
    singles: list
    counts: list[int]
    stds: list[float]
    c_lo: int
    c_hi: int
    s_lo: float
    s_hi: float

    def describe(self, n_excluded: int, excluded_what: str) -> str:
        return (
            f"{len(self.multi) + len(self.singles)} group(s): {len(self.multi)} with >=2 members, "
            f"{len(self.singles)} singleton(s); {n_excluded} {excluded_what} excluded\n"
            f"count range {self.c_lo}..{self.c_hi}, "
            f"length-std range {self.s_lo:.3f}..{self.s_hi:.3f} pt"
        )


def summarise_groups(groups: list[list[tuple[Vector, LineFit]]]) -> GroupSummary:
    multi = [g for g in groups if len(g) >= 2]
    stats = [group_stats(g) for g in multi]
    counts = [c for c, _ in stats]
    stds = [sd for _, sd in stats]
    return GroupSummary(
        multi=multi,
        singles=[g for g in groups if len(g) == 1],
        counts=counts,
        stds=stds,
        c_lo=min(counts, default=0), c_hi=max(counts, default=0),
        s_lo=min(stds, default=0.0), s_hi=max(stds, default=0.0),
    )


def add_gradient_layers(
    report: "DebugReport", summary: GroupSummary, excluded: list[Vector], *,
    prefix: str = "", excluded_label: str = "excluded (non-straight)",
) -> None:
    """`<prefix>gradient` count + length-std layers (every multi-member
    group's members coloured `gradient4` from the page-wide min..max, drawn
    low-to-high so the extremes end up on top), then `<prefix>other`
    singletons (grey) + excluded (light grey)."""

    def _layer(label: str, values: list[float], lo: float, hi: float) -> None:
        colour, vecs = {}, []
        for i in sorted(range(len(summary.multi)), key=values.__getitem__):
            rgb = gradient4(normalise(values[i], lo, hi))
            for v, _ in summary.multi[i]:
                colour[id(v)] = rgb
                vecs.append(v)
        report.add_vectors(f"{prefix}gradient", label, gradient4(1.0), vecs, lambda v: colour[id(v)])

    _layer(f"vector count ({summary.c_lo}-{summary.c_hi})", summary.counts, summary.c_lo, summary.c_hi)
    _layer(f"length std ({summary.s_lo:.2f}-{summary.s_hi:.2f} pt)", summary.stds, summary.s_lo, summary.s_hi)
    report.add_vectors(f"{prefix}other", "singletons", GREY,
                       [v for g in summary.singles for v, _ in g], lambda v: GREY)
    report.add_vectors(f"{prefix}other", excluded_label, LIGHT_GREY, excluded, lambda v: LIGHT_GREY)


def group_histograms(summary: GroupSummary, title: str):
    """Two-panel figure: group size and group length std (>= 2 members),
    bars coloured with `gradient4` along their own axis."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    size_bins = (
        np.arange(2, max(summary.counts, default=2) + 2) - 0.5
        if max(summary.counts, default=0) < 60 else 40
    )
    for ax, values, bins in ((axes[0], summary.counts, size_bins), (axes[1], summary.stds, 40)):
        if not values:
            continue
        _, edges, patches = ax.hist(values, bins=bins, edgecolor="white")
        lo, hi = edges[0], edges[-1]
        for left, right, patch in zip(edges[:-1], edges[1:], patches):
            patch.set_facecolor(gradient4(normalise((left + right) / 2, lo, hi)))
    axes[0].set_title(f"{title} group size (>=2 members; {len(summary.singles)} singletons not shown)")
    axes[0].set_xlabel("members in group")
    axes[0].set_ylabel("groups")
    axes[1].set_title(f"{title} group length std (>=2 members)")
    axes[1].set_xlabel("population std of member lengths (pt)")
    axes[1].set_ylabel("groups")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Output: pipeline_report_viewer-compatible folder
# --------------------------------------------------------------------------

def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9+.-]+", "_", text).strip("_") or "layer"


class DebugReport:
    """One `outputs/<name>/<ts>__<pdf-stem>_p<N>/` folder: `add_vectors`/
    `add_boxes`/`add_layer` write one single-page PDF per layer
    (`<stage>__<label>.pdf`), `save_figure` a PNG, `finish` the
    `manifest.json` `scripts/pipeline_report_viewer.py` reads. Open with
    `python scripts/pipeline_report_viewer.py <out_dir>`."""

    def __init__(self, name: str, source_pdf: str, page_meta: PageMeta, *, line_width: float = 1.0):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.name = name
        self.source_pdf = str(Path(source_pdf).resolve())
        self.page_meta = page_meta
        self.line_width = line_width
        self.out_dir = output_dir(name, f"{stamp}__{Path(source_pdf).stem}_p{page_meta.index}")
        self.layers: list[dict] = []

    def add_layer(self, stage: str, label: str, color: RGB | str, pdf_bytes: bytes) -> None:
        file = f"{_slug(stage)}__{_slug(label)}.pdf"
        (self.out_dir / file).write_bytes(pdf_bytes)
        self.layers.append({
            "stage": stage, "layer": label, "file": file,
            "color": color if isinstance(color, str) else to_hex(color),
        })

    def add_vectors(self, stage: str, label: str, color: RGB | str, vectors: list[Vector], color_of) -> None:
        """One layer of `vectors` restroked in `color_of(v)`. Empty layers
        are skipped (nothing to toggle)."""
        if not vectors:
            return
        self.add_layer(stage, label, color, render_vectors_pdf(
            self.page_meta, vectors, color_of=color_of, width=self.line_width,
        ))

    def add_boxes(self, stage: str, label: str, color: RGB | str, boxes: list) -> None:
        if boxes:
            self.add_layer(stage, label, color, render_boxes_pdf(self.page_meta, boxes, width=0.75))

    def save_figure(self, fig, name: str) -> Path:
        path = self.out_dir / f"{_slug(name)}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        return path

    def finish(self) -> Path:
        manifest = {
            "source_pdf": self.source_pdf,
            "pages": [self.page_meta.index],
            "engine": "notebook",
            "variant": self.name,
            "layers": self.layers,
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return self.out_dir

    def viewer_command(self) -> str:
        return f'.venv/Scripts/python.exe scripts/pipeline_report_viewer.py "{self.out_dir}"'
