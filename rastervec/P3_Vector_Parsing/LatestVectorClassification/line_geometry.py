"""Line geometry for LatestVectorClassification -- straight-Vector line
fits, collinear/parallel grouping, proper-crossing counts, the crossed-grid
test and the quad ink-ownership measure. Grouping/crossing primitives are
ported from the experimental probe notebooks'
`rastervec/notebooks/_vector_probe_helpers.py` (own copy, per the "P3
backends are self-contained" rule).

The unit is always one whole `Vector` (one `get_drawings()` drawing):

- **straight Vectors** -- `straight_line(v)`: every item is "l" and every
  point lies within `tol` of one line. Only these take part in collinear/
  parallel grouping.
- **grouping** -- `group_parallel` (single-linkage on the folded [0, 180)
  angle, wrap-aware) and `group_collinear` (`group_parallel`, then *anchored*
  grouping on each line's perpendicular offset, so dense hatching never
  chains into one "line"). Same infinite line, no along-line gap limit.
- **crossings** -- `line_crossing_counts(vectors)`: per Vector made only of
  "l" items, how many distinct foreign pieces properly (interior, X-shaped)
  cross it. Nothing is flattened: "l" segments and "re"/"qu" edges are
  tested as segments (one count per crossing piece), "c" curves exactly
  (`line_cubic_crossings` -- every proper crossing counts). Touching
  endpoints, corners, tangents and T-junctions don't count.
- **grid** -- `dominant_grid(flagged)`: the flagged Vectors whose segments
  are all parallel/perpendicular to one shared direction, when that
  direction holds a majority of the flagged "l" length.
- **ink ownership** -- the tiered quad test's pieces: `bbox_inside_quad`
  (all bbox corners inside), `quad_area`, `piece_overlap_fraction` (share of
  a Vector's pieces touching the quad, curves as chords) and
  `ink_fraction_in_quad` (the fraction of a Vector's path length inside a
  convex quad -- the expensive last resort).

Angles are page space (y down), `atan2(dy, dx)` folded to [0, 180) -- the
same convention as `Text.angle()`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rastervec.commons.models import Vector

Point = tuple[float, float]
Segment = tuple[float, float, float, float]  # x0, y0, x1, y1
Cubic = tuple[Point, Point, Point, Point]


@dataclass(frozen=True)
class LineFit:
    """A straight Vector's line: `angle` folded to [0, 180) degrees,
    `midpoint` of its extent, `length` = extent along the line."""

    angle: float
    midpoint: Point
    length: float


Straight = tuple[Vector, LineFit]


def straight_line(v: Vector, tol: float) -> LineFit | None:
    """`LineFit` if every item of `v` is "l" and every point lies within
    `tol` pt of the line through its two mutually farthest points, else
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


def split_straight(vectors: list[Vector], tol: float) -> tuple[list[Straight], list[Vector]]:
    """`(straight, other)` -- `(Vector, LineFit)` pairs for straight
    Vectors, plain Vectors for everything else."""
    straight, other = [], []
    for v in vectors:
        fit = straight_line(v, tol)
        if fit is None:
            other.append(v)
        else:
            straight.append((v, fit))
    return straight, other


def _anchored_1d(values: list[float], tol: float) -> list[list[int]]:
    """Anchored grouping on a line: indices sorted by value; a group starts
    at its smallest value and takes every following value within `tol` of
    that anchor, so a group spans at most `tol`. Not single-linkage: dense
    hatching puts parallel lines' offsets fractions of a point apart, and
    single-linkage chained them into one wide "collinear" group."""
    order = sorted(range(len(values)), key=values.__getitem__)
    groups: list[list[int]] = []
    for i in order:
        if groups and values[i] - values[groups[-1][0]] <= tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def chain_angles(angles: list[float], tol: float) -> list[list[int]]:
    """Single-linkage on the [0, 180) circle: angles sorted, split wherever
    the gap to the next exceeds `tol`; the first and last groups merge when
    the wrap-around gap is also within `tol`. Single-linkage on purpose:
    short CAD segments carry real angle jitter that a hard span cap splits."""
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


def mean_axis_angle(angles: list[float]) -> float:
    """Circular mean of axial angles (period 180), via doubled angles."""
    s = sum(math.sin(math.radians(2 * a)) for a in angles)
    c = sum(math.cos(math.radians(2 * a)) for a in angles)
    return (math.degrees(math.atan2(s, c)) / 2.0) % 180.0


def axial_distance(a: float, b: float) -> float:
    """Smallest separation of two axial angles (period 180)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def group_parallel(straight: list[Straight], angle_tol: float) -> list[list[Straight]]:
    """Single-linkage groups of straight Vectors by folded angle
    (`chain_angles` -- wrap-aware, so 179.8 deg and 0.1 deg join)."""
    angles = [fit.angle for _, fit in straight]
    return [[straight[i] for i in g] for g in chain_angles(angles, angle_tol)]


def group_collinear(
    straight: list[Straight], angle_tol: float, offset_tol: float,
) -> list[list[Straight]]:
    """Same-infinite-line groups: `group_parallel`, then within each angle
    group anchored grouping on each line's perpendicular offset (its
    midpoint projected onto the group's mean-angle normal) within
    `offset_tol` pt. No along-line gap limit."""
    out = []
    for group in group_parallel(straight, angle_tol):
        theta = math.radians(mean_axis_angle([fit.angle for _, fit in group]))
        nx, ny = -math.sin(theta), math.cos(theta)
        rhos = [fit.midpoint[0] * nx + fit.midpoint[1] * ny for _, fit in group]
        out.extend([group[i] for i in g] for g in _anchored_1d(rhos, offset_tol))
    return out


def group_length_std(group: list[Straight]) -> float:
    """Population std of member lengths, pt."""
    return float(np.std([fit.length for _, fit in group])) if group else 0.0


def golden_hues(k: int) -> list[float]:
    """`k` well-spread hues in [0, 1) (golden-ratio steps) -- debug colours."""
    return [(i * 0.618033988749895) % 1.0 for i in range(k)]


# --------------------------------------------------------------------------
# Item pieces -- no flattening: straight pieces stay segments, curves stay
# cubics.
# --------------------------------------------------------------------------

def _ring(points: list[Point]) -> list[Segment]:
    return [(*points[i], *points[(i + 1) % len(points)]) for i in range(len(points))]


def item_pieces(v: Vector) -> tuple[list[Segment], list[Cubic]]:
    """`(segments, cubics)` for every item of `v`: "l" as one segment,
    "re" (x0, y0, x1, y1) and "qu" (ul, ur, lr, ll) as their 4 edges, "c"
    as one cubic (p0, p1, p2, p3). Zero-length segments dropped."""
    segs: list[Segment] = []
    cubics: list[Cubic] = []
    for item in v.items:
        kind = item[0]
        if kind == "l":
            segs.append((*item[1], *item[2]))
        elif kind == "re":
            x0, y0, x1, y1 = item[1]
            segs.extend(_ring([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
        elif kind == "qu":
            segs.extend(_ring([tuple(p) for p in item[1]]))
        elif kind == "c":
            cubics.append((tuple(item[1]), tuple(item[2]), tuple(item[3]), tuple(item[4])))
    return [s for s in segs if s[0] != s[2] or s[1] != s[3]], cubics


def _pieces_bbox(segs: list[Segment], cubics: list[Cubic]) -> tuple[float, float, float, float]:
    """Bbox of every segment endpoint and cubic control point (a cubic lies
    inside its control hull); inverted (never overlapping) when empty."""
    xs = [c for s in segs for c in (s[0], s[2])] + [p[0] for cb in cubics for p in cb]
    ys = [c for s in segs for c in (s[1], s[3])] + [p[1] for cb in cubics for p in cb]
    if not xs:
        return (math.inf, math.inf, -math.inf, -math.inf)
    return (min(xs), min(ys), max(xs), max(ys))


def _bboxes_overlap(a, b) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


# --------------------------------------------------------------------------
# Proper crossings
# --------------------------------------------------------------------------

def _signed_dist(seg: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    dx = seg[..., 2] - seg[..., 0]
    dy = seg[..., 3] - seg[..., 1]
    length = np.hypot(dx, dy)
    length = np.where(length == 0, 1.0, length)
    return (dx * (py - seg[..., 1]) - dy * (px - seg[..., 0])) / length


def crossing_matrix(a: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    """Bool (n, m): `[i, j]` is True when segment `a[i]` properly crosses
    segment `b[j]` -- each segment's endpoints lie strictly on opposite
    sides of the other's line, each by more than `eps` pt. Touching
    endpoints, shared corners, T-junctions and collinear overlap never
    count."""
    out = np.zeros((len(a), len(b)), dtype=bool)
    if len(a) == 0 or len(b) == 0:
        return out
    A = a[:, None, :]
    B = b[None, :, :]
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


def _cubic_point(cubic: Cubic, t: float) -> Point:
    p0, p1, p2, p3 = cubic
    u = 1.0 - t
    a, b, c, d = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
    return (
        a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
        a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
    )


def line_cubic_crossings(seg: Segment, cubic: Cubic, eps: float) -> int:
    """How many times the cubic bezier `cubic` properly crosses the segment
    `seg` -- exactly, no flattening. The bezier is substituted into the
    segment's signed line distance `f(t)` (a cubic polynomial in `t`) and
    solved (`numpy.roots`); a real root `t` in (0, 1) counts when `f`
    changes sign there by more than `eps` pt on both sides (sampled midway
    to the neighbouring roots / the curve ends -- so a tangent touch or a
    double root never counts) and the crossing point lies strictly inside
    the segment (more than `eps` from either endpoint)."""
    x0, y0, x1, y1 = seg
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length == 0.0:
        return 0
    p0, p1, p2, p3 = (np.asarray(p, dtype=float) for p in cubic)
    # B(t) = a t^3 + b t^2 + c t + d
    a = -p0 + 3 * p1 - 3 * p2 + p3
    b = 3 * p0 - 6 * p1 + 3 * p2
    c = -3 * p0 + 3 * p1
    d = p0

    def lin(v: np.ndarray) -> float:
        return (dx * v[1] - dy * v[0]) / length

    offset = (dx * y0 - dy * x0) / length
    coeffs = [lin(a), lin(b), lin(c), lin(d) - offset]
    if all(abs(k) < 1e-12 for k in coeffs):
        return 0  # the curve lies on the line -- collinear overlap

    def f(t: float) -> float:
        px, py = _cubic_point(cubic, t)
        return (dx * (py - y0) - dy * (px - x0)) / length

    roots = sorted(
        float(r.real) for r in np.roots(coeffs)
        if abs(r.imag) < 1e-9 and 0.0 < r.real < 1.0
    )
    bounds = [0.0] + roots + [1.0]
    count = 0
    for k, t in enumerate(roots, start=1):
        left = f((bounds[k - 1] + t) / 2.0)
        right = f((t + bounds[k + 1]) / 2.0)
        if left * right >= 0 or abs(left) <= eps or abs(right) <= eps:
            continue
        px, py = _cubic_point(cubic, t)
        s = ((px - x0) * dx + (py - y0) * dy) / length  # pt along the segment
        if eps < s < length - eps:
            count += 1
    return count


def is_line_only(v: Vector) -> bool:
    """Every item is "l" (and there is at least one)."""
    return bool(v.items) and all(item[0] == "l" for item in v.items)


def line_crossing_counts(vectors: list[Vector], *, eps: float) -> list[int | None]:
    """Per Vector of `vectors` (same order): `None` unless it is made only of
    "l" items; otherwise how many foreign pieces properly cross it. Foreign
    = every other Vector of `vectors`, tested unflattened: each "l"
    segment / "re"-"qu" edge crossing any of this Vector's segments counts
    once (a foreign edge crossing two of its segments is still one), each
    "c" curve counts every proper crossing with every one of this Vector's
    segments (`line_cubic_crossings`). Its own items never count."""
    pieces = [item_pieces(v) for v in vectors]
    seg_arrays = [np.asarray(s, dtype=float).reshape(-1, 4) for s, _ in pieces]
    bboxes = [_pieces_bbox(s, c) for s, c in pieces]
    counts: list[int | None] = [None] * len(vectors)
    for i, v in enumerate(vectors):
        if not is_line_only(v) or not len(seg_arrays[i]):
            continue
        own = seg_arrays[i]
        n = 0
        for j in range(len(vectors)):
            if j == i or not _bboxes_overlap(bboxes[i], bboxes[j]):
                continue
            if len(seg_arrays[j]):
                n += int(crossing_matrix(own, seg_arrays[j], eps).any(axis=0).sum())
            for cubic in pieces[j][1]:
                n += sum(line_cubic_crossings(tuple(s), cubic, eps) for s in own)
        counts[i] = n
    return counts


# --------------------------------------------------------------------------
# Crossed grid
# --------------------------------------------------------------------------

def _mod90_distance(a: float, b: float) -> float:
    d = abs(a - b) % 90.0
    return min(d, 90.0 - d)


def _mean_mod90(angles: list[float], weights: list[float]) -> float:
    """Weighted circular mean with period 90 (quadrupled angles)."""
    s = sum(w * math.sin(math.radians(4 * a)) for a, w in zip(angles, weights))
    c = sum(w * math.cos(math.radians(4 * a)) for a, w in zip(angles, weights))
    return (math.degrees(math.atan2(s, c)) / 4.0) % 90.0


def grid_direction(v: Vector, tol: float) -> tuple[float | None, float]:
    """`(direction, length)` of an "l"-only Vector: `direction` (deg, folded
    mod 90) when every one of its segments lies within `tol` of it -- i.e.
    all mutually parallel or perpendicular -- else `None` (mixed); `length`
    is its total segment length."""
    angles, lengths = [], []
    for item in v.items:
        (xa, ya), (xb, yb) = item[1], item[2]
        seg_len = math.hypot(xb - xa, yb - ya)
        if seg_len == 0.0:
            continue
        angles.append(math.degrees(math.atan2(yb - ya, xb - xa)) % 90.0)
        lengths.append(seg_len)
    if not angles:
        return None, 0.0
    direction = _mean_mod90(angles, lengths)
    if any(_mod90_distance(a, direction) > tol for a in angles):
        return None, sum(lengths)
    return direction, sum(lengths)


def dominant_grid(flagged: list[Vector], *, tol: float, dominance: float) -> list[Vector]:
    """The flagged Vectors on the dominant grid: each flagged Vector with a
    `grid_direction` is a candidate anchor; the anchor whose members (every
    flagged Vector whose direction is within `tol` of it, mod 90) hold the
    largest total length wins, and its members are returned only when that
    length is more than `dominance` of *all* flagged Vectors' length (mixed
    ones included). A single flagged, non-mixed Vector is its own 100%
    grid."""
    info = [grid_direction(v, tol) for v in flagged]
    total = sum(length for _, length in info)
    if total <= 0.0:
        return []
    best: list[int] = []
    best_len = 0.0
    for direction, _ in info:
        if direction is None:
            continue
        members = [
            k for k, (d, _) in enumerate(info)
            if d is not None and _mod90_distance(d, direction) <= tol
        ]
        members_len = sum(info[k][1] for k in members)
        if members_len > best_len:
            best, best_len = members, members_len
    # A hair of slack so an exact split (float noise aside) is never a majority.
    if best_len / total <= dominance + 1e-9:
        return []
    return [flagged[k] for k in best]


# --------------------------------------------------------------------------
# Ink ownership
# --------------------------------------------------------------------------

def _clip_length_convex(seg: Segment, poly: np.ndarray) -> float:
    """Length of `seg` inside the convex polygon `poly` ((n, 2), either
    winding) -- Cyrus-Beck parametric clipping."""
    p0 = np.array(seg[:2], dtype=float)
    p1 = np.array(seg[2:], dtype=float)
    d = p1 - p0
    seg_len = float(np.hypot(*d))
    if seg_len == 0.0:
        return 0.0
    # Signed area -> winding; inward normal of edge (a -> b) is
    # (-ey, ex) for counter-clockwise (positive area, y-up maths).
    area = 0.0
    n = len(poly)
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        area += x1 * y2 - x2 * y1
    if area == 0.0:
        return 0.0
    sign = 1.0 if area > 0 else -1.0
    t_lo, t_hi = 0.0, 1.0
    for k in range(n):
        a = poly[k]
        e = poly[(k + 1) % n] - a
        normal = sign * np.array([-e[1], e[0]])
        num = float(np.dot(normal, p0 - a))  # >= 0 inside
        den = float(np.dot(normal, d))
        if den == 0.0:
            if num < 0.0:
                return 0.0
            continue
        t = -num / den
        if den > 0.0:
            t_lo = max(t_lo, t)
        else:
            t_hi = min(t_hi, t)
        if t_lo >= t_hi:
            return 0.0
    return (t_hi - t_lo) * seg_len


def _point_in_convex(pt: Point, poly: np.ndarray) -> bool:
    signs = []
    n = len(poly)
    for k in range(n):
        a, b = poly[k], poly[(k + 1) % n]
        signs.append((b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0]))
    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)


def ink_segments(v: Vector, curve_samples: int) -> list[Segment]:
    """`v`'s path as straight pieces *for length measurement only*: "l"
    segments, "re"/"qu" edges, and each "c" sampled at `curve_samples`
    points."""
    segs, cubics = item_pieces(v)
    for cubic in cubics:
        pts = [_cubic_point(cubic, i / curve_samples) for i in range(curve_samples + 1)]
        segs.extend((*pts[i], *pts[i + 1]) for i in range(curve_samples))
    return segs


def ink_fraction_in_quad(v: Vector, quad, *, curve_samples: int) -> float:
    """Fraction (0..1) of `v`'s ink -- its path length -- that lies inside
    the convex `quad` (4 page-space points, the rotated detect quad itself,
    not its envelope). A zero-length Vector counts as all-in or all-out by
    its bbox centre."""
    poly = np.asarray(quad, dtype=float).reshape(-1, 2)
    segs = ink_segments(v, curve_samples)
    total = sum(math.hypot(s[2] - s[0], s[3] - s[1]) for s in segs)
    if total == 0.0:
        x0, y0, x1, y1 = v.bbox
        return 1.0 if _point_in_convex(((x0 + x1) / 2.0, (y0 + y1) / 2.0), poly) else 0.0
    inside = sum(_clip_length_convex(s, poly) for s in segs)
    return min(1.0, inside / total)


def quad_area(quad) -> float:
    """Unsigned area of the polygon `quad` (shoelace)."""
    poly = np.asarray(quad, dtype=float).reshape(-1, 2)
    x, y = poly[:, 0], poly[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))) / 2.0


def bbox_inside_quad(bbox, quad) -> bool:
    """True when all 4 corners of `bbox` lie inside (or on) the convex
    `quad`. Plain floats, no numpy -- it runs once per (vector, quad) pair
    and decides most of them, where numpy's per-call overhead dominates."""
    pts = [(float(p[0]), float(p[1])) for p in quad]
    edges = [(ax, ay, bx - ax, by - ay) for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1])]
    x0, y0, x1, y1 = bbox
    pos = neg = False
    for px, py in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        for ax, ay, ex, ey in edges:
            cross = ex * (py - ay) - ey * (px - ax)
            pos |= cross > 0.0
            neg |= cross < 0.0
            if pos and neg:
                return False
    return True


def piece_overlap_fraction(v: Vector, quad) -> float:
    """Fraction (0..1) of `v`'s pieces that touch the convex `quad` -- "l"
    segments, "re"/"qu" edges, and each "c" as its chord p0->p3 (no curve
    sampling: a cheap gate, not a length measure). One vectorised
    Cyrus-Beck test over every piece. `nan` when `v` has no pieces; 0 for a
    degenerate (zero-area) quad."""
    segs, cubics = item_pieces(v)
    rows = segs + [(c[0][0], c[0][1], c[3][0], c[3][1]) for c in cubics]
    if not rows:
        return math.nan
    poly = np.asarray(quad, dtype=float).reshape(-1, 2)
    x, y = poly[:, 0], poly[:, 1]
    area = float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    if area == 0.0:
        return 0.0
    edges = np.roll(poly, -1, axis=0) - poly
    normals = math.copysign(1.0, area) * np.stack([-edges[:, 1], edges[:, 0]], axis=1)
    arr = np.asarray(rows, dtype=float)
    p0 = arr[:, :2]
    d = arr[:, 2:] - p0
    num = np.einsum("sek,ek->se", p0[:, None, :] - poly[None, :, :], normals)  # >= 0 inside
    den = d @ normals.T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -num / den
    t_lo = np.maximum(np.where(den > 0.0, t, -np.inf).max(axis=1), 0.0)
    t_hi = np.minimum(np.where(den < 0.0, t, np.inf).min(axis=1), 1.0)
    parallel_out = ((den == 0.0) & (num < 0.0)).any(axis=1)
    overlap = (t_lo <= t_hi) & ~parallel_out
    return float(overlap.mean())
