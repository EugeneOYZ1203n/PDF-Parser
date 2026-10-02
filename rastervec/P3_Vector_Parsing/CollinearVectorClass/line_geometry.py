"""Line geometry for CollinearVectorClass -- straight-Vector line fits,
collinear/parallel grouping and proper-crossing counts. Ported from the
experimental probe notebooks' `rastervec/notebooks/_vector_probe_helpers.py`
(own copy, per the "P3 backends are self-contained" rule).

The unit is always one whole `Vector` (one `get_drawings()` drawing):

- **straight Vectors** -- `straight_line(v)`: every item is "l" and every
  point lies within `tol` of one line. Only these take part in collinear/
  parallel grouping.
- **grouping** -- `group_parallel` (single-linkage on the folded [0, 180)
  angle, wrap-aware) and `group_collinear` (`group_parallel`, then *anchored*
  grouping on each line's perpendicular offset, so dense hatching never
  chains into one "line"). Same infinite line, no along-line gap limit.
- **crossings** -- `crossing_segment_counts(vectors)`: per Vector, how many
  distinct segments of *other* Vectors properly (interior, X-shaped) cross
  one of its own. Items are flattened to segments ("re"/"qu" as 4 edges,
  "c" as a polyline); touching endpoints, corners and T-junctions don't
  count.

Angles are page space (y down), `atan2(dy, dx)` folded to [0, 180) -- the
same convention as `Text.angle()`, so a group angle is directly a candidate
text direction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rastervec.commons.models import Vector

Point = tuple[float, float]
Segment = tuple[float, float, float, float]  # x0, y0, x1, y1


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


def group_angle(group: list[Straight]) -> float:
    return mean_axis_angle([fit.angle for _, fit in group])


def group_total_length(group: list[Straight]) -> float:
    return sum(fit.length for _, fit in group)


def group_length_std(group: list[Straight]) -> float:
    """Population std of member lengths, pt."""
    return float(np.std([fit.length for _, fit in group])) if group else 0.0


def dedupe_angles(angles: list[float], tol: float) -> list[float]:
    """Merge axial angles that chain within `tol` (`chain_angles`) into one
    circular mean each, sorted."""
    return sorted(mean_axis_angle([angles[i] for i in g]) for g in chain_angles(angles, tol))


# --------------------------------------------------------------------------
# Proper crossings
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
    return [(*points[i], *points[(i + 1) % len(points)]) for i in range(len(points))]


def vector_segments(v: Vector, curve_steps: int) -> list[Segment]:
    """Every item of `v` as straight segments: "l" as-is, "re" (x0, y0, x1,
    y1) and "qu" (ul, ur, lr, ll) as their 4 edges, "c" flattened to
    `curve_steps` segments. Zero-length segments dropped."""
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


def crossing_segment_counts(vectors: list[Vector], *, eps: float, curve_steps: int) -> list[int]:
    """Per Vector of `vectors` (same order), the number of distinct
    segments belonging to *other* Vectors that properly cross at least one
    of its own segments. One foreign segment crossing two of this Vector's
    segments counts once; its own segments never count."""
    segs = [np.asarray(vector_segments(v, curve_steps), dtype=float).reshape(-1, 4) for v in vectors]
    bboxes = []
    for s in segs:
        if len(s):
            xs, ys = s[:, [0, 2]], s[:, [1, 3]]
            bboxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
        else:
            bboxes.append((math.inf, math.inf, -math.inf, -math.inf))
    counts = [0] * len(vectors)
    for i, j in _overlapping_pairs(bboxes):
        m = crossing_matrix(segs[i], segs[j], eps)
        counts[i] += int(m.any(axis=0).sum())
        counts[j] += int(m.any(axis=1).sum())
    return counts
