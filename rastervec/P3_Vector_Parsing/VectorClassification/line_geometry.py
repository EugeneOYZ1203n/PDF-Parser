"""Line geometry for VectorClassification's debug-only collinear/parallel
group layers -- straight-Vector line fits plus collinear/parallel grouping.
Own copy of the grouping half of `CollinearVectorClass/line_geometry.py`
(per the "P3 backends are self-contained" rule); none of it changes what
VectorClassification outputs.

The unit is always one whole `Vector` (one `get_drawings()` drawing):

- **straight Vectors** -- `straight_line(v)`: every item is "l" and every
  point lies within `tol` of one line. Only these take part in grouping.
- **grouping** -- `group_parallel` (single-linkage on the folded [0, 180)
  angle, wrap-aware) and `group_collinear` (`group_parallel`, then *anchored*
  grouping on each line's perpendicular offset, so dense hatching never
  chains into one "line"). Same infinite line, no along-line gap limit.

Angles are page space (y down), `atan2(dy, dx)` folded to [0, 180).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rastervec.commons.models import Vector

Point = tuple[float, float]


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


def golden_hues(k: int) -> list[float]:
    """`k` well-spread hues in [0, 1) (golden-ratio steps)."""
    return [(i * 0.618033988749895) % 1.0 for i in range(k)]
