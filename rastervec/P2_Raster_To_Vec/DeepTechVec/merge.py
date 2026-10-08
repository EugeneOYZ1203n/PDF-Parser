"""Merging the refined primitives of all patches into one drawing (paper
Sec. 3.4 + Appendix C, Fig. 13 / 14).

Lines (Fig. 13):
  (a-c) link two lines when they are close and collinear enough -- but not
        merely parallel (a parallel line offset beside another fails the
        distance-to-line test) -- and join the links into connected
        components;
  (d)   replace each component with one least-squares line fit to its
        endpoints (PCA direction, extended over the endpoints' projections;
        width = length-weighted mean);
  (e)   snap intersecting lines by cutting "dangling" ends past an
        intersection that are shorter than `DANGLE_FRAC` of the line.

Quadratic curves (Fig. 14): repeatedly, for a pair of close curves with
close widths whose second curve's endpoints and midpoint lie close to the
first curve (extended), fit one quadratic R(u) by least squares to
P(0), P(t_b), P(1), Q(0), Q(s_b), Q(1) at the parameters of Eq. 15,

    R(0), R(t_b u_q1 / t_q1), R(u_q1 / t_q1), R(u_q1), R(1 - (1 - s_b)(1 - u_q1)), R(1)

with u_q1 found by brute force; the pair is replaced when the fit is close
enough. Until no pair changes.

Ours: the thresholds; t_b = s_b = 1/2 (this backend's curve "midpoint" is
Q(1/2), see refine.py); the pair's orientation is the one that puts Q's
start nearest P's end.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from . import geometry as geo
from .config import (
    DANGLE_FRAC, MERGE_CURVE_DIST_PX, MERGE_CURVE_FIT_PX, MERGE_CURVE_U_STEPS, MERGE_LINE_ANGLE_DEG,
    MERGE_LINE_DIST_PX, MERGE_LINE_GAP_PX, MERGE_WIDTH_RATIO,
)


def merge_prims(prims: list[geo.Prim]) -> list[geo.Prim]:
    lines = [p for p in prims if p.is_line]
    curves = [p for p in prims if not p.is_line]
    return merge_lines(lines) + merge_curves(curves)


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------
def _widths_close(a: float, b: float) -> bool:
    lo, hi = sorted((max(a, 1e-6), max(b, 1e-6)))
    return hi / lo <= MERGE_WIDTH_RATIO


def lines_linked(a: geo.Prim, b: geo.Prim) -> bool:
    """Close + collinear (Fig. 13 a): directions within the angle limit,
    both ends of the shorter within `MERGE_LINE_DIST_PX` of the longer's
    infinite line, overlapping or gapped by at most `MERGE_LINE_GAP_PX`."""
    if a.length() < b.length():
        a, b = b, a
    la = a.length()
    if la < 1e-9 or not _widths_close(a.width, b.width):
        return False
    u = (a.end - a.start) / la
    lb = b.length()
    if lb > 1e-9:
        ub = (b.end - b.start) / lb
        if abs(float(u @ ub)) < math.cos(math.radians(MERGE_LINE_ANGLE_DEG)):
            return False
    n = np.array([-u[1], u[0]])
    rel = np.asarray(b.pts, float) - a.start
    if np.abs(rel @ n).max() > MERGE_LINE_DIST_PX:
        return False
    t = rel @ u
    return not (t.min() > la + MERGE_LINE_GAP_PX or t.max() < -MERGE_LINE_GAP_PX)


def fit_line(group: list[geo.Prim]) -> geo.Prim:
    """Least-squares line through every endpoint of the group (Fig. 13 d)."""
    pts = np.concatenate([np.asarray(p.pts, float) for p in group])
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    u = vt[0]
    t = (pts - c) @ u
    lens = np.array([max(p.length(), 1e-6) for p in group])
    width = float(np.sum(lens * [p.width for p in group]) / lens.sum())
    conf = float(max(p.conf for p in group))
    return geo.Prim(np.stack([c + t.min() * u, c + t.max() * u]), width, conf)


def merge_lines(lines: list[geo.Prim]) -> list[geo.Prim]:
    if len(lines) < 2:
        return list(lines)
    mids = np.array([0.5 * (p.start + p.end) for p in lines])
    lens = np.array([p.length() for p in lines])
    tree = cKDTree(mids)
    parent = list(range(len(lines)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    radius = 0.5 * lens.max() + 0.5 * lens.max() + MERGE_LINE_GAP_PX
    for i, j in tree.query_pairs(radius):
        if np.hypot(*(mids[i] - mids[j])) > 0.5 * (lens[i] + lens[j]) + MERGE_LINE_GAP_PX + MERGE_LINE_DIST_PX:
            continue
        if find(i) != find(j) and lines_linked(lines[i], lines[j]):
            parent[find(i)] = find(j)
    groups: dict[int, list[geo.Prim]] = {}
    for i, p in enumerate(lines):
        groups.setdefault(find(i), []).append(p)
    merged = [g[0] if len(g) == 1 else fit_line(g) for g in groups.values()]
    return snap_dangling(merged)


def _segment_intersection(a0, a1, b0, b1):
    """Parameters (s, t) of the crossing of segments a and b, or None."""
    da, db = a1 - a0, b1 - b0
    den = da[0] * db[1] - da[1] * db[0]
    if abs(den) < 1e-9:
        return None
    r = b0 - a0
    s = (r[0] * db[1] - r[1] * db[0]) / den
    t = (r[0] * da[1] - r[1] * da[0]) / den
    if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
        return s, t
    return None


def snap_dangling(lines: list[geo.Prim]) -> list[geo.Prim]:
    """Fig. 13 e: where two lines cross, an end that sticks out past the
    crossing by less than `DANGLE_FRAC` of its line is cut back to it."""
    if len(lines) < 2:
        return lines
    pts = [np.array(p.pts, float) for p in lines]
    mids = np.array([0.5 * (p[0] + p[1]) for p in pts])
    lens = np.array([np.hypot(*(p[1] - p[0])) for p in pts])
    tree = cKDTree(mids)
    for i, j in tree.query_pairs(float(lens.max())):
        hit = _segment_intersection(pts[i][0], pts[i][1], pts[j][0], pts[j][1])
        if hit is None:
            continue
        for k, par in ((i, hit[0]), (j, hit[1])):
            frac = DANGLE_FRAC
            x = pts[k][0] + par * (pts[k][1] - pts[k][0])
            if 0.0 < par < frac:
                pts[k][0] = x
            elif 1.0 - frac < par < 1.0:
                pts[k][1] = x
    return [geo.Prim(p, q.width, q.conf, q.meta) for p, q in zip(pts, lines)]


# ---------------------------------------------------------------------------
# Quadratic curves
# ---------------------------------------------------------------------------
_EXT_T = np.linspace(-1.0, 2.0, 121)


def _dist_to_curve(q: np.ndarray, pts: np.ndarray, t=_EXT_T) -> tuple[np.ndarray, np.ndarray]:
    """Distance of `pts` to the (extended) parabola of quadratic `q`, and the
    parameter of the nearest sample."""
    samples = geo.quad_eval(q, t)
    d = np.linalg.norm(pts[:, None, :] - samples[None], axis=2)
    k = d.argmin(axis=1)
    return d[np.arange(len(pts)), k], t[k]


def _bernstein(u: np.ndarray) -> np.ndarray:
    return np.stack([(1 - u) ** 2, 2 * u * (1 - u), u * u], axis=1)


def fit_pair(p: np.ndarray, q: np.ndarray, t_b: float = 0.5, s_b: float = 0.5,
             steps: int = MERGE_CURVE_U_STEPS) -> "tuple[np.ndarray, float] | None":
    """Eq. 14-15: the least-squares quadratic R replacing the pair (P then Q),
    with u_q1 brute-forced; returns (R, max point error) or None."""
    targets = np.stack([geo.quad_eval(p, [0.0])[0], geo.quad_eval(p, [t_b])[0], p[2],
                        q[0], geo.quad_eval(q, [s_b])[0], q[2]])
    _, t_q1 = _dist_to_curve(p, q[:1], np.linspace(0.05, 3.0, 300))
    t_q1 = float(t_q1[0])
    best = None
    for u_q1 in np.linspace(0.02, 0.98, steps):
        u = np.array([0.0, t_b * u_q1 / t_q1, u_q1 / t_q1, u_q1, 1 - (1 - s_b) * (1 - u_q1), 1.0])
        if u[2] > 1.0 + 1e-9:
            continue
        a = _bernstein(np.clip(u, 0.0, 1.0))
        ctrl, *_ = np.linalg.lstsq(a, targets, rcond=None)
        err = float(np.max(np.linalg.norm(a @ ctrl - targets, axis=1)))
        if best is None or err < best[1]:
            best = (ctrl, err)
    return best


def _orient_pair(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The orientation of both curves that puts Q's start nearest P's end."""
    best = None
    for pp in (p, p[::-1]):
        for qq in (q, q[::-1]):
            d = float(np.hypot(*(pp[2] - qq[0])))
            if best is None or d < best[0]:
                best = (d, pp, qq)
    return best[1].copy(), best[2].copy()


def curves_close(p: geo.Prim, q: geo.Prim) -> bool:
    """Fig. 14 b-c: widths close; Q's endpoints and midpoint near P's
    (extended) curve; and the two actually touch or overlap."""
    if not _widths_close(p.width, q.width):
        return False
    probe = np.stack([q.pts[0], geo.quad_eval(q.pts, [0.5])[0], q.pts[2]])
    d, _ = _dist_to_curve(p.pts, probe)
    if d.max() > MERGE_CURVE_DIST_PX:
        return False
    ends_p = np.stack([p.pts[0], p.pts[2]])
    gap = np.min(np.linalg.norm(ends_p[:, None] - np.stack([q.pts[0], q.pts[2]])[None], axis=2))
    d_on, _ = _dist_to_curve(p.pts, np.stack([q.pts[0], q.pts[2]]), np.linspace(0.0, 1.0, 41))
    return gap <= MERGE_CURVE_DIST_PX or d_on.min() <= MERGE_CURVE_DIST_PX


def merge_curves(curves: list[geo.Prim], max_rounds: int = 50) -> list[geo.Prim]:
    cur = [geo.Prim(np.array(c.pts, float), c.width, c.conf, c.meta) for c in curves]
    for _ in range(max_rounds):
        if len(cur) < 2:
            break
        ends = np.array([c.pts[k] for c in cur for k in (0, 2)])
        tree = cKDTree(ends)
        lens = np.array([c.length() for c in cur])
        radius = float(lens.max()) + MERGE_CURVE_DIST_PX
        used: set[int] = set()
        new: list[geo.Prim] = []
        for a, b in sorted(tree.query_pairs(radius)):
            i, j = a // 2, b // 2
            if i == j or i in used or j in used:
                continue
            if not curves_close(cur[i], cur[j]):
                continue
            p, q = _orient_pair(cur[i].pts, cur[j].pts)
            fit = fit_pair(p, q)
            if fit is None or fit[1] > MERGE_CURVE_FIT_PX:
                continue
            li, lj = max(lens[i], 1e-6), max(lens[j], 1e-6)
            width = float((li * cur[i].width + lj * cur[j].width) / (li + lj))
            new.append(geo.Prim(fit[0], width, max(cur[i].conf, cur[j].conf)))
            used.update((i, j))
        if not used:
            break
        cur = [c for k, c in enumerate(cur) if k not in used] + new
    return cur
