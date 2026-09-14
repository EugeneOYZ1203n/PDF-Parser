"""Polygonal approximation + arc/circle detection (Dosch 2000 Sec 2.3-2.4)."""
from __future__ import annotations

import numpy as np

from .geom import polyline_length, taubin_circle_fit
from .types_ import Arc, Point


# ---------------------------------------------------------------- polygonal approx


def _max_dev(pts: np.ndarray, i: int, j: int) -> tuple[float, int]:
    """Max perpendicular distance of pts[i..j] from the chord pts[i]-pts[j]."""
    a, b = pts[i], pts[j]
    ab = b - a
    n = np.hypot(*ab)
    if n < 1e-9:
        d = np.hypot(*(pts[i:j + 1] - a).T)
    else:
        d = np.abs(np.cross(np.tile(ab, (j - i + 1, 1)), pts[i:j + 1] - a)) / n
    k = int(np.argmax(d))
    return float(d[k]), i + k


def approximate_rdp(chain: list[Point], eps: float) -> list[Point]:
    """Douglas-Peucker (the Wall & Danielsson single-threshold role)."""
    pts = np.asarray(chain, float)
    if len(pts) < 3:
        return [tuple(p) for p in pts]
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        d, k = _max_dev(pts, i, j)
        if d > eps:
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return [tuple(p) for p in pts[keep]]


def approximate_rosin_west(chain: list[Point], min_significance: float = 0.02) -> list[Point]:
    """Parameter-light recursive split (Rosin & West 1989).

    Split while max-deviation/segment-length (the 'significance' ratio) exceeds
    min_significance; the paper's variant is fully parameter-free (split to 0
    deviation, then merge back by significance) -- this keeps one small ratio
    threshold, which is stabler in practice and still needs no length units.
    """
    pts = np.asarray(chain, float)
    n = len(pts)
    if n < 3:
        return [tuple(p) for p in pts]
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        d, k = _max_dev(pts, i, j)
        seg_len = float(np.hypot(*(pts[j] - pts[i])))
        signif = d / seg_len if seg_len > 1e-6 else d
        if signif > min_significance and d > 0.75:
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return [tuple(p) for p in pts[keep]]


# ---------------------------------------------------------------- arc detection


def _turn_angles(poly: np.ndarray) -> np.ndarray:
    v = np.diff(poly, axis=0)
    ang = np.arctan2(v[:, 1], v[:, 0])
    dt = np.diff(ang)
    return (dt + np.pi) % (2 * np.pi) - np.pi   # wrapped to (-pi, pi]


def detect_arcs(
    polyline: list[Point],
    pixel_chain: list[Point],
    *,
    angle_tol_deg: float = 18.0,
    fit_tol_px: float = 2.2,
    min_points: int = 4,
    min_radius: float = 5.0,
) -> tuple[list[Arc], list[tuple[Point, Point]]]:
    """Split a polyline into arc hypotheses (runs of >=`min_points` vertices with
    near-constant, same-sign turn) validated by a Taubin circle fit on the
    underlying pixel chain (Dosch-Masini-Tombre 2000). Returns (arcs, leftover
    straight segments)."""
    poly = np.asarray(polyline, float)
    px = np.asarray(pixel_chain, float)
    if len(poly) < min_points:
        return [], _as_segments(poly)

    turns = _turn_angles(poly)                      # len = len(poly) - 2
    tol = np.radians(angle_tol_deg)
    arcs: list[Arc] = []
    consumed = np.zeros(len(poly), bool)

    i = 0
    while i < len(turns):
        if abs(turns[i]) < np.radians(3.0):        # essentially straight here
            i += 1
            continue
        sign = np.sign(turns[i])
        j = i
        while (
            j < len(turns)
            and np.sign(turns[j]) == sign
            and abs(abs(turns[j]) - abs(turns[i])) < tol
        ):
            j += 1
        # vertices poly[i .. j+1] form the hypothesis
        v0, v1 = i, j + 1
        if v1 - v0 + 1 >= min_points:
            arc = _validate_arc(poly, px, v0, v1, fit_tol_px, min_radius)
            if arc is not None:
                arcs.append(arc)
                consumed[v0:v1 + 1] = True
        i = j + 1

    # closed full-circle check on the whole chain
    if len(arcs) == 0 and len(poly) >= 8:
        first, last = poly[0], poly[-1]
        if np.hypot(*(first - last)) < 0.15 * polyline_length(poly):
            cx, cy, r, rms = taubin_circle_fit(px)
            if r > min_radius and rms < fit_tol_px * 1.4:
                arcs.append(_make_arc(px, cx, cy, r, closed=True))
                consumed[:] = True

    leftovers = _as_segments(poly[~consumed]) if consumed.any() else _as_segments(poly)
    return arcs, leftovers


def _validate_arc(poly, px, v0, v1, fit_tol_px, min_radius) -> Arc | None:
    p0, p1 = poly[v0], poly[v1]
    # map polyline vertex range to a pixel-chain slice by nearest pixel
    k0 = int(np.argmin(np.hypot(*(px - p0).T)))
    k1 = int(np.argmin(np.hypot(*(px - p1).T)))
    lo, hi = sorted((k0, k1))
    sub = px[lo:hi + 1]
    if len(sub) < 5:
        return None
    cx, cy, r, rms = taubin_circle_fit(sub)
    if r < min_radius or rms > fit_tol_px:
        return None
    # reject near-straight (huge radius vs chord)
    chord = float(np.hypot(*(sub[0] - sub[-1])))
    if r > 12 * max(chord, 1.0):
        return None
    return _make_arc(sub, cx, cy, r, closed=False)


def _make_arc(sub, cx, cy, r, closed: bool) -> Arc:
    ang = np.degrees(np.arctan2(sub[:, 1] - cy, sub[:, 0] - cx))
    a0, a1 = float(ang[0]), float(ang[-1])
    if closed:
        a0, a1 = 0.0, 360.0
        t = np.linspace(0, 2 * np.pi, 64)
    else:
        t = np.radians(np.linspace(a0, a1, max(8, len(sub) // 2)))
    poly = [(float(cx + r * np.cos(tt)), float(cy + r * np.sin(tt))) for tt in t]
    return Arc(center=(float(cx), float(cy)), radius=float(r), a0=a0, a1=a1,
              polyline=poly, closed=closed)


def _as_segments(poly: np.ndarray) -> list[tuple[Point, Point]]:
    poly = np.asarray(poly, float)
    return [((float(poly[i][0]), float(poly[i][1])),
             (float(poly[i + 1][0]), float(poly[i + 1][1])))
            for i in range(len(poly) - 1)]
