"""Pure numpy geometry for inference (`inference.py`, `refine.py`,
`merge.py`, `adapter.py`).

A `Prim` is this backend's output primitive: the paper's line segment
`(p1, p2)` or quadratic Bezier `(p1, p2 control, p3)`, plus a width.

Coordinates are pixels of whatever frame the caller works in.
"""

from __future__ import annotations


import numpy as np
from raster_parser.models.deep_tech_vec import Prim
from raster_parser.commons.raster_text.strokes import piece_is_flat, rect_predicate, tile_starts


# ---------------------------------------------------------------------------
# Region tests (shared by primitive clipping)
# ---------------------------------------------------------------------------
def _inside_runs(evaluate, inside, samples: int) -> list[tuple[float, float]]:
    """Parameter intervals `[ta, tb]` of a curve `evaluate(t)` inside the
    region (sampled, then bisected at each boundary)."""
    t = np.linspace(0.0, 1.0, samples + 1)
    mask = inside(evaluate(t))
    runs = []
    i = 0
    while i < len(t):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(t) and mask[j + 1]:
            j += 1
        ta = 0.0 if i == 0 else _refine(evaluate, inside, t[i], t[i - 1])
        tb = 1.0 if j == len(t) - 1 else _refine(evaluate, inside, t[j], t[j + 1])
        if tb > ta:
            runs.append((ta, tb))
        i = j + 1
    return runs


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def quad_eval(q: np.ndarray, t) -> np.ndarray:
    t = np.asarray(t, float)[:, None]
    mt = 1.0 - t
    return mt * mt * q[0] + 2.0 * mt * t * q[1] + t * t * q[2]


def quad_to_cubic(q: np.ndarray) -> np.ndarray:
    """Exact degree elevation of a quadratic to a cubic."""
    q = np.asarray(q, float)
    return np.stack(
        [q[0], q[0] + 2.0 / 3.0 * (q[1] - q[0]), q[2] + 2.0 / 3.0 * (q[1] - q[2]), q[2]]
    )


def split_quad(q: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """The sub-quadratic over `[t0, t1]` (blossoming)."""
    q = np.asarray(q, float)

    def blossom(a: float, b: float) -> np.ndarray:
        return (1 - a) * (1 - b) * q[0] + ((1 - a) * b + a * (1 - b)) * q[1] + a * b * q[2]

    return np.stack([blossom(t0, t0), blossom(t0, t1), blossom(t1, t1)])


def prim_eval(p: Prim, t) -> np.ndarray:
    if p.is_line:
        t = np.asarray(t, float)[:, None]
        return p.pts[0] + (p.pts[1] - p.pts[0]) * t
    return quad_eval(p.pts, t)


def prim_points(p: Prim, n: int = 8) -> np.ndarray:
    """`(n + 1, 2)` polyline (2 points for a line)."""
    if p.is_line:
        return np.array(p.pts, float)
    return quad_eval(p.pts, np.linspace(0.0, 1.0, n + 1))


def polyline_length(pts: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0


def quad_is_flat(q: np.ndarray, tol: float) -> bool:
    return piece_is_flat(quad_to_cubic(q), tol)


def sub_prim(p: Prim, t0: float, t1: float) -> Prim:
    if p.is_line:
        a, b = prim_eval(p, [t0, t1])
        pts = np.stack([a, b])
    else:
        pts = split_quad(p.pts, t0, t1)
    return Prim(pts, p.width, p.conf, dict(p.meta))


def clip_prim(p: Prim, rect, min_len: float = 1.0, samples: int = 32) -> list[Prim]:
    """The parts of a primitive inside `rect = (x0, y0, x1, y1)` (rect may
    have infinite sides). A line is clipped exactly (Liang-Barsky)."""
    if p.is_line:
        seg = _clip_segment(p.pts[0], p.pts[1], rect)
        if seg is None:
            return []
        out = Prim(np.stack(seg), p.width, p.conf, dict(p.meta))
        return [out] if out.length() >= min_len else []
    inside = rect_predicate(rect)
    runs = _inside_runs(lambda t: quad_eval(p.pts, t), inside, samples)
    parts = [sub_prim(p, ta, tb) for ta, tb in runs]
    return [q for q in parts if q.length() >= min_len]


def _clip_segment(a, b, rect):
    x0, y0, x1, y1 = rect
    a = np.asarray(a, float)
    d = np.asarray(b, float) - a
    t0, t1 = 0.0, 1.0
    for pk, qk in ((-d[0], a[0] - x0), (d[0], x1 - a[0]), (-d[1], a[1] - y0), (d[1], y1 - a[1])):
        if pk == 0:
            if qk < 0:
                return None
            continue
        r = qk / pk
        if pk < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return None
    return a + d * t0, a + d * t1


# ---------------------------------------------------------------------------
# Patch grid (tile layout: `commons.raster_text.strokes`)
# ---------------------------------------------------------------------------
def tile_grid(h: int, w: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    """Top-left `(x0, y0)` of every tile covering an `h`x`w` image."""
    return [(x, y) for y in tile_starts(h, tile, overlap) for x in tile_starts(w, tile, overlap)]


def core_rect(
    x0: int, y0: int, size: int, h: int, w: int, overlap: int
) -> tuple[float, float, float, float]:
    """The part of a tile it "owns" (`core_spans` on both axes), so
    neighbouring cores tile the image without gaps or double ownership."""
    ax, bx = core_spans(w, size, overlap)[x0]
    ay, by = core_spans(h, size, overlap)[y0]
    return (ax, ay, bx, by)


def _refine(evaluate, inside, t_in: float, t_out: float, iters: int = 12) -> float:
    """Bisect for the boundary between an inside and an outside parameter."""
    for _ in range(iters):
        mid = 0.5 * (t_in + t_out)
        if inside(evaluate([mid]))[0]:
            t_in = mid
        else:
            t_out = mid
    return t_in


def core_spans(total: int, tile: int, overlap: int) -> dict[int, tuple[float, float]]:
    """Tile start -> the range `[a, b)` along one axis that tile owns.
    Neighbouring tiles split their *actual* overlap at its middle (the last
    tile is shifted back to the edge, so its overlap with the previous one
    can be much more than `overlap`); image borders stay open (+-inf), so
    geometry slightly past the edge is kept."""
    starts = tile_starts(total, tile, overlap)
    out = {}
    for i, s in enumerate(starts):
        a = float("-inf") if i == 0 else (s + starts[i - 1] + tile) / 2.0
        b = float("inf") if i == len(starts) - 1 else (starts[i + 1] + s + tile) / 2.0
        out[s] = (a, b)
    return out
