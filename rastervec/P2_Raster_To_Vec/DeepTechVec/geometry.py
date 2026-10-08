"""Pure numpy geometry shared by inference (`inference.py`, `merge.py`,
`adapter.py`) and training (`train_data.py`).

Two shapes of geometry live here:

* a *stroke* -- the shared dataset's ground truth: a chain of cubic Bezier
  pieces `(K, 4, 2)` (helpers copied from DeepVectoriser/geometry.py);
* a `Prim` -- this backend's output primitive, the paper's line segment
  `(p1, p2)` or quadratic Bezier `(p1, p2 control, p3)`, plus a width.

Coordinates are pixels of whatever frame the caller works in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

Stroke = np.ndarray  # (K, 4, 2)


# ---------------------------------------------------------------------------
# Cubic strokes (copied from DeepVectoriser/geometry.py)
# ---------------------------------------------------------------------------
def line_to_cubic(a, b) -> np.ndarray:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return np.stack([a, a + (b - a) / 3.0, a + (b - a) * 2.0 / 3.0, b])


def bezier_eval(piece: np.ndarray, t) -> np.ndarray:
    """Points `(len(t), 2)` of one cubic at parameters `t`."""
    t = np.asarray(t, float)[:, None]
    mt = 1.0 - t
    p0, c1, c2, p3 = piece
    return mt ** 3 * p0 + 3 * mt ** 2 * t * c1 + 3 * mt * t ** 2 * c2 + t ** 3 * p3


def sample_stroke(stroke: Stroke, per_piece: int = 16) -> np.ndarray:
    """Polyline `(N, 2)` through every piece (shared joints not repeated)."""
    if len(stroke) == 0:
        return np.zeros((0, 2))
    t = np.linspace(0.0, 1.0, per_piece + 1)
    pts = [bezier_eval(stroke[0], t)]
    for piece in stroke[1:]:
        pts.append(bezier_eval(piece, t)[1:])
    return np.concatenate(pts, axis=0)


def stroke_length(stroke: Stroke) -> float:
    pts = sample_stroke(stroke, 8)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0


def split_cubic_at(p: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    a = p[0] + (p[1] - p[0]) * t
    b = p[1] + (p[2] - p[1]) * t
    c = p[2] + (p[3] - p[2]) * t
    ab = a + (b - a) * t
    bc = b + (c - b) * t
    m = ab + (bc - ab) * t
    return np.stack([p[0], a, ab, m]), np.stack([m, bc, c, p[3]])


def split_bezier(piece: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """The sub-cubic of `piece` over `[t0, t1]` (de Casteljau)."""
    piece = np.asarray(piece, float)
    right = piece if t1 >= 1.0 else split_cubic_at(piece, t1)[0]
    if t0 <= 0.0:
        return right
    return split_cubic_at(right, t0 / t1 if t1 > 0 else 0.0)[1]


def piece_is_flat(piece: np.ndarray, tol: float) -> bool:
    """Both controls within `tol` of the chord (or a degenerate chord)."""
    p0, c1, c2, p3 = np.asarray(piece, float)
    d = p3 - p0
    n = float(np.hypot(*d))
    if n < 1e-9:
        return max(np.hypot(*(c1 - p0)), np.hypot(*(c2 - p0))) <= tol
    for c in (c1, c2):
        v = c - p0
        if abs(d[0] * v[1] - d[1] * v[0]) / n > tol:
            return False
        s = float(np.dot(v, d)) / (n * n)
        if s < -tol / n or s > 1 + tol / n:
            return False
    return True


def rect_predicate(rect):
    """`inside(points (N, 2)) -> bool (N,)` for `rect = (x0, y0, x1, y1)`."""
    x0, y0, x1, y1 = rect

    def inside(pts: np.ndarray) -> np.ndarray:
        return (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
    return inside


def _refine(evaluate, inside, t_in: float, t_out: float, iters: int = 12) -> float:
    """Bisect for the boundary between an inside and an outside parameter."""
    for _ in range(iters):
        mid = 0.5 * (t_in + t_out)
        if inside(evaluate([mid]))[0]:
            t_in = mid
        else:
            t_out = mid
    return t_in


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


def clip_stroke(stroke: Stroke, rect, samples: int = 32, min_len: float = 1.5) -> list[Stroke]:
    """The parts of `stroke` inside axis-aligned `rect`, each a stroke of its
    own (leaving and re-entering splits it); parts shorter than `min_len`
    are dropped."""
    inside = rect_predicate(rect)
    out: list[Stroke] = []
    cur: list[np.ndarray] = []

    def flush():
        if cur:
            s = np.stack(cur)
            if stroke_length(s) >= min_len:
                out.append(s)
            cur.clear()

    for piece in stroke:
        piece = np.asarray(piece, float)
        runs = _inside_runs(lambda t, p=piece: bezier_eval(p, t), inside, samples)
        if runs == [(0.0, 1.0)]:
            cur.append(piece)
            continue
        if not runs:
            flush()
            continue
        for ta, tb in runs:
            if ta > 0.0:
                flush()
            cur.append(split_bezier(piece, ta, tb))
            if tb < 1.0:
                flush()
    flush()
    return out


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
@dataclass
class Prim:
    """One primitive: `pts` is `(2, 2)` for a line segment or `(3, 2)` for a
    quadratic Bezier `(p1, control, p3)`; `width` in px."""
    pts: np.ndarray
    width: float
    conf: float = 1.0
    meta: dict = field(default_factory=dict)

    @property
    def is_line(self) -> bool:
        return len(self.pts) == 2

    @property
    def start(self) -> np.ndarray:
        return self.pts[0]

    @property
    def end(self) -> np.ndarray:
        return self.pts[-1]

    def length(self) -> float:
        return polyline_length(prim_points(self, 16))


def quad_eval(q: np.ndarray, t) -> np.ndarray:
    t = np.asarray(t, float)[:, None]
    mt = 1.0 - t
    return mt * mt * q[0] + 2.0 * mt * t * q[1] + t * t * q[2]


def quad_to_cubic(q: np.ndarray) -> np.ndarray:
    """Exact degree elevation of a quadratic to a cubic."""
    q = np.asarray(q, float)
    return np.stack([q[0], q[0] + 2.0 / 3.0 * (q[1] - q[0]), q[2] + 2.0 / 3.0 * (q[1] - q[2]), q[2]])


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


def transform_prim(p: Prim, matrix: np.ndarray, scale: float = 1.0) -> Prim:
    """Apply a 2x3 affine to the control points (exact for Beziers); the
    width is multiplied by `scale`."""
    pts = np.asarray(p.pts, float) @ matrix[:, :2].T + matrix[:, 2]
    return Prim(pts, p.width * scale, p.conf, dict(p.meta))


# ---------------------------------------------------------------------------
# Cubic stroke -> primitives (training targets)
# ---------------------------------------------------------------------------
def cubic_to_lines(piece: np.ndarray, tol: float, depth: int = 10) -> list[np.ndarray]:
    """Flatten one cubic into chords `(2, 2)` within `tol` (recursive halving)."""
    piece = np.asarray(piece, float)
    if depth <= 0 or piece_is_flat(piece, tol):
        return [np.stack([piece[0], piece[3]])]
    a, b = split_cubic_at(piece, 0.5)
    return cubic_to_lines(a, tol, depth - 1) + cubic_to_lines(b, tol, depth - 1)


def merge_chords(chords: list[np.ndarray], angle_deg: float) -> list[np.ndarray]:
    """Join consecutive chords (each starting where the last ended) whose
    directions stay within `angle_deg` of the running line's."""
    out: list[np.ndarray] = []
    cos_tol = math.cos(math.radians(angle_deg))
    for c in chords:
        if np.hypot(*(c[1] - c[0])) < 1e-9:
            continue
        if out:
            last = out[-1]
            if np.allclose(last[1], c[0], atol=1e-6):
                d0 = last[1] - last[0]
                d1 = c[1] - c[0]
                cos = float(np.dot(d0, d1)) / (np.hypot(*d0) * np.hypot(*d1))
                d2 = c[1] - last[0]
                cos2 = float(np.dot(d0, d2)) / (np.hypot(*d0) * max(np.hypot(*d2), 1e-9))
                if cos >= cos_tol and cos2 >= cos_tol:
                    out[-1] = np.stack([last[0], c[1]])
                    continue
        out.append(np.array(c, float))
    return out


def stroke_to_lines(stroke: Stroke, tol: float, angle_deg: float) -> list[np.ndarray]:
    chords = [c for piece in stroke for c in cubic_to_lines(piece, tol)]
    return merge_chords(chords, angle_deg)


def fit_quad(piece: np.ndarray, samples: int = 16) -> tuple[np.ndarray, float]:
    """Least-squares quadratic with the cubic's endpoints fixed; returns the
    quadratic and its max sample error (px)."""
    piece = np.asarray(piece, float)
    t = np.linspace(0.0, 1.0, samples + 1)
    pts = bezier_eval(piece, t)
    b0, b1, b2 = (1 - t) ** 2, 2 * t * (1 - t), t ** 2
    resid = pts - b0[:, None] * piece[0] - b2[:, None] * piece[3]
    den = float(np.sum(b1 * b1))
    ctrl = (b1[:, None] * resid).sum(axis=0) / den if den > 0 else 0.5 * (piece[0] + piece[3])
    q = np.stack([piece[0], ctrl, piece[3]])
    err = float(np.max(np.linalg.norm(quad_eval(q, t) - pts, axis=1)))
    return q, err


def cubic_to_quads(piece: np.ndarray, tol: float, depth: int = 8) -> list[np.ndarray]:
    q, err = fit_quad(piece)
    if err <= tol or depth <= 0:
        return [q]
    a, b = split_cubic_at(np.asarray(piece, float), 0.5)
    return cubic_to_quads(a, tol, depth - 1) + cubic_to_quads(b, tol, depth - 1)


def stroke_to_quads(stroke: Stroke, tol: float) -> list[np.ndarray]:
    return [q for piece in stroke for q in cubic_to_quads(piece, tol)]


# ---------------------------------------------------------------------------
# Paper's canonical ordering (Sec. 3.2: "we sort the endpoints in each target
# primitive and the target primitives by their parameters lexicographically")
# ---------------------------------------------------------------------------
def _lex(p) -> tuple[float, float]:
    return (round(float(p[0]), 3), round(float(p[1]), 3))


def orient_pts(pts: np.ndarray) -> np.ndarray:
    """Endpoints in lexicographic order (a quadratic is reversed whole, so
    its control point stays in the middle)."""
    pts = np.asarray(pts, float)
    if _lex(pts[-1]) < _lex(pts[0]):
        return pts[::-1].copy()
    return pts


def sort_prims(prims: list[Prim]) -> list[Prim]:
    oriented = [Prim(orient_pts(p.pts), p.width, p.conf, p.meta) for p in prims]
    return sorted(oriented, key=lambda p: tuple(v for pt in p.pts for v in _lex(pt)) + (round(p.width, 3),))


# ---------------------------------------------------------------------------
# Patch grid (copied from DeepVectoriser/geometry.py)
# ---------------------------------------------------------------------------
def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = max(1, tile - overlap)
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def tile_grid(h: int, w: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    """Top-left `(x0, y0)` of every tile covering an `h`x`w` image."""
    return [(x, y) for y in tile_starts(h, tile, overlap) for x in tile_starts(w, tile, overlap)]


def core_rect(x0: int, y0: int, size: int, h: int, w: int, overlap: int) -> tuple[float, float, float, float]:
    """The part of a tile it "owns": its rect shrunk by half the overlap on
    every side that has a neighbour (image borders keep their full extent)."""
    half = overlap / 2.0
    cx0 = x0 + half if x0 > 0 else float("-inf")
    cy0 = y0 + half if y0 > 0 else float("-inf")
    cx1 = x0 + size - half if x0 + size < w else float("inf")
    cy1 = y0 + size - half if y0 + size < h else float("inf")
    return (cx0, cy0, cx1, cy1)
