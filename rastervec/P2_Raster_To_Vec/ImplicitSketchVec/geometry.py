"""Pure numpy geometry shared by inference (`inference.py`, `postprocess.py`,
`adapter.py`) and training (`train_data.py`).

* cubic Bezier *strokes* `(K, 4, 2)` -- the shared dataset's ground truth
  (helpers copied from DeepVectoriser/geometry.py);
* dense uniform sampling of strokes (the UDF / dual-contouring ground truth);
* the raster tile grid (copied from DeepVectoriser/geometry.py);
* the paper's two smoothing options for an extracted polyline (Sec. 7):
  Schneider's Bezier fitting (Graphics Gems, 1990) and Ramer-Douglas-Peucker.

Coordinates are pixels of whatever frame the caller works in.
"""
from __future__ import annotations

import math

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


def bezier_deriv(piece: np.ndarray, t) -> np.ndarray:
    t = np.asarray(t, float)[:, None]
    mt = 1.0 - t
    p0, c1, c2, p3 = piece
    return 3 * mt ** 2 * (c1 - p0) + 6 * mt * t * (c2 - c1) + 3 * t ** 2 * (p3 - c2)


def sample_stroke(stroke: Stroke, per_piece: int = 16) -> np.ndarray:
    """Polyline `(N, 2)` through every piece (shared joints not repeated)."""
    if len(stroke) == 0:
        return np.zeros((0, 2))
    t = np.linspace(0.0, 1.0, per_piece + 1)
    pts = [bezier_eval(stroke[0], t)]
    for piece in stroke[1:]:
        pts.append(bezier_eval(piece, t)[1:])
    return np.concatenate(pts, axis=0)


def polyline_length(pts: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0


def stroke_length(stroke: Stroke) -> float:
    return polyline_length(sample_stroke(stroke, 8))


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


def _refine(piece, inside, t_in: float, t_out: float, iters: int = 12) -> float:
    for _ in range(iters):
        mid = 0.5 * (t_in + t_out)
        if inside(bezier_eval(piece, [mid]))[0]:
            t_in = mid
        else:
            t_out = mid
    return t_in


def clip_stroke(stroke: Stroke, rect, samples: int = 32, min_len: float = 0.5) -> list[Stroke]:
    """The parts of `stroke` inside axis-aligned `rect = (x0, y0, x1, y1)`,
    each a stroke of its own (leaving and re-entering splits it)."""
    x0, y0, x1, y1 = rect

    def inside(pts: np.ndarray) -> np.ndarray:
        return (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)

    out: list[Stroke] = []
    cur: list[np.ndarray] = []

    def flush():
        if cur:
            s = np.stack(cur)
            if stroke_length(s) >= min_len:
                out.append(s)
            cur.clear()

    t = np.linspace(0.0, 1.0, samples + 1)
    for piece in stroke:
        piece = np.asarray(piece, float)
        mask = inside(bezier_eval(piece, t))
        if mask.all():
            cur.append(piece)
            continue
        if not mask.any():
            flush()
            continue
        i = 0
        while i < len(t):
            if not mask[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(t) and mask[j + 1]:
                j += 1
            ta = 0.0 if i == 0 else _refine(piece, inside, t[i], t[i - 1])
            tb = 1.0 if j == len(t) - 1 else _refine(piece, inside, t[j], t[j + 1])
            if i != 0:
                flush()
            if tb > ta:
                cur.append(split_bezier(piece, ta, tb))
            if j != len(t) - 1:
                flush()
            i = j + 1
    flush()
    return out


def dense_samples(stroke: Stroke, step: float) -> np.ndarray:
    """Points along the stroke spaced at most `step` apart (per piece, by its
    sampled length)."""
    out = []
    for k, piece in enumerate(stroke):
        approx = polyline_length(bezier_eval(piece, np.linspace(0, 1, 9)))
        n = max(2, int(math.ceil(approx / step)) + 1)
        t = np.linspace(0.0, 1.0, n)
        pts = bezier_eval(piece, t)
        out.append(pts if k == 0 else pts[1:])
    return np.concatenate(out, axis=0) if out else np.zeros((0, 2))


# ---------------------------------------------------------------------------
# Tile grid (copied from DeepVectoriser/geometry.py)
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


def core_spans(total: int, tile: int, overlap: int) -> dict[int, tuple[int, int]]:
    """Tile start -> the integer pixel range `[a, b)` along one axis that tile
    owns. Neighbouring tiles split their *actual* overlap at its middle (the
    last tile is shifted back to the edge, so its overlap with the previous
    one can be much more than `overlap`); the cores tile the axis exactly."""
    starts = tile_starts(total, tile, overlap)
    out = {}
    for i, s in enumerate(starts):
        a = 0 if i == 0 else (s + starts[i - 1] + tile) // 2
        b = total if i == len(starts) - 1 else (starts[i + 1] + s + tile) // 2
        out[s] = (a, b)
    return out


# ---------------------------------------------------------------------------
# Smoothing: Ramer-Douglas-Peucker
# ---------------------------------------------------------------------------
def rdp(pts: np.ndarray, eps: float) -> np.ndarray:
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        d = b - a
        n = float(np.hypot(*d))
        seg = pts[i + 1:j] - a
        dist = np.abs(d[0] * seg[:, 1] - d[1] * seg[:, 0]) / n if n > 1e-12 else np.linalg.norm(seg, axis=1)
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack.extend([(i, m), (m, j)])
    return pts[keep]


# ---------------------------------------------------------------------------
# Smoothing: Schneider's least-squares cubic fitting (Graphics Gems 1990)
# ---------------------------------------------------------------------------
def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(*v))
    return v / n if n > 1e-12 else np.zeros(2)


def _chord_params(pts: np.ndarray) -> np.ndarray:
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return d / d[-1] if d[-1] > 0 else np.linspace(0, 1, len(pts))


def _generate(pts: np.ndarray, u: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    p0, p3 = pts[0], pts[-1]
    b0, b1, b2, b3 = (1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u ** 2 * (1 - u), u ** 3
    a1 = b1[:, None] * t1
    a2 = b2[:, None] * t2
    c00, c01, c11 = float((a1 * a1).sum()), float((a1 * a2).sum()), float((a2 * a2).sum())
    tmp = pts - (b0 + b1)[:, None] * p0 - (b2 + b3)[:, None] * p3
    x0, x1 = float((a1 * tmp).sum()), float((a2 * tmp).sum())
    det = c00 * c11 - c01 * c01
    seg = float(np.hypot(*(p3 - p0)))
    if abs(det) > 1e-12:
        al1, al2 = (x0 * c11 - x1 * c01) / det, (c00 * x1 - c01 * x0) / det
    else:
        al1 = al2 = seg / 3.0
    eps = 1e-6 * seg
    if al1 < eps or al2 < eps:
        al1 = al2 = seg / 3.0
    return np.stack([p0, p0 + al1 * t1, p3 + al2 * t2, p3])


def _max_error(pts: np.ndarray, bez: np.ndarray, u: np.ndarray) -> tuple[float, int]:
    d = np.linalg.norm(bezier_eval(bez, u) - pts, axis=1)
    k = int(np.argmax(d[1:-1])) + 1 if len(pts) > 2 else len(pts) // 2
    return float(d[k]), k


def _reparam(pts: np.ndarray, bez: np.ndarray, u: np.ndarray) -> np.ndarray:
    """One Newton-Raphson step per point toward its nearest curve parameter."""
    q = bezier_eval(bez, u)
    q1 = bezier_deriv(bez, u)
    d2 = np.stack([6 * (1 - u) * (bez[2] - 2 * bez[1] + bez[0])[0] + 6 * u * (bez[3] - 2 * bez[2] + bez[1])[0],
                   6 * (1 - u) * (bez[2] - 2 * bez[1] + bez[0])[1] + 6 * u * (bez[3] - 2 * bez[2] + bez[1])[1]], 1)
    num = ((q - pts) * q1).sum(1)
    den = (q1 * q1).sum(1) + ((q - pts) * d2).sum(1)
    step = np.where(np.abs(den) > 1e-12, num / np.where(np.abs(den) > 1e-12, den, 1.0), 0.0)
    return np.clip(u - step, 0.0, 1.0)


def fit_cubics(pts: np.ndarray, tol: float, max_depth: int = 12) -> list[np.ndarray]:
    """Cubic pieces `(4, 2)` fitting the polyline within `tol` px."""
    pts = np.asarray(pts, float)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9])
    pts = pts[keep]
    if len(pts) < 2:
        return []
    if len(pts) == 2:
        return [line_to_cubic(pts[0], pts[1])]
    t1 = _unit(pts[1] - pts[0])
    t2 = _unit(pts[-2] - pts[-1])
    return _fit(pts, t1, t2, tol, max_depth)


def _fit(pts, t1, t2, tol, depth) -> list[np.ndarray]:
    if len(pts) == 2:
        dist = float(np.hypot(*(pts[1] - pts[0]))) / 3.0
        return [np.stack([pts[0], pts[0] + t1 * dist, pts[1] + t2 * dist, pts[1]])]
    u = _chord_params(pts)
    bez = _generate(pts, u, t1, t2)
    err, split = _max_error(pts, bez, u)
    if err < tol:
        return [bez]
    # Newton reparameterisation before giving up on one piece (the original
    # only tries within 4x the tolerance; chord-length parameters are often
    # further off than that on a single smooth curve, so we always try)
    for _ in range(8):
        u = _reparam(pts, bez, u)
        bez2 = _generate(pts, u, t1, t2)
        err2, split2 = _max_error(pts, bez2, u)
        if err2 >= err - 1e-9:
            break
        bez, err, split = bez2, err2, split2
        if err < tol:
            return [bez]
    if depth <= 0:
        return [bez]
    split = min(max(split, 1), len(pts) - 2)
    tc = _unit(pts[split - 1] - pts[split + 1])
    return _fit(pts[:split + 1], t1, tc, tol, depth - 1) + _fit(pts[split:], -tc, t2, tol, depth - 1)
