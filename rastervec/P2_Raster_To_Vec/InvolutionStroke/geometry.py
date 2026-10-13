"""Pure numpy geometry for InvolutionStroke's inference and adapter:

* the raster tile grid and per-tile core ownership (copied from
  ImplicitSketchVec/geometry.py, originally DeepVectoriser's);
* clipping a polyline to a tile's core rect (ours);
* smoothing a dense stroke polyline into cubic pieces: split at sharp
  corners (ours), then Schneider's Bezier fit or Ramer-Douglas-Peucker
  (copied from ImplicitSketchVec/geometry.py).

Coordinates are pixels of whatever frame the caller works in.
"""
from __future__ import annotations

import math

import numpy as np

Stroke = np.ndarray  # (K, 4, 2) cubic pieces


# ---------------------------------------------------------------------------
# Cubic strokes (copied from ImplicitSketchVec/geometry.py)
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


# ---------------------------------------------------------------------------
# Tile grid (copied from ImplicitSketchVec/geometry.py)
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
# Clipping a polyline to an axis-aligned rect (ours)
# ---------------------------------------------------------------------------
def _crossing(p: np.ndarray, q: np.ndarray, rect) -> np.ndarray:
    """The point where segment p (inside) -> q (outside) leaves `rect`."""
    x0, y0, x1, y1 = rect
    d = q - p
    t = 1.0
    for lo, hi, pc, dc in ((x0, x1, p[0], d[0]), (y0, y1, p[1], d[1])):
        if dc > 0:
            t = min(t, (hi - pc) / dc)
        elif dc < 0:
            t = min(t, (lo - pc) / dc)
    return p + max(0.0, t) * d


def clip_polyline(pts: np.ndarray, rect, min_len: float = 0.5) -> list[np.ndarray]:
    """The parts of polyline `pts` inside `rect = (x0, y0, x1, y1)` (closed),
    each its own polyline; segment/border crossings are interpolated exactly,
    so a stroke cut by two neighbouring cores meets itself on the border."""
    pts = np.asarray(pts, float)
    x0, y0, x1, y1 = rect
    inside = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
    out: list[np.ndarray] = []
    cur: list[np.ndarray] = []

    def flush():
        if len(cur) >= 2 and polyline_length(np.array(cur)) >= min_len:
            out.append(np.array(cur))
        cur.clear()

    for i, p in enumerate(pts):
        if inside[i]:
            if not cur and i > 0:
                cur.append(_crossing(p, pts[i - 1], rect))
            cur.append(p)
        else:
            if cur:
                cur.append(_crossing(pts[i - 1], p, rect))
                flush()
            elif i > 0:
                # both ends outside: the segment may still pass through
                a, b = pts[i - 1], p
                ta, tb = _slab(a, b, rect)
                if ta < tb:
                    cur.extend([a + ta * (b - a), a + tb * (b - a)])
                    flush()
    flush()
    return out


def _slab(a: np.ndarray, b: np.ndarray, rect) -> tuple[float, float]:
    """Liang-Barsky parameter interval of segment a->b inside `rect`."""
    x0, y0, x1, y1 = rect
    d = b - a
    t0, t1 = 0.0, 1.0
    for p, q in ((-d[0], a[0] - x0), (d[0], x1 - a[0]), (-d[1], a[1] - y0), (d[1], y1 - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return 1.0, 0.0
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
    return t0, t1


# ---------------------------------------------------------------------------
# Smoothing: corners (ours), Ramer-Douglas-Peucker + Schneider's cubic fit
# (copied from ImplicitSketchVec/geometry.py)
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


def rdp_indices(pts: np.ndarray, eps: float) -> np.ndarray:
    """Indices into `pts` of the vertices `rdp` keeps."""
    pts = np.asarray(pts, float)
    kept = rdp(pts, eps)
    out, j = [], 0
    for q in kept:
        while not np.array_equal(pts[j], q):
            j += 1
        out.append(j)
    return np.array(out, int)


def corner_splits(pts: np.ndarray, eps: float, corner_deg: float) -> list[int]:
    """Interior indices of `pts` where the RDP-simplified polyline turns by
    more than `corner_deg` -- the stroke's sharp corners."""
    idx = rdp_indices(pts, eps)
    out = []
    for a, b, c in zip(idx, idx[1:], idx[2:]):
        u, v = pts[b] - pts[a], pts[c] - pts[b]
        nu, nv = float(np.hypot(*u)), float(np.hypot(*v))
        if nu < 1e-9 or nv < 1e-9:
            continue
        cos = float(np.dot(u, v)) / (nu * nv)
        if math.degrees(math.acos(max(-1.0, min(1.0, cos)))) > corner_deg:
            out.append(int(b))
    return out


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


def presmooth(run: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian-smooth a dense run along its length (`sigma_px` of arc
    length), ends pinned. The upstream polylines are 0.5 px staircases;
    fitting them raw takes ~10x the pieces for the same accuracy."""
    from scipy.ndimage import gaussian_filter1d

    if sigma_px <= 0 or len(run) < 5:
        return run
    step = float(np.median(np.linalg.norm(np.diff(run, axis=0), axis=1)))
    if step <= 0:
        return run
    sigma = sigma_px / step
    out = np.stack([gaussian_filter1d(run[:, k], sigma, mode="nearest") for k in (0, 1)], axis=1)
    out[0], out[-1] = run[0], run[-1]
    return out


def _straight_fit(run: np.ndarray, tol: float, trim: float) -> "tuple[np.ndarray, np.ndarray] | None":
    """`(point, unit direction)` of the run's least-squares line when its
    interior -- `trim` px of arc length cut off each end, where a centerline
    rounds or chamfers into a corner -- is straight within `tol`; else None."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(run, axis=0), axis=1))])
    if s[-1] <= 3 * trim:
        inner = run
    else:
        inner = run[(s >= trim) & (s <= s[-1] - trim)]
    if len(inner) < 2 or len(rdp(inner, tol)) != 2:
        return None
    c = inner.mean(axis=0)
    _, _, vt = np.linalg.svd(inner - c, full_matrices=False)
    return c, vt[0]


def _intersect(l1, l2) -> "np.ndarray | None":
    (p, d), (q, e) = l1, l2
    den = d[0] * e[1] - d[1] * e[0]
    if abs(den) < 0.2:  # nearly parallel (< ~11.5 deg)
        return None
    t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den
    return p + t * d


def _project(line, pt: np.ndarray) -> np.ndarray:
    p, d = line
    return p + float(np.dot(pt - p, d)) * d


def smooth_stroke(pts: np.ndarray, mode: str, tol: float, corner_deg: float, *, corner_eps: float = 2.0,
                  presmooth_px: float = 1.0) -> Stroke:
    """A dense stroke polyline -> cubic pieces `(K, 4, 2)`:

    1. cut at sharp corners (turns > `corner_deg` on the `corner_eps`-RDP'd
       stroke, so a rectangle keeps its corners);
    2. a run whose interior is straight within `tol` becomes one line, on
       its least-squares fit; a corner between two straight runs moves to
       their lines' intersection (undoing the centerline's chamfer), a free
       end is projected onto its line;
    3. the other runs are RDP'd into lines (`mode="rdp"`) or pre-smoothed
       and Schneider-fitted (`mode="bezier"`) between those corner points."""
    pts = np.asarray(pts, float)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9])
    pts = pts[keep]
    if len(pts) < 2:
        return np.zeros((0, 4, 2))
    cuts = [0] + corner_splits(pts, corner_eps, corner_deg) + [len(pts) - 1]
    runs = [pts[a:b + 1] for a, b in zip(cuts, cuts[1:])]
    lines = [_straight_fit(r, tol, 2.0 * corner_eps) for r in runs]
    closed = len(runs) > 1 and float(np.hypot(*(pts[0] - pts[-1]))) < 1e-6

    # vertex k sits between runs k-1 and k (vertex 0 / len(runs): the ends)
    verts = [pts[c].copy() for c in cuts]
    for k in range(len(verts)):
        prev = k - 1 if k > 0 else (len(runs) - 1 if closed else None)
        nxt = k if k < len(runs) else (0 if closed else None)
        lp = lines[prev] if prev is not None else None
        ln = lines[nxt] if nxt is not None else None
        v = verts[k]
        if lp is not None and ln is not None and prev != nxt:
            x = _intersect(lp, ln)
            if x is not None and float(np.hypot(*(x - v))) <= 3.0 * corner_eps:
                verts[k] = x
        elif prev is None and ln is not None:
            verts[k] = _project(ln, v)
        elif nxt is None and lp is not None:
            verts[k] = _project(lp, v)
    if closed:
        verts[-1] = verts[0]

    pieces: list[np.ndarray] = []
    for k, run in enumerate(runs):
        va, vb = verts[k], verts[k + 1]
        if lines[k] is not None:
            pieces.append(line_to_cubic(va, vb))
            continue
        run = run.copy()
        run[0], run[-1] = va, vb
        if mode == "rdp":
            r = rdp(run, tol)
            pieces += [line_to_cubic(p, q) for p, q in zip(r, r[1:])]
        else:
            pieces += fit_cubics(presmooth(run, presmooth_px), tol)
    return np.stack(pieces) if pieces else np.zeros((0, 4, 2))
