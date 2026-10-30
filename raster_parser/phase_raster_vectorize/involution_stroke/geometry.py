"""Pure numpy geometry for InvolutionStroke's inference and adapter:

* the raster tile grid and per-tile core ownership (shared with the other stroke backends:
  `commons.raster_text.strokes`);
* clipping a polyline to a tile's core rect (ours);
* smoothing a dense stroke polyline into cubic pieces: split at sharp
  corners (ours), then Schneider's Bezier fit or Ramer-Douglas-Peucker
  (shared: `commons.raster_text.strokes`).

Coordinates are pixels of whatever frame the caller works in.
"""

from __future__ import annotations

import math

import numpy as np
from raster_parser.commons.raster_text.strokes import (
    Stroke,
    fit_cubics,
    line_to_cubic,
    rdp,
    tile_starts,
)
from raster_parser.commons.raster_text.strokes import (  # noqa: F401 -- moved to commons; still reachable from this module
    piece_is_flat,
    sample_stroke,
)
from raster_parser.commons.raster_text.strokes import (  # noqa: F401 -- moved to commons; still reachable from this module
    core_spans,
)


# ---------------------------------------------------------------------------
# Cubic strokes (shared: `commons.raster_text.strokes`)
# ---------------------------------------------------------------------------
def polyline_length(pts: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0


# ---------------------------------------------------------------------------
# Tile grid (shared: `commons.raster_text.strokes`)
# ---------------------------------------------------------------------------
def tile_grid(h: int, w: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    """Top-left `(x0, y0)` of every tile covering an `h`x`w` image."""
    return [(x, y) for y in tile_starts(h, tile, overlap) for x in tile_starts(w, tile, overlap)]


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
# (shared: `commons.raster_text.strokes`)
# ---------------------------------------------------------------------------
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


# Presmoothing needs at least this many points in a run.
_PRESMOOTH_MIN_POINTS = 5


def presmooth(run: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian-smooth a dense run along its length (`sigma_px` of arc
    length), ends pinned. The upstream polylines are 0.5 px staircases;
    fitting them raw takes ~10x the pieces for the same accuracy."""
    from scipy.ndimage import gaussian_filter1d

    if sigma_px <= 0 or len(run) < _PRESMOOTH_MIN_POINTS:
        return run
    step = float(np.median(np.linalg.norm(np.diff(run, axis=0), axis=1)))
    if step <= 0:
        return run
    sigma = sigma_px / step
    out = np.stack([gaussian_filter1d(run[:, k], sigma, mode="nearest") for k in (0, 1)], axis=1)
    out[0], out[-1] = run[0], run[-1]
    return out


def _straight_fit(
    run: np.ndarray, tol: float, trim: float
) -> "tuple[np.ndarray, np.ndarray] | None":
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


def smooth_stroke(
    pts: np.ndarray,
    mode: str,
    tol: float,
    corner_deg: float,
    *,
    corner_eps: float = 2.0,
    presmooth_px: float = 1.0,
) -> Stroke:
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
    runs = [pts[a : b + 1] for a, b in zip(cuts, cuts[1:])]
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
