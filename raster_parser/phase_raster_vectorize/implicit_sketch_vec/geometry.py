"""Pure numpy geometry shared by inference (`inference.py`, `postprocess.py`,
`adapter.py`) and training (`train_data.py`).

* cubic Bezier *strokes* `(K, 4, 2)` -- the shared dataset's ground truth
  (helpers shared via `commons.raster_text.strokes`);
* dense uniform sampling of strokes (the UDF / dual-contouring ground truth);
* the raster tile grid (shared: `commons.raster_text.strokes`);
* the paper's two smoothing options for an extracted polyline (Sec. 7):
  Schneider's Bezier fitting (Graphics Gems, 1990) and Ramer-Douglas-Peucker.

Coordinates are pixels of whatever frame the caller works in.
"""

from __future__ import annotations

import math

import numpy as np
from raster_parser.commons.raster_text.strokes import (
    Stroke,
    _refine,
    bezier_eval,
    sample_stroke,
    tile_starts,
)
from raster_parser.commons.raster_text.strokes import (  # noqa: F401 -- moved to commons; still reachable from this module
    core_spans,
    fit_cubics,
    line_to_cubic,
    rdp,
)


# ---------------------------------------------------------------------------
# Cubic strokes (shared: `commons.raster_text.strokes`)
# ---------------------------------------------------------------------------
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
# Tile grid (shared: `commons.raster_text.strokes`)
# ---------------------------------------------------------------------------
def tile_grid(h: int, w: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    """Top-left `(x0, y0)` of every tile covering an `h`x`w` image."""
    return [(x, y) for y in tile_starts(h, tile, overlap) for x in tile_starts(w, tile, overlap)]


# ---------------------------------------------------------------------------
# Smoothing: Ramer-Douglas-Peucker
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Smoothing: Schneider's least-squares cubic fitting (Graphics Gems 1990)
# ---------------------------------------------------------------------------
