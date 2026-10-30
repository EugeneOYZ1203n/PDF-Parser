"""Stroke geometry and per-layer plumbing shared by the stroke-vectorising raster backends
(`involution_stroke`, `deep_tech_vec`, `deep_vectoriser`, `implicit_sketch_vec`): cubic Bézier
strokes (`Stroke`, `(K, 4, 2)` pieces) -- evaluation, flattening, Schneider-style cubic fitting,
RDP -- overlapping tile layout, a colour layer's mask image, and a stroke as a page-space `Vector`.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from raster_parser.commons.helpers.geometry import points_bbox
from raster_parser.models.vector import Vector
from raster_parser.commons.raster_text.debug_tools import traced_vector_kwargs

# A cubic piece whose control points sit within this many pixels of its chord is drawn as a line.
FLAT_TOL_PX = 0.75
# torch intra-op threads for CPU inference (None = torch's default).
INFER_THREADS: "int | None" = None

Mapper = Callable[[tuple[float, float]], tuple[float, float]]


Stroke = np.ndarray  # (K, 4, 2) cubic pieces


def bezier_eval(piece: np.ndarray, t) -> np.ndarray:
    """Points `(len(t), 2)` of one cubic at parameters `t`."""
    t = np.asarray(t, float)[:, None]
    mt = 1.0 - t
    p0, c1, c2, p3 = piece
    return mt**3 * p0 + 3 * mt**2 * t * c1 + 3 * mt * t**2 * c2 + t**3 * p3


def bezier_deriv(piece: np.ndarray, t) -> np.ndarray:
    t = np.asarray(t, float)[:, None]
    mt = 1.0 - t
    p0, c1, c2, p3 = piece
    return 3 * mt**2 * (c1 - p0) + 6 * mt * t * (c2 - c1) + 3 * t**2 * (p3 - c2)


def line_to_cubic(a, b) -> np.ndarray:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return np.stack([a, a + (b - a) / 3.0, a + (b - a) * 2.0 / 3.0, b])


def sample_stroke(stroke: Stroke, per_piece: int = 16) -> np.ndarray:
    """Polyline `(N, 2)` through every piece (shared joints not repeated)."""
    if len(stroke) == 0:
        return np.zeros((0, 2))
    t = np.linspace(0.0, 1.0, per_piece + 1)
    pts = [bezier_eval(stroke[0], t)]
    for piece in stroke[1:]:
        pts.append(bezier_eval(piece, t)[1:])
    return np.concatenate(pts, axis=0)


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


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = max(1, tile - overlap)
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


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
        seg = pts[i + 1 : j] - a
        dist = (
            np.abs(d[0] * seg[:, 1] - d[1] * seg[:, 0]) / n
            if n > 1e-12
            else np.linalg.norm(seg, axis=1)
        )
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack.extend([(i, m), (m, j)])
    return pts[keep]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(*v))
    return v / n if n > 1e-12 else np.zeros(2)


def _chord_params(pts: np.ndarray) -> np.ndarray:
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return d / d[-1] if d[-1] > 0 else np.linspace(0, 1, len(pts))


def _generate(pts: np.ndarray, u: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    p0, p3 = pts[0], pts[-1]
    b0, b1, b2, b3 = (1 - u) ** 3, 3 * u * (1 - u) ** 2, 3 * u**2 * (1 - u), u**3
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
    d2 = np.stack(
        [
            6 * (1 - u) * (bez[2] - 2 * bez[1] + bez[0])[0]
            + 6 * u * (bez[3] - 2 * bez[2] + bez[1])[0],
            6 * (1 - u) * (bez[2] - 2 * bez[1] + bez[0])[1]
            + 6 * u * (bez[3] - 2 * bez[2] + bez[1])[1],
        ],
        1,
    )
    num = ((q - pts) * q1).sum(1)
    den = (q1 * q1).sum(1) + ((q - pts) * d2).sum(1)
    step = np.where(np.abs(den) > 1e-12, num / np.where(np.abs(den) > 1e-12, den, 1.0), 0.0)
    return np.clip(u - step, 0.0, 1.0)


def _fit(pts, t1, t2, tol, depth) -> list[np.ndarray]:
    if len(pts) == 2:
        dist = float(np.hypot(*(pts[1] - pts[0]))) / 3.0
        return [np.stack([pts[0], pts[0] + t1 * dist, pts[1] + t2 * dist, pts[1]])]
    u = _chord_params(pts)
    bez = _generate(pts, u, t1, t2)
    err, split = _max_error(pts, bez, u)
    if err < tol:
        return [bez]
    for _ in range(_REPARAM_PASSES):
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
    return _fit(pts[: split + 1], t1, tc, tol, depth - 1) + _fit(
        pts[split:], -tc, t2, tol, depth - 1
    )


# Schneider fit: recursion depth (pieces up to 2**depth) and Newton reparametrisation passes.
_FIT_MAX_DEPTH = 12
_REPARAM_PASSES = 8
# A layer-mask scale this close to 1 is used as is (no resampling).
_NO_RESAMPLE_TOL = 0.02
# Points sampled per cubic piece for the stroke-width estimate.
_WIDTH_SAMPLES_PER_PIECE = 4


def fit_cubics(pts: np.ndarray, tol: float, max_depth: int = _FIT_MAX_DEPTH) -> list[np.ndarray]:
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


def rect_predicate(rect):
    """`inside(points (N, 2)) -> bool (N,)` for `rect = (x0, y0, x1, y1)`."""
    x0, y0, x1, y1 = rect

    def inside(pts: np.ndarray) -> np.ndarray:
        return (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)

    return inside


def _refine(piece, inside, t_in: float, t_out: float, iters: int = 12) -> float:
    """Bisect for the boundary between an inside and an outside parameter."""
    for _ in range(iters):
        mid = 0.5 * (t_in + t_out)
        if inside(bezier_eval(piece, [mid]))[0]:
            t_in = mid
        else:
            t_out = mid
    return t_in


def layer_image(labels: np.ndarray, layer: int, scale: float) -> np.ndarray:
    """One color layer as a gray mask (ink 0, everything else 255),
    resampled to the canonical scale (area-averaged when shrinking, so thin
    lines stay as gray rather than vanishing). Shared by inference and
    `prep_dataset.py` so the model always sees the same kind of input."""
    gray = np.where(labels == layer, 0, 255).astype(np.uint8)
    if abs(scale - 1.0) < _NO_RESAMPLE_TOL:
        return gray
    h, w = gray.shape
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(gray, size, interpolation=interp)


def stroke_width_px(stroke: np.ndarray, dist: np.ndarray) -> float:
    """2 x the median distance-to-background along the stroke (>= 1 px)."""
    pts = np.round(sample_stroke(stroke, _WIDTH_SAMPLES_PER_PIECE) - 0.5).astype(int)
    h, w = dist.shape
    ok = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    if not ok.any():
        return 1.0
    vals = dist[pts[ok, 1], pts[ok, 0]]
    vals = vals[vals > 0]
    return max(1.0, 2.0 * float(np.median(vals))) if len(vals) else 1.0


def stroke_to_vector(
    stroke: np.ndarray, to_page: Mapper, page_index: int, seqno: int, color, width_pt: float
) -> Vector:
    """One `Vector` per stroke: per Bezier piece an `"l"` (flat, judged in
    canonical px) or a `"c"` item, mapped to page space."""
    items = []
    pts: list[tuple[float, float]] = []
    for piece in stroke:
        p = [to_page(q) for q in piece]
        if piece_is_flat(piece, FLAT_TOL_PX):
            items.append(("l", p[0], p[3]))
        else:
            items.append(("c", p[0], p[1], p[2], p[3]))
        pts.extend(p)
    return Vector(
        type="s",
        items=items,
        width=width_pt,
        rect=points_bbox(pts),
        **traced_vector_kwargs(page_index, seqno, color),
    )


def for_cpu_inference(model):
    """Eval mode with frozen parameters (no autograd bookkeeping), plus the
    `INFER_THREADS` setting for this process."""
    import torch

    if INFER_THREADS:
        torch.set_num_threads(INFER_THREADS)
    model.eval()
    model.requires_grad_(False)
    return model
