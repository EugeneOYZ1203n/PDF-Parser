"""Training-data helpers for `prep_dataset.py` and `train.py` (numpy only;
never imported by the pipeline itself).

Ground truth is the repo's own vector -> raster labelling method:
`evaluation.labelling.visible_geometry.visible_geometry_for_page` -- every
drawing piece cut down to what the page render (`get_pixmap(Matrix(dpi/72))`)
actually shows, plus the render's
colour layers from the raster prelude's own split (the layers
inference sees) and each piece's layers. Pieces are unrotated page space,
mapped into the render's pixel frame through
`page.rotation_matrix * Matrix(dpi/72)`.

On-disk layout (one entry per page x ink color layer that has GT strokes,
`<key>` = `<page key>__L<layer>`):

    <out>/layers/<key>.gray.png      uint8 layer mask (canonical scale), lossless
                                     PNG, cropped to the layer's content +
                                     CROP_MARGIN_PX
    <out>/layers/<key>.strokes.npz   pieces (P,4,2) f32, offsets (n+1,) i64,
                                     widths (n,) f32 px, bboxes (n,4) f32,
                                     ink_pts (Q,2) i32 (x, y), color (3,) u8 --
                                     all in the crop's frame; origin (2,) i64 =
                                     crop's (x0, y0) on the page, page_shape (2,)
    <out>/pages/<page key>.json      per-page manifest (resume marker)
    <out>/index.json                 train/val layer keys + counts
    <cache>/<key>.gray.npy           the PNG decoded once by `train.py`
                                     (default <out>/cache/), memory-mapped

An older dataset with full-page `layers/<key>.gray.npy` files still loads.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from raster_parser.models.deep_vectoriser import Crop
from . import geometry as geo
from raster_parser.commons.raster_text.training import LayerData
from raster_parser.commons.raster_text.training import (  # noqa: F401 -- moved to commons; still reachable from this module
    gray_path,
    load_layer,
)

_SIZE_STEP = 32
SIZES = tuple(range(64, 256 + 1, _SIZE_STEP))  # the paper's 64..256 px crops; step ours
_INK_SAMPLES = 20000
CROP_MARGIN_PX = max(SIZES)  # stored context around a layer's content
PNG_COMPRESSION = 3  # cv2 PNG level: fast, ~15-100x smaller than raw


# ---------------------------------------------------------------------------
# Ground truth from a PDF page
# ---------------------------------------------------------------------------
# Pixels darker than this are ink when cropping a layer to its content.
_INK_GRAY_MAX = 200
# Training augmentation (ours; the paper gives none): probability of each effect and its range.
_AUG_THICKEN_P = 0.15
_AUG_BLUR_P, _AUG_BLUR_SIGMA = 0.3, (0.3, 0.8)
_AUG_NOISE_P, _AUG_NOISE_STD = 0.3, (2, 10)
_AUG_JPEG_P, _AUG_JPEG_QUALITY = 0.2, (60, 95)
# `cv2.polylines(..., shift=_SUBPIXEL_SHIFT)` reads fixed-point coordinates: x 2**shift.
_SUBPIXEL_SHIFT = 4
_SUBPIXEL = 1 << _SUBPIXEL_SHIFT


def chain_annotations(annotations, matrix, tol_px: float = 0.05) -> list[tuple[np.ndarray, float]]:
    """`GeometryPiece`s (in their extraction order) -> strokes in px.

    Consecutive annotations with the same paint (color, fill, width) whose
    end meets the next start (within `tol_px`) are chained into one stroke
    -- a polyline / rectangle / glyph contour is one stroke (the paper's
    stroke unit). `"l"` becomes a cubic with controls at 1/3 and 2/3. A
    piece repeating earlier geometry (either direction) is dropped -- two GT
    strokes over the same pixels would be unlearnable.
    Returns `[(stroke (K,4,2), width_px), ...]`; width = annotation width
    (1 px minimum; fill-only edges have none)."""
    return [(s, w) for s, w, _g in _chain(annotations, matrix, tol_px, None)]


def chain_annotations_by_layer(annotations, matrix, layers_of, tol_px: float = 0.05):
    """As `chain_annotations`, but a chain also never crosses a change of
    colour layers (`layers_of[i]`, from `visible_geometry_for_page`). Returns
    `[(stroke, width_px, layers), ...]`."""
    return _chain(annotations, matrix, tol_px, [tuple(g) for g in layers_of])


def _chain(annotations, matrix, tol_px: float, groups) -> list[tuple[np.ndarray, float, tuple]]:
    """Vectorized: every piece is mapped and keyed up front; only the
    order-dependent chaining walks the annotations one by one."""
    scale = float(np.hypot(matrix.a, matrix.b))
    usable, pieces = _pieces_px(annotations, matrix)
    if not usable:
        return []
    keep = np.ptp(pieces, axis=1).max(axis=1) >= 1e-6  # drop degenerate (a point)
    keys = _piece_keys(pieces)
    starts, ends = pieces[:, 0], pieces[:, 3]

    out: list[tuple[np.ndarray, float, tuple]] = []
    cur: list[int] = []
    cur_key = None
    cur_w = 1.0
    cur_g: tuple = ()
    seen: set = set()

    def flush():
        if cur:
            out.append((pieces[cur], cur_w, cur_g))
            cur.clear()

    for n, i in enumerate(usable):
        if not keep[n]:
            continue
        if keys[n] in seen:
            continue  # same geometry again (e.g. a closed 2-point path retracing itself)
        seen.add(keys[n])
        ann = annotations[i]
        g = groups[i] if groups is not None else ()
        key = (
            tuple(ann.color) if ann.color else None,
            tuple(ann.fill) if ann.fill else None,
            ann.width,
            g,
        )
        if cur and key == cur_key and math.hypot(*(ends[cur[-1]] - starts[n])) <= tol_px:
            cur.append(n)
            continue
        flush()
        cur.append(n)
        cur_key = key
        cur_g = g
        cur_w = max(1.0, float(ann.width or 0.0) * scale)
    flush()
    return out


def _pieces_px(annotations, matrix) -> tuple[list[int], np.ndarray]:
    """`(indices of usable annotations, their cubics (N,4,2) in px)`: an
    `"l"` needs 2 points, anything else 4. Points go through `matrix` in
    float32, exactly as MuPDF's `fz_transform_point` (`pymupdf.Point * Matrix`)
    computes them, then everything continues in float64."""
    usable: list[int] = []
    lines: list[bool] = []
    raw = np.zeros((len(annotations), 4, 2), np.float64)
    for i, ann in enumerate(annotations):
        pts = ann.points
        if ann.kind == "l":
            if len(pts) != 2:
                continue
            raw[len(usable), :2] = pts
            lines.append(True)
        else:
            if len(pts) != 4:
                continue
            raw[len(usable)] = pts
            lines.append(False)
        usable.append(i)
    raw = raw[: len(usable)].astype(np.float32)
    m = np.array([matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f], np.float32)
    x, y = raw[..., 0], raw[..., 1]
    px = np.stack([x * m[0] + y * m[2] + m[4], x * m[1] + y * m[3] + m[5]], axis=-1).astype(
        np.float64
    )
    is_line = np.array(lines, bool)
    a, b = px[is_line, 0], px[is_line, 1]
    px[is_line] = np.stack(
        [a, a + (b - a) / 3.0, a + (b - a) * 2.0 / 3.0, b], axis=1
    )  # geo.line_to_cubic
    return usable, px


def _piece_keys(pieces: np.ndarray) -> list[bytes]:
    """Direction-independent identity of each piece (0.01 px grid);
    `+ 0.0` folds -0.0 into 0.0 so equal coordinates give equal bytes."""
    fwd = np.round(pieces, 2) + 0.0
    rev = np.ascontiguousarray(fwd[:, ::-1])
    fwd = np.ascontiguousarray(fwd)
    return [min(f.tobytes(), r.tobytes()) for f, r in zip(fwd, rev)]


def page_ground_truth(pdf_path: str, page_index: int, dpi: float):
    """`(VisibleGeometry, [(stroke px, width px, layers)])` for one page --
    `geom.labels` / `geom.colors_rgb` are the render's colour layers."""
    from raster_parser.evaluation.labelling.visible_geometry import visible_geometry_for_page

    geom = visible_geometry_for_page(str(pdf_path), page_index, dpi)
    return geom, chain_annotations_by_layer(geom.annotations, geom.matrix, geom.layers_of)


def cut_by_mask(strokes, keep_mask: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Keep only the parts of each stroke over `keep_mask` (True = keep),
    splitting where they leave it (used to drop erased-text ink)."""
    h, w = keep_mask.shape

    def inside(pts: np.ndarray) -> np.ndarray:
        ij = np.floor(pts).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
        res = np.ones(len(pts), bool)  # off-image: not erased
        res[ok] = keep_mask[ij[ok, 1], ij[ok, 0]]
        return res

    out = []
    for s, wpx in strokes:
        for part in geo.clip_stroke_by(s, inside):
            out.append((part, wpx))
    return out


def scale_strokes(strokes, factor: float):
    if abs(factor - 1.0) < 1e-9:
        return strokes
    return [(s * factor, wpx * factor) for s, wpx, *_rest in strokes]


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def save_layer(layers_dir: Path, key: str, gray: np.ndarray, strokes, color, seed: int = 0) -> dict:
    """Save one (page, layer) training sample; returns `{"offset": [x0, y0],
    "shape": [h, w]}` of the stored crop.

    The mask is cropped to its content (ink pixels + GT strokes) plus
    `CROP_MARGIN_PX` -- the largest training crop, so every window around
    ink is still complete -- and stored as a lossless PNG; strokes, bboxes
    and ink samples are stored in that crop's frame."""
    layers_dir.mkdir(parents=True, exist_ok=True)
    h, w = gray.shape
    ys, xs = np.nonzero(gray < _INK_GRAY_MAX)
    n = len(strokes)
    xy = [s.reshape(-1, 2) for s, _ in strokes]
    lo = [m.min(0) for m in xy] + ([np.array([xs.min(), ys.min()])] if len(xs) else [])
    hi = [m.max(0) for m in xy] + ([np.array([xs.max(), ys.max()])] if len(xs) else [])
    if lo:
        x0 = max(0, int(math.floor(min(p[0] for p in lo))) - CROP_MARGIN_PX)
        y0 = max(0, int(math.floor(min(p[1] for p in lo))) - CROP_MARGIN_PX)
        x1 = min(w, int(math.ceil(max(p[0] for p in hi))) + 1 + CROP_MARGIN_PX)
        y1 = min(h, int(math.ceil(max(p[1] for p in hi))) + 1 + CROP_MARGIN_PX)
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    off = np.array([x0, y0], np.float64)
    crop = np.ascontiguousarray(gray[y0:y1, x0:x1], dtype=np.uint8)

    pieces = (
        (np.concatenate([s for s, _ in strokes]) - off).astype(np.float32)
        if n
        else np.zeros((0, 4, 2), np.float32)
    )
    offsets = np.zeros(n + 1, np.int64)
    if n:
        offsets[1:] = np.cumsum([len(s) for s, _ in strokes])
    widths = np.array([wd for _, wd in strokes], np.float32)
    bboxes = np.array([[*(m.min(0) - off), *(m.max(0) - off)] for m in xy], np.float32).reshape(
        n, 4
    )
    if len(xs) > _INK_SAMPLES:
        pick = np.random.default_rng(seed).choice(len(xs), _INK_SAMPLES, replace=False)
        xs, ys = xs[pick], ys[pick]
    ink_pts = np.stack([xs - x0, ys - y0], axis=1).astype(np.int32)
    ok, png = cv2.imencode(".png", crop, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    if not ok:
        raise RuntimeError(f"PNG encode failed for {key}")
    png.tofile(str(layers_dir / f"{key}.gray.png"))
    np.savez_compressed(
        layers_dir / f"{key}.strokes.npz",
        pieces=pieces,
        offsets=offsets,
        widths=widths,
        bboxes=bboxes,
        ink_pts=ink_pts,
        color=np.asarray(color, np.uint8),
        origin=np.array([x0, y0], np.int64),
        page_shape=np.array([h, w], np.int64),
    )
    return {"offset": [x0, y0], "shape": list(crop.shape)}


def load_index(data_dir: Path) -> dict:
    return json.loads((Path(data_dir) / "index.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Crops
# ---------------------------------------------------------------------------


def _cut_window(gray: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
    out = np.full((size, size), 255, np.uint8)
    h, w = gray.shape
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(w, x0 + size), min(h, y0 + size)
    if xb > xa and yb > ya:
        out[ya - y0 : yb - y0, xa - x0 : xb - x0] = gray[ya:yb, xa:xb]
    return out


def _strokes_in(layer: LayerData, x0: float, y0: float, size: int, max_prims: int):
    rect = (x0, y0, x0 + size, y0 + size)
    b = layer.bboxes
    if len(b) == 0:
        return [], []
    hit = np.nonzero(
        (b[:, 2] >= rect[0]) & (b[:, 0] <= rect[2]) & (b[:, 3] >= rect[1]) & (b[:, 1] <= rect[3])
    )[0]
    strokes, widths = [], []
    off = np.array([x0, y0], float)
    for i in hit:
        for part in geo.clip_stroke(layer.strokes[i] - off, (0, 0, size, size)):
            for sub in geo.split_long(part, max_prims):
                strokes.append(sub)
                widths.append(float(layer.widths[i]))
    return strokes, widths


def _sorted_pairs(strokes, widths):
    pairs = [(geo.orient_stroke(s), w) for s, w in zip(strokes, widths)]
    order = sorted(
        range(len(pairs)),
        key=lambda i: (geo._lex_key(pairs[i][0][0, 0]), geo._lex_key(pairs[i][0][-1, 3])),
    )
    return [pairs[i][0] for i in order], [pairs[i][1] for i in order]


def sample_crop(
    layer: LayerData,
    size: int,
    rng: np.random.Generator,
    n_stroke: int,
    max_prims: int,
    ink_bias: float = 0.9,
) -> Crop:
    """A `size` x `size` crop (white-padded at image borders) with its GT
    strokes clipped, split to `max_prims`, oriented and sorted. Centred on a
    random ink pixel with probability `ink_bias`. More strokes than
    `n_stroke` -> the window shrinks (32 px at a time, down to 64, kept
    centred and white-padded back to `size`); still too many -> truncated."""
    h, w = layer.gray.shape
    if len(layer.ink_pts) and rng.random() < ink_bias:
        cx, cy = layer.ink_pts[rng.integers(len(layer.ink_pts))]
        x0 = int(cx) - int(rng.integers(0, size))
        y0 = int(cy) - int(rng.integers(0, size))
    else:
        x0 = int(rng.integers(0, max(1, w - size + 1)))
        y0 = int(rng.integers(0, max(1, h - size + 1)))
    x0 = int(np.clip(x0, 0, max(0, w - size)))
    y0 = int(np.clip(y0, 0, max(0, h - size)))
    win = size
    strokes, widths = _strokes_in(layer, x0, y0, win, max_prims)
    while len(strokes) > n_stroke and win > SIZES[0]:
        win -= _SIZE_STEP
        strokes, widths = _strokes_in(layer, x0, y0, win, max_prims)
    gray = _cut_window(layer.gray, x0, y0, win)
    if win < size:
        full = np.full((size, size), 255, np.uint8)
        full[:win, :win] = gray
        gray = full
    n_found = len(strokes)
    strokes, widths = _sorted_pairs(strokes, widths)
    return Crop(gray, strokes[:n_stroke], widths[:n_stroke], window=(x0, y0, win), n_found=n_found)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def _rot90(crop: Crop) -> Crop:
    """`np.rot90` (counter-clockwise): (x, y) -> (y, S - x)."""
    s = crop.gray.shape[0]
    strokes = []
    for st in crop.strokes:
        t = st.copy()
        t[..., 0], t[..., 1] = st[..., 1], s - st[..., 0]
        strokes.append(t)
    return Crop(np.ascontiguousarray(np.rot90(crop.gray)), strokes, list(crop.widths))


def _flip(crop: Crop) -> Crop:
    """`np.fliplr`: (x, y) -> (S - x, y)."""
    s = crop.gray.shape[1]
    strokes = []
    for st in crop.strokes:
        t = st.copy()
        t[..., 0] = s - st[..., 0]
        strokes.append(t)
    return Crop(np.ascontiguousarray(np.fliplr(crop.gray)), strokes, list(crop.widths))


def augment(crop: Crop, rng: np.random.Generator) -> Crop:
    for _ in range(int(rng.integers(4))):
        crop = _rot90(crop)
    if rng.random() < 0.5:
        crop = _flip(crop)
    g = crop.gray
    if rng.random() < _AUG_THICKEN_P:  # thicker ink (min filter on gray = ink dilate)
        g = cv2.erode(g, np.ones((2, 2), np.uint8))
    if rng.random() < _AUG_BLUR_P:
        g = cv2.GaussianBlur(g, (0, 0), float(rng.uniform(*_AUG_BLUR_SIGMA)))
    if rng.random() < _AUG_NOISE_P:
        noise = rng.normal(0, rng.uniform(*_AUG_NOISE_STD), g.shape)
        g = np.clip(g.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < _AUG_JPEG_P:
        ok, enc = cv2.imencode(
            ".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(*_AUG_JPEG_QUALITY))]
        )
        if ok:
            g = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    strokes, widths = _sorted_pairs(crop.strokes, crop.widths)
    return Crop(g, strokes, widths)


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------
def stroke_mask(stroke: np.ndarray, width_px: float, size: int) -> np.ndarray:
    """The stroke rasterized at its width -- the Stroke Decoder's target."""
    canvas = np.zeros((size, size), np.uint8)
    pts = geo.sample_stroke(stroke, 16)
    fixed = np.round((pts - 0.5) * _SUBPIXEL).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        canvas, [fixed], False, 1, max(1, int(round(width_px))), cv2.LINE_8, shift=_SUBPIXEL_SHIFT
    )
    return canvas


def build_batch(
    crops: list[Crop],
    n_stroke: int,
    max_prims: int,
    with_raster: bool = True,
    id_offsets: list[int] | None = None,
) -> dict:
    """Same-size crops -> numpy arrays (all coordinates normalized by S):

    gray (B,S,S) u8; endpoints (B,N,4); valid (B,N);
    per GT stroke, flattened in (image, slot) order:
    stroke_img (M,), stroke_slot (M,), seq_in (M,T,6) = [curve_0, c_1..c_{T-1}],
    curves (M,T,6) targets, prim_mask (M,T), n_prims (M,), start (M,2),
    raster (M,S,S) u8, stroke_ids (M,) (bootstrap ids, -1 if none)."""
    b = len(crops)
    s = crops[0].gray.shape[0]
    t = max_prims
    gray = np.stack([c.gray for c in crops])
    endpoints = np.zeros((b, n_stroke, 4), np.float32)
    valid = np.zeros((b, n_stroke), bool)
    s_img, s_slot, seq_in, curves, pmask, nprims, start, raster, ids = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for i, c in enumerate(crops):
        for j, (st, wpx) in enumerate(zip(c.strokes[:n_stroke], c.widths[:n_stroke])):
            st = st[:t] / s
            k = len(st)
            endpoints[i, j] = [*st[0, 0], *st[-1, 3]]
            valid[i, j] = True
            cv = np.zeros((t, 6), np.float32)
            cv[:k] = st[:, 1:, :].reshape(k, 6)
            si = np.zeros((t, 6), np.float32)
            si[0] = np.tile(st[0, 0], 3)
            si[1:k] = cv[: k - 1]
            m = np.zeros(t, bool)
            m[:k] = True
            s_img.append(i)
            s_slot.append(j)
            seq_in.append(si)
            curves.append(cv)
            pmask.append(m)
            nprims.append(k)
            start.append(st[0, 0])
            if with_raster:
                raster.append(stroke_mask(st * s, wpx, s))
            ids.append(-1 if id_offsets is None else id_offsets[i] + j)
    m_ = len(s_img)
    return {
        "gray": gray,
        "endpoints": endpoints,
        "valid": valid,
        "stroke_img": np.array(s_img, np.int64),
        "stroke_slot": np.array(s_slot, np.int64),
        "seq_in": np.array(seq_in, np.float32).reshape(m_, t, 6),
        "curves": np.array(curves, np.float32).reshape(m_, t, 6),
        "prim_mask": np.array(pmask, bool).reshape(m_, t),
        "n_prims": np.array(nprims, np.int64),
        "start": np.array(start, np.float32).reshape(m_, 2),
        "raster": (np.stack(raster) if raster else np.zeros((0, s, s), np.uint8)),
        "stroke_ids": np.array(ids, np.int64),
        "size": s,
    }
