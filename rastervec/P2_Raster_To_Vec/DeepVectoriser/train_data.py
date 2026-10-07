"""Training-data helpers for `prep_dataset.py` and `train.py` (numpy only;
never imported by the pipeline itself).

Ground truth is the repo's own vector -> raster labelling method
(`scripts/label/master_label.py`'s raster geometry step): the page rendered
with `get_pixmap(Matrix(dpi/72))` (exactly how `rasterised.pdf` is built)
and `Evaluation.Labelling.raster_label.raster_geometry_for_page`'s
`GeometryAnnotation`s -- unrotated page space -- mapped into the render's
pixel frame through `page.rotation_matrix * Matrix(dpi/72)`.

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
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import geometry as geo

SIZES = tuple(range(64, 257, 32))  # the paper's 64..256 px, step 32
_INK_SAMPLES = 20000
CROP_MARGIN_PX = max(SIZES)  # stored context around a layer's content
PNG_COMPRESSION = 3          # cv2 PNG level: fast, ~15-100x smaller than raw


# ---------------------------------------------------------------------------
# Ground truth from a PDF page
# ---------------------------------------------------------------------------
def render_page(page, dpi: float) -> np.ndarray:
    """RGB uint8 render, identical to `master_label._run_raster_step`'s."""
    import fitz

    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return np.array(arr[:, :, :3], dtype=np.uint8, copy=True)  # writable (erase_text edits it)


def page_to_pixel_matrix(page, dpi: float):
    """Unrotated page space -> the render's pixel frame."""
    import fitz

    zoom = dpi / 72.0
    return page.rotation_matrix * fitz.Matrix(zoom, zoom)


def chain_annotations(annotations, matrix, tol_px: float = 0.05) -> list[tuple[np.ndarray, float]]:
    """`GeometryAnnotation`s (in their extraction order) -> strokes in px.

    Consecutive annotations with the same paint (color, fill, width) whose
    end meets the next start (within `tol_px`) are chained into one stroke
    -- a polyline / rectangle / glyph contour is one stroke (the paper's
    stroke unit). `"l"` becomes a cubic with controls at 1/3 and 2/3. A
    piece repeating earlier geometry (either direction) is dropped -- two GT
    strokes over the same pixels would be unlearnable.
    Returns `[(stroke (K,4,2), width_px), ...]`; width = annotation width
    (1 px minimum; fill-only edges have none).

    Vectorized: every piece is mapped and keyed up front; only the
    order-dependent chaining walks the annotations one by one."""
    scale = float(np.hypot(matrix.a, matrix.b))
    usable, pieces = _pieces_px(annotations, matrix)
    if not usable:
        return []
    keep = np.ptp(pieces, axis=1).max(axis=1) >= 1e-6       # drop degenerate (a point)
    keys = _piece_keys(pieces)
    starts, ends = pieces[:, 0], pieces[:, 3]

    out: list[tuple[np.ndarray, float]] = []
    cur: list[int] = []
    cur_key = None
    cur_w = 1.0
    seen: set = set()

    def flush():
        if cur:
            out.append((pieces[cur], cur_w))
            cur.clear()

    for n, i in enumerate(usable):
        if not keep[n]:
            continue
        if keys[n] in seen:
            continue  # same geometry again (e.g. a closed 2-point path retracing itself)
        seen.add(keys[n])
        ann = annotations[i]
        key = (tuple(ann.color) if ann.color else None, tuple(ann.fill) if ann.fill else None, ann.width)
        if cur and key == cur_key and math.hypot(*(ends[cur[-1]] - starts[n])) <= tol_px:
            cur.append(n)
            continue
        flush()
        cur.append(n)
        cur_key = key
        cur_w = max(1.0, float(ann.width or 0.0) * scale)
    flush()
    return out


def _pieces_px(annotations, matrix) -> tuple[list[int], np.ndarray]:
    """`(indices of usable annotations, their cubics (N,4,2) in px)`: an
    `"l"` needs 2 points, anything else 4. Points go through `matrix` in
    float32, exactly as MuPDF's `fz_transform_point` (`fitz.Point * Matrix`)
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
    raw = raw[:len(usable)].astype(np.float32)
    m = np.array([matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f], np.float32)
    x, y = raw[..., 0], raw[..., 1]
    px = np.stack([x * m[0] + y * m[2] + m[4], x * m[1] + y * m[3] + m[5]], axis=-1).astype(np.float64)
    is_line = np.array(lines, bool)
    a, b = px[is_line, 0], px[is_line, 1]
    px[is_line] = np.stack([a, a + (b - a) / 3.0, a + (b - a) * 2.0 / 3.0, b], axis=1)  # geo.line_to_cubic
    return usable, px


def _piece_keys(pieces: np.ndarray) -> list[bytes]:
    """Direction-independent identity of each piece (0.01 px grid);
    `+ 0.0` folds -0.0 into 0.0 so equal coordinates give equal bytes."""
    fwd = np.round(pieces, 2) + 0.0
    rev = np.ascontiguousarray(fwd[:, ::-1])
    fwd = np.ascontiguousarray(fwd)
    return [min(f.tobytes(), r.tobytes()) for f, r in zip(fwd, rev)]


def page_ground_truth(pdf_path: str, page_index: int, dpi: float):
    """`(rgb render, [(stroke px, width px)], page rotation)` for one page."""
    import fitz

    from rastervec.Evaluation.Labelling.raster_label import raster_geometry_for_page

    with fitz.open(pdf_path) as doc:
        page = doc[page_index]
        rgb = render_page(page, dpi)
        matrix = page_to_pixel_matrix(page, dpi)
        rotation = int(page.rotation)
    anns = raster_geometry_for_page(str(pdf_path), page_index)
    return rgb, chain_annotations(anns, matrix), rotation


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


def assign_layers(strokes, labels: np.ndarray, ink_layers: list[int], min_frac: float = 0.5) -> list[list[int]]:
    """Each stroke's color layers: every ink layer found in the 3x3
    neighbourhood of at least `min_frac` of its sample points. A stroke can
    belong to several layers -- e.g. a filled shape's outline is both its
    stroke layer's ink and its fill layer's boundary ("outline as strokes")
    -- or to none (an invisible / fully covered stroke).

    Vectorized: all strokes are sampled in one batch (the same points as
    `geo.sample_stroke(s, 6)`), and "layer in the 3x3 neighbourhood" is one
    lookup into that layer's 3x3-dilated mask."""
    if not strokes:
        return []
    h, w = labels.shape
    counts = np.array([len(s) for s, _ in strokes])
    pieces = np.concatenate([s for s, _ in strokes if len(s)]) if counts.any() else np.zeros((0, 4, 2))
    t = np.linspace(0.0, 1.0, 7)[:, None]
    mt = 1.0 - t
    p0, c1, c2, p3 = (pieces[:, None, k] for k in range(4))
    samples = mt ** 3 * p0 + 3 * mt ** 2 * t * c1 + 3 * mt * t ** 2 * c2 + t ** 3 * p3   # (P, 7, 2)
    first = np.zeros(len(pieces), bool)
    first[(np.cumsum(counts) - counts)[counts > 0]] = True
    take = np.ones(samples.shape[:2], bool)
    take[~first, 0] = False                         # shared joints counted once, as sample_stroke
    pts = np.floor(samples[take]).astype(np.int64)
    x = np.clip(pts[:, 0], 0, w - 1)
    y = np.clip(pts[:, 1], 0, h - 1)
    # An off-page coordinate clips every neighbour offset onto the edge, so
    # that axis contributes no neighbours: off in x -> vertical only, off in
    # y -> horizontal only, off in both -> the corner pixel itself.
    in_x = (pts[:, 0] >= 0) & (pts[:, 0] < w)
    in_y = (pts[:, 1] >= 0) & (pts[:, 1] < h)
    off_page = {(3, 1): ~in_x & in_y, (1, 3): in_x & ~in_y, (1, 1): ~in_x & ~in_y}
    n_pts = np.where(counts > 0, 6 * counts + 1, 0)
    starts = np.cumsum(n_pts) - n_pts
    nonempty = n_pts > 0
    frac: dict[int, np.ndarray] = {}
    for layer in ink_layers:
        mask = (labels == layer).view(np.uint8)
        near = cv2.dilate(mask, np.ones((3, 3), np.uint8))[y, x].astype(np.int64)
        for (kh, kw), sel in off_page.items():
            if sel.any():
                kmask = mask if (kh, kw) == (1, 1) else cv2.dilate(mask, np.ones((kh, kw), np.uint8))
                near[sel] = kmask[y[sel], x[sel]]
        hits = np.zeros(len(strokes))
        if len(near):
            hits[nonempty] = np.add.reduceat(near, starts[nonempty])
        frac[layer] = hits / np.maximum(n_pts, 1)
    return [[layer for layer in ink_layers if nonempty[i] and frac[layer][i] >= min_frac]
            for i in range(len(strokes))]


def scale_strokes(strokes, factor: float):
    if abs(factor - 1.0) < 1e-9:
        return strokes
    return [(s * factor, wpx * factor) for s, wpx in strokes]


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
    ys, xs = np.nonzero(gray < 200)
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

    pieces = (np.concatenate([s for s, _ in strokes]) - off).astype(np.float32) if n         else np.zeros((0, 4, 2), np.float32)
    offsets = np.zeros(n + 1, np.int64)
    if n:
        offsets[1:] = np.cumsum([len(s) for s, _ in strokes])
    widths = np.array([wd for _, wd in strokes], np.float32)
    bboxes = np.array([[*(m.min(0) - off), *(m.max(0) - off)] for m in xy], np.float32).reshape(n, 4)
    if len(xs) > _INK_SAMPLES:
        pick = np.random.default_rng(seed).choice(len(xs), _INK_SAMPLES, replace=False)
        xs, ys = xs[pick], ys[pick]
    ink_pts = np.stack([xs - x0, ys - y0], axis=1).astype(np.int32)
    ok, png = cv2.imencode(".png", crop, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    if not ok:
        raise RuntimeError(f"PNG encode failed for {key}")
    png.tofile(str(layers_dir / f"{key}.gray.png"))
    np.savez_compressed(layers_dir / f"{key}.strokes.npz", pieces=pieces, offsets=offsets, widths=widths,
                        bboxes=bboxes, ink_pts=ink_pts, color=np.asarray(color, np.uint8),
                        origin=np.array([x0, y0], np.int64), page_shape=np.array([h, w], np.int64))
    return {"offset": [x0, y0], "shape": list(crop.shape)}


@dataclass
class LayerData:
    key: str
    gray: np.ndarray
    strokes: list[np.ndarray]
    widths: np.ndarray
    bboxes: np.ndarray
    ink_pts: np.ndarray


def default_cache_dir(layers_dir: Path) -> Path:
    return Path(layers_dir).parent / "cache"


def gray_path(layers_dir: Path, key: str, cache_dir: "Path | None" = None) -> Path:
    """The memory-mappable `.npy` of a layer's mask: an old-format
    `layers/<key>.gray.npy` as is, else the PNG decoded once into
    `cache_dir` (default `<data>/cache/`; re-decoded when the PNG is newer).
    Written to a temp file and renamed, so an interrupted decode never
    leaves a half-written cache file."""
    layers_dir = Path(layers_dir)
    legacy = layers_dir / f"{key}.gray.npy"
    if legacy.is_file():
        return legacy
    png = layers_dir / f"{key}.gray.png"
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir(layers_dir)
    target = cache / f"{key}.gray.npy"
    if target.is_file() and target.stat().st_mtime >= png.stat().st_mtime:
        return target
    cache.mkdir(parents=True, exist_ok=True)
    gray = cv2.imdecode(np.fromfile(str(png), np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"could not decode {png}")
    tmp = target.with_name(target.name + f".{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, gray)
    os.replace(tmp, target)
    return target


def load_layer(layers_dir: Path, key: str, cache_dir: "Path | None" = None) -> LayerData:
    gray = np.load(gray_path(layers_dir, key, cache_dir), mmap_mode="r")
    z = np.load(Path(layers_dir) / f"{key}.strokes.npz")
    pieces, offsets = z["pieces"].astype(np.float64), z["offsets"]
    strokes = [pieces[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
    return LayerData(key, gray, strokes, z["widths"], z["bboxes"], z["ink_pts"])


def load_index(data_dir: Path) -> dict:
    return json.loads((Path(data_dir) / "index.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Crops
# ---------------------------------------------------------------------------
@dataclass
class Crop:
    gray: np.ndarray                                   # (S, S) uint8
    strokes: list[np.ndarray] = field(default_factory=list)  # sorted, oriented, local px
    widths: list[float] = field(default_factory=list)


def _cut_window(gray: np.ndarray, x0: int, y0: int, size: int) -> np.ndarray:
    out = np.full((size, size), 255, np.uint8)
    h, w = gray.shape
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(w, x0 + size), min(h, y0 + size)
    if xb > xa and yb > ya:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = gray[ya:yb, xa:xb]
    return out


def _strokes_in(layer: LayerData, x0: float, y0: float, size: int, max_prims: int):
    rect = (x0, y0, x0 + size, y0 + size)
    b = layer.bboxes
    if len(b) == 0:
        return [], []
    hit = np.nonzero((b[:, 2] >= rect[0]) & (b[:, 0] <= rect[2]) & (b[:, 3] >= rect[1]) & (b[:, 1] <= rect[3]))[0]
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
    order = sorted(range(len(pairs)), key=lambda i: (geo._lex_key(pairs[i][0][0, 0]), geo._lex_key(pairs[i][0][-1, 3])))
    return [pairs[i][0] for i in order], [pairs[i][1] for i in order]


def sample_crop(layer: LayerData, size: int, rng: np.random.Generator, n_stroke: int, max_prims: int,
                ink_bias: float = 0.9) -> Crop:
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
    while len(strokes) > n_stroke and win > 64:
        win -= 32
        strokes, widths = _strokes_in(layer, x0, y0, win, max_prims)
    gray = _cut_window(layer.gray, x0, y0, win)
    if win < size:
        full = np.full((size, size), 255, np.uint8)
        full[:win, :win] = gray
        gray = full
    strokes, widths = _sorted_pairs(strokes, widths)
    return Crop(gray, strokes[:n_stroke], widths[:n_stroke])


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
    if rng.random() < 0.15:  # thicker ink (min filter on gray = ink dilate)
        g = cv2.erode(g, np.ones((2, 2), np.uint8))
    if rng.random() < 0.3:
        g = cv2.GaussianBlur(g, (0, 0), float(rng.uniform(0.3, 0.8)))
    if rng.random() < 0.3:
        noise = rng.normal(0, rng.uniform(2, 10), g.shape)
        g = np.clip(g.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.2:
        ok, enc = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 95))])
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
    fixed = np.round((pts - 0.5) * 16).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [fixed], False, 1, max(1, int(round(width_px))), cv2.LINE_8, shift=4)
    return canvas


def build_batch(crops: list[Crop], n_stroke: int, max_prims: int, with_raster: bool = True,
                id_offsets: list[int] | None = None) -> dict:
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
    s_img, s_slot, seq_in, curves, pmask, nprims, start, raster, ids = [], [], [], [], [], [], [], [], []
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
            si[1:k] = cv[:k - 1]
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
        "gray": gray, "endpoints": endpoints, "valid": valid,
        "stroke_img": np.array(s_img, np.int64), "stroke_slot": np.array(s_slot, np.int64),
        "seq_in": np.array(seq_in, np.float32).reshape(m_, t, 6),
        "curves": np.array(curves, np.float32).reshape(m_, t, 6),
        "prim_mask": np.array(pmask, bool).reshape(m_, t),
        "n_prims": np.array(nprims, np.int64),
        "start": np.array(start, np.float32).reshape(m_, 2),
        "raster": (np.stack(raster) if raster else np.zeros((0, s, s), np.uint8)),
        "stroke_ids": np.array(ids, np.int64),
        "size": s,
    }
