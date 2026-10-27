"""Vector -> raster ground truth restricted to what the page render actually
shows -- the raster geometry GT behind `raster_label.raster_geometry_for_page`
(benchmark `vector_to_raster`, `master_label.py`) and DeepVectoriser /
ImplicitSketchVec's `prep_dataset.py`.

Every `get_drawings()` path is decomposed into `"l"`/`"c"` pieces
(`label_schema.geometry_annotations_for_vector`), then filtered against a
render of the page:

1. render   page -> RGB at `dpi` (`get_pixmap(Matrix(dpi/72))`, the same render
            `rasterised.pdf` is built from); page -> px is
            `page.rotation_matrix * zoom`.
2. layers   the P2 raster prelude's colour split, minus OCR: CLAHE + unsharp on
            LAB L (`RasterText.enhance`), then `RasterText.color_layers.
            split_layers` -- the same layers inference sees. Paper = `PAPER`.
3. paint    a path whose stroke and fill would both read as paper under
            `split_layers`' own ink rule (L above its Otsu threshold and chroma
            <= `INK_CHROMA_MIN`), or are absent / fully transparent, is dropped
            outright -- a white table-cell fill whose edge sits next to a grid
            line would otherwise pass the pixel test below.
4. visible  each piece is sampled every `SAMPLE_STEP_PX`; a sample is visible
            when any non-paper pixel lies within `VISIBLE_DILATE_PX` (Chebyshev)
            of it. Each maximal visible run becomes its own piece (a line ->
            sub-line, a cubic -> its de Casteljau sub-curve, an untouched piece
            keeps its exact points), so a clipped line keeps only its visible
            span, a dashed line becomes one piece per dash, and the part of a
            stroke under a later opaque fill disappears. A run that is only part
            of its piece and shorter than `MIN_VISIBLE_PX` (a hidden line
            crossing other ink) is dropped; a fully visible piece is always kept,
            however short (glyph contours are made of tiny pieces).
5. layers   a run belongs to every colour layer present near at least
            `LAYER_MIN_FRAC` of its samples (a filled shape's outline sits on its
            stroke layer and its fill layer), else to its most frequent one.

Ink with no vector behind it -- font text, embedded images, shadings,
annotations -- has no GT here by construction.

Memory (A1 at 300 dpi, ~70 M px): the RGB render (210 MB), LAB (210 MB) and
`split_layers`' temporaries peak around 1.3 GB inside step 2; afterwards only
the `uint8` labels (70 MB) and one dilated ink mask (70 MB) stay alive, and
samples are processed in chunks of at most `SAMPLE_CHUNK` points.
"""
from __future__ import annotations

import gc
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import cv2
import numpy as np

from rastervec.Evaluation.Labelling.label_schema import GeometryAnnotation, geometry_annotations_for_vector
from rastervec.P1_Reading_Native.reader import Reader
from rastervec.P1_Reading_Native.vector_extract import extract_vectors
from rastervec.P2_Raster_To_Vec.RasterText.color_layers import PAPER, split_layers
from rastervec.P2_Raster_To_Vec.RasterText.config import INK_CHROMA_MIN
from rastervec.P2_Raster_To_Vec.RasterText.enhance import clahe, unsharp_mask

_LOG = logging.getLogger("rastervec.Labelling.visible_geometry")

DEFAULT_DPI = 300.0
SAMPLE_STEP_PX = 0.5
VISIBLE_DILATE_PX = 1
MIN_VISIBLE_PX = 3.0
LAYER_MIN_FRAC = 0.5
SAMPLE_CHUNK = 2_000_000
PAPER_LABEL = 255  # `VisibleGeometry.labels` value for paper


@dataclass
class VisibleGeometry:
    """`annotations` (page space, `source="auto"`, extraction order) with
    `layers_of[i]` = the colour layers annotation i lies on. `labels` is the
    render's layer map (`uint8`, `PAPER_LABEL` on paper; `uint16` past 254
    layers), `colors_rgb[k]` layer k's median original colour, `matrix` the
    page -> px transform, `shape` the render's (h, w)."""

    annotations: list[GeometryAnnotation]
    layers_of: list[tuple[int, ...]]
    labels: np.ndarray
    colors_rgb: np.ndarray
    matrix: object
    rotation: int
    dpi: float
    paper_rgb: tuple[int, int, int]
    stats: dict = field(default_factory=dict)

    @property
    def n_layers(self) -> int:
        return len(self.colors_rgb)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.labels.shape)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Render + layers
# ---------------------------------------------------------------------------
def render_page(page, dpi: float) -> np.ndarray:
    """RGB uint8 render, identical to `master_label._run_raster_step`'s."""
    import fitz

    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return np.array(arr[:, :, :3], dtype=np.uint8, copy=True)


def page_to_pixel_matrix(page, dpi: float):
    """Unrotated page space -> the render's pixel frame."""
    import fitz

    zoom = dpi / 72.0
    return page.rotation_matrix * fitz.Matrix(zoom, zoom)


def color_layers(rgb: np.ndarray):
    """`RasterText`'s layer split on `rgb` without the prelude's OCR: the LAB
    of `enhance.clahe_lab` (computed in place -- its enhanced RGB is never
    built) -> `split_layers`."""
    lab = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = unsharp_mask(clahe(lab[:, :, 0]))
    layers = split_layers(lab, rgb)
    del lab
    return layers


def _compact_labels(labels: np.ndarray, n_layers: int) -> np.ndarray:
    dtype = np.uint8 if n_layers < PAPER_LABEL else np.uint16
    paper = PAPER_LABEL if dtype == np.uint8 else np.iinfo(np.uint16).max
    out = labels.astype(dtype)
    out[labels == PAPER] = paper
    return out


def _paper_rgb(rgb: np.ndarray, paper_mask: np.ndarray) -> tuple[int, int, int]:
    idx = np.flatnonzero(paper_mask.ravel())
    if idx.size == 0:
        return (255, 255, 255)
    idx = idx[:: idx.size // 100_000 + 1]
    return tuple(int(c) for c in np.median(rgb.reshape(-1, 3)[idx], axis=0))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Paint filter
# ---------------------------------------------------------------------------
def _as_rgb01(color) -> "tuple[float, float, float] | None":
    if color is None:
        return None
    c = [float(x) for x in color]
    if len(c) == 1:
        return (c[0], c[0], c[0])
    if len(c) == 4:  # CMYK, should PyMuPDF ever hand one through
        k = c[3]
        return tuple((1 - x) * (1 - k) for x in c[:3])  # type: ignore[return-value]
    return (c[0], c[1], c[2])


def paint_reads_as_ink(color, otsu_l: "float | None") -> bool:
    """Whether a flat paint colour would be ink under `split_layers`' rule
    (8-bit LAB: L at/below the Otsu threshold, or chroma above
    `INK_CHROMA_MIN`) -- i.e. whether it shows against the paper."""
    rgb = _as_rgb01(color)
    if rgb is None:
        return False
    px = np.clip(np.round(np.array(rgb) * 255), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    L, a, b = (float(v) for v in cv2.cvtColor(px, cv2.COLOR_RGB2LAB)[0, 0])
    if float(np.hypot(a - 128.0, b - 128.0)) > INK_CHROMA_MIN:
        return True
    return otsu_l is not None and L <= otsu_l


def vector_paint_visible(v, otsu_l: "float | None") -> bool:
    """A stroke or fill that reads as ink and isn't fully transparent."""
    if v.opacity is not None and v.opacity <= 0:
        return False
    stroke = ("s" in v.type and v.color is not None and v.stroke_opacity != 0
              and paint_reads_as_ink(v.color, otsu_l))
    fill = ("f" in v.type and v.fill is not None and v.fill_opacity != 0
            and paint_reads_as_ink(v.fill, otsu_l))
    return bool(stroke or fill)


# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------
def _pieces_px(annotations: list[GeometryAnnotation], matrix) -> np.ndarray:
    """(N, 4, 2) cubic control points in px (`"l"` -> controls at 1/3, 2/3);
    points mapped in float32 like MuPDF's `fz_transform_point`."""
    raw = np.zeros((len(annotations), 4, 2), np.float32)
    is_line = np.zeros(len(annotations), bool)
    for i, a in enumerate(annotations):
        if a.kind == "l":
            raw[i, 0], raw[i, 3] = a.points[0], a.points[-1]
            is_line[i] = True
        else:
            raw[i] = a.points
    m = np.array([matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f], np.float32)
    x, y = raw[..., 0], raw[..., 1]
    px = np.stack([x * m[0] + y * m[2] + m[4], x * m[1] + y * m[3] + m[5]], axis=-1).astype(np.float64)
    a, b = px[is_line, 0], px[is_line, 3]
    px[is_line, 1] = a + (b - a) / 3.0
    px[is_line, 2] = a + (b - a) * 2.0 / 3.0
    return px


def _bezier(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Points of cubics `p` (S, 4, 2) at `t` (S,)."""
    t = t[:, None]
    mt = 1.0 - t
    return mt ** 3 * p[:, 0] + 3 * mt ** 2 * t * p[:, 1] + 3 * mt * t ** 2 * p[:, 2] + t ** 3 * p[:, 3]


def _layer_lookup(labels: np.ndarray, x: np.ndarray, y: np.ndarray, paper: int, r: int) -> np.ndarray:
    """(S, (2r+1)^2) labels around each in-image sample (paper off-image)."""
    h, w = labels.shape
    cols = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            xx, yy = x + dx, y + dy
            ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
            col = np.full(len(x), paper, labels.dtype)
            col[ok] = labels[yy[ok], xx[ok]]
            cols.append(col)
    return np.stack(cols, axis=1)


@dataclass
class Runs:
    """Visible runs, in piece order then t: `piece` index, `[t0, t1]` on it,
    `full` (the whole piece, t0 = 0 / t1 = 1 exactly), and `layers[i]` =
    `layer_sets[layer_idx[i]]`."""

    piece: np.ndarray
    t0: np.ndarray
    t1: np.ndarray
    full: np.ndarray
    layer_idx: np.ndarray
    layer_sets: list[tuple[int, ...]]

    def __len__(self) -> int:
        return len(self.piece)


def visible_runs(pieces: np.ndarray, labels: np.ndarray, n_layers: int, paper: int, *,
                 step: float = SAMPLE_STEP_PX, radius: int = VISIBLE_DILATE_PX,
                 min_px: float = MIN_VISIBLE_PX, layer_min_frac: float = LAYER_MIN_FRAC,
                 chunk: int = SAMPLE_CHUNK) -> tuple[Runs, dict]:
    """The visible runs of every cubic piece (px); see the module docstring,
    steps 4-5. Fully vectorized per chunk of <= `chunk` samples (whole
    pieces per chunk, so no run spans two chunks)."""
    stats = {"pieces": len(pieces), "invisible_pieces": 0, "cut_pieces": 0, "short_runs": 0}
    parts: list[tuple] = []
    if len(pieces):
        poly_len = np.linalg.norm(np.diff(pieces, axis=1), axis=2).sum(axis=1)   # control polygon >= arc
        n_samp = np.maximum(2, np.ceil(poly_len / step).astype(np.int64) + 1)
        ink = np.ascontiguousarray(labels != paper).view(np.uint8)
        if radius > 0:
            ink = cv2.dilate(ink, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
        bounds = np.searchsorted(np.cumsum(n_samp), np.arange(chunk, int(n_samp.sum()), chunk), side="right")
        edges = [0, *sorted(set(int(e) for e in bounds if 0 < e < len(pieces))), len(pieces)]
        for start, stop in zip(edges, edges[1:]):
            part = _chunk_runs(pieces[start:stop], n_samp[start:stop], poly_len[start:stop], labels, ink,
                               n_layers, paper, radius, min_px, layer_min_frac, stats)
            if part is not None:
                parts.append((part[0] + start, *part[1:]))
    if not parts:
        e = np.zeros(0)
        return Runs(e.astype(np.int64), e, e, e.astype(bool), e.astype(np.int64), []), stats
    piece, t0, t1, full, mask = (np.concatenate(x) for x in zip(*parts))
    sets, layer_idx = np.unique(mask, axis=0, return_inverse=True)
    layer_sets = [tuple(int(j) for j in np.flatnonzero(row)) for row in sets]
    return Runs(piece, t0, t1, full, layer_idx.reshape(-1), layer_sets), stats


def _chunk_runs(pieces, n_samp, poly_len, labels, ink, n_layers, paper, radius, min_px,
                layer_min_frac, stats):
    """`(piece, t0, t1, full, layer mask (R, n_layers))` of one chunk, or None."""
    h, w = labels.shape
    total = int(n_samp.sum())
    pid = np.repeat(np.arange(len(pieces)), n_samp)
    first = np.cumsum(n_samp) - n_samp
    k = np.arange(total) - first[pid]
    last = n_samp[pid] - 1
    t = k / last
    pts = np.floor(_bezier(pieces[pid], t)).astype(np.int64)
    x, y = pts[:, 0], pts[:, 1]
    del pts
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    vis = np.zeros(total, bool)
    vis[inside] = ink[y[inside], x[inside]] > 0

    # maximal runs of visible samples within one piece
    is_start = vis & ~(np.concatenate([[False], vis[:-1]]) & (k > 0))
    is_end = vis & ~(np.concatenate([vis[1:], [False]]) & (k < last))
    starts, ends = np.flatnonzero(is_start), np.flatnonzero(is_end)
    seen = np.zeros(len(pieces), bool)
    seen[pid[starts]] = True
    stats["invisible_pieces"] += int((~seen).sum())
    n_runs = len(starts)
    if not n_runs:
        return None

    # run id of every sample inside a run
    run_of = np.cumsum(is_start) - 1
    sel = np.flatnonzero(vis)
    # layer vote: fraction of each run's samples with layer j within `radius`
    near = _layer_lookup(labels, x[sel], y[sel], paper, radius)
    rid = run_of[sel]
    n_in = np.bincount(rid, minlength=n_runs).astype(np.float64)
    frac = np.zeros((n_runs, max(n_layers, 1)))
    for j in range(n_layers):
        frac[:, j] = np.bincount(rid, weights=(near == j).any(axis=1), minlength=n_runs) / n_in
    del near
    mask = frac[:, :n_layers] >= layer_min_frac
    orphan = ~mask.any(axis=1)
    if n_layers and orphan.any():
        mask[np.flatnonzero(orphan), np.argmax(frac[orphan, :n_layers], axis=1)] = True

    rp = pid[starts]
    k0, k1, kl = k[starts], k[ends], n_samp[rp] - 1
    full = (k0 == 0) & (k1 == kl)
    dt = 1.0 / kl
    # the true edge lies between the last visible and first hidden sample
    t0 = np.where(k0 == 0, 0.0, np.maximum(0.0, t[starts] - dt / 2))
    t1 = np.where(k1 == kl, 1.0, np.minimum(1.0, t[ends] + dt / 2))
    short = ~full & ((t1 - t0) * poly_len[rp] < min_px)
    stats["short_runs"] += int(short.sum())
    cut = np.zeros(len(pieces), bool)
    cut[rp[~full]] = True
    stats["cut_pieces"] += int(cut.sum())
    keep = ~short
    return rp[keep], t0[keep], t1[keep], full[keep], mask[keep]


def _sub_points(kind: str, pts: list, t0: float, t1: float) -> list[tuple[float, float]]:
    """The [t0, t1] part of a line / cubic (page space; the page -> px map is
    affine, so px parameters carry over)."""
    p = np.asarray(pts, np.float64)
    if kind == "l":
        a, b = p[0], p[-1]
        return [tuple(a + (b - a) * t0), tuple(a + (b - a) * t1)]

    def split(c, t):  # de Casteljau -> (left, right)
        p01, p12, p23 = c[0] + (c[1] - c[0]) * t, c[1] + (c[2] - c[1]) * t, c[2] + (c[3] - c[2]) * t
        p012, p123 = p01 + (p12 - p01) * t, p12 + (p23 - p12) * t
        mid = p012 + (p123 - p012) * t
        return np.array([c[0], p01, p012, mid]), np.array([mid, p123, p23, c[3]])

    c = p
    if t1 < 1.0:
        c = split(c, t1)[0]
    if t0 > 0.0:
        c = split(c, t0 / t1)[1]
    return [tuple(q) for q in c]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
@contextmanager
def _gc_paused():
    """Pause the cyclic GC: a dense sheet allocates ~10^6 small, acyclic
    objects (drawing dicts, Vectors, annotations), and generation-2 passes
    over them cost ~40 % of building them."""
    was = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was:
            gc.enable()


def visible_geometry_for_page(pdf_path: str, page_index: int, dpi: float = DEFAULT_DPI) -> VisibleGeometry:
    """Steps 1-5 of the module docstring for one page. The image-heavy steps
    (render, colour split) finish and free their arrays before the
    object-heavy ones (vector extraction, annotations) start, so the two
    memory peaks never stack."""
    secs: dict[str, float] = {}
    with Reader(str(pdf_path)) as reader, _gc_paused():
        page = reader.get_page(page_index)
        t = time.perf_counter()
        rgb = render_page(page.fitz_page, dpi)
        matrix = page_to_pixel_matrix(page.fitz_page, dpi)
        rotation = int(page.fitz_page.rotation)
        secs["render"] = time.perf_counter() - t

        t = time.perf_counter()
        layers = color_layers(rgb)
        n_layers = layers.n_layers
        paper_rgb = _paper_rgb(rgb, layers.labels == PAPER)
        labels = _compact_labels(layers.labels, n_layers)
        colors_rgb, otsu_l = layers.colors_rgb, layers.otsu_l
        del rgb, layers
        paper = int(np.iinfo(labels.dtype).max) if labels.dtype != np.uint8 else PAPER_LABEL
        secs["layers"] = time.perf_counter() - t

        t = time.perf_counter()
        vectors = extract_vectors(page)
        n_vectors = len(vectors)
        secs["extract"] = time.perf_counter() - t

        t = time.perf_counter()
        kept = [v for v in vectors if vector_paint_visible(v, otsu_l)]
        n_kept = len(kept)
        del vectors
        anns = [a for v in kept for a in geometry_annotations_for_vector(v)]
        del kept
        anns = [a for a in anns if len(a.points) == (2 if a.kind == "l" else 4)]
        secs["paint"] = time.perf_counter() - t

        t = time.perf_counter()
        runs, vstats = visible_runs(_pieces_px(anns, matrix), labels, n_layers, paper)
        out_anns: list[GeometryAnnotation] = []
        for i, full, t0, t1 in zip(runs.piece.tolist(), runs.full.tolist(), runs.t0.tolist(), runs.t1.tolist()):
            a = anns[i]
            if not full:
                a = a.model_copy(update={"points": _sub_points(a.kind, a.points, t0, t1)})
            out_anns.append(a)
        layers_of = [runs.layer_sets[j] for j in runs.layer_idx.tolist()]
        del anns
        secs["visibility"] = time.perf_counter() - t

    stats = {
        "vectors": n_vectors, "paint_dropped_vectors": n_vectors - n_kept,
        "pieces": vstats["pieces"], "invisible_pieces": vstats["invisible_pieces"],
        "cut_pieces": vstats["cut_pieces"], "short_runs": vstats["short_runs"],
        "annotations": len(out_anns), "layers": n_layers,
        "seconds": {k: round(v, 3) for k, v in secs.items()},
    }
    _LOG.debug("visible_geometry_for_page(%s, %d): %s", pdf_path, page_index, stats)
    return VisibleGeometry(
        annotations=out_anns, layers_of=layers_of, labels=labels, colors_rgb=colors_rgb,
        matrix=matrix, rotation=rotation, dpi=float(dpi), paper_rgb=paper_rgb, stats=stats,
    )
