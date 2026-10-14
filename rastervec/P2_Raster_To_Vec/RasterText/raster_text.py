"""The shared P2 raster prelude -- the first call of every P2 backend's
`extract`. Per embedded `Image` Phase 1 hands over, in order:

  1. tiled OCR       on the *raw* image (PaddleOCR is tuned for natural
                     images): 960 px / 20 % tiles -> detect -> cross-tile
                     merge -> pad -> detect again -> pad -> deskew ->
                     recognize (`text_ocr.py`, `paddle_engine.py`)
  2. colour CLAHE    CLAHE + unsharp mask on LAB L only (`enhance.py`)
  3. colour layers   ink (L <= Otsu, or high chroma) -> one achromatic layer
                     + one layer per hue-histogram peak (`color_layers.py`)
  4. text removal    each recognized word's ink layer erased to paper
                     (`text_removal.py`)

`prepare_images` returns the page-space OCR `Text`s plus one
`PreparedImage` per image -- the enhanced, text-erased RGB, its colour
layers and the pixel -> page mapping -- which each backend then vectorizes
layer by layer (each layer's binary mask).

Debug: the prelude's own layers (`enhance/enhanced colour image`,
`ink/ink mask` + `chroma map`, `color_layers/all layers` + one
`color_layers/layer N #rrggbb` per layer, `text_removal/cleaned image` +
`erased pixels`, and the `ocr/*` box / quad / text layers) are streamed to
`on_debug_layer` at the end of `prepare_images`, or kept (`keep_debug`) and
rendered later by `prelude_debug_layers` -- the backends' batch
`render_debug` path. Per image, a hue/lightness histogram PNG
(`PreludeDebug.histograms`) shows the Otsu threshold, chroma cutoff, hue
peaks and each layer's hue arc."""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
from tqdm import tqdm

from rastervec.commons.helpers.geometry import compute_origin, transform_direction
from rastervec.commons.logging_setup import get_logger
from rastervec.commons.models import Image, Page, Text
from rastervec.commons.renderer.draw import (
    bbox_spec, draw_image, image_spec, new_page_doc, quad_spec, render_specs_pdf,
)
from rastervec.P2_Raster_To_Vec.RasterText.color_layers import (
    ColorLayers, chroma_of, recolor, split_layers,
)
from rastervec.P2_Raster_To_Vec.RasterText.config import ACHROMATIC_CHROMA_MAX, INK_CHROMA_MIN
from rastervec.P2_Raster_To_Vec.RasterText.enhance import clahe_lab
from rastervec.P2_Raster_To_Vec.RasterText.text_ocr import run_ocr
from rastervec.P2_Raster_To_Vec.RasterText.text_removal import erase_text

DebugLayer = "tuple[str, str, str, bytes]"
OnDebugLayer = "Callable[[str, str, str, bytes], None]"
OnDebugImage = "Callable[[str, str, Callable[[], object]], None]"
# `on_debug_image` folder key for the histogram PNGs
HISTOGRAM_FOLDER = "color_layer_histograms"
Mapper = Callable[[tuple[float, float]], tuple[float, float]]

_LOG = get_logger("P2.RasterText")


@dataclass
class PreparedImage:
    """One embedded image after the prelude: `enhanced_rgb` (CLAHE'd, text
    erased, HxWx3 uint8, native resolution) and its `layers` (`labels` -1 =
    paper), plus the pixel -> page map and pixels per page point."""

    image: Image
    to_page: Mapper
    px_per_pt: float
    enhanced_rgb: np.ndarray
    layers: ColorLayers


@dataclass
class _RasterItem:
    stage: str
    label: str
    hexcolor: str
    array: np.ndarray
    image: Image


@dataclass
class PreludeDebug:
    """Page-space OCR geometry, kept raster debug items (`keep_debug`), and
    one `(name, png bytes)` hue/lightness histogram per image."""

    boxes: dict = field(default_factory=lambda: {
        "tile": [], "tile_detect": [], "merged": [], "refined_passed": [], "refined_failed": [],
    })
    quads: dict = field(default_factory=lambda: {
        "tile_detect": [], "refined_passed": [], "refined_failed": [],
    })
    rasters: list[_RasterItem] = field(default_factory=list)
    histograms: list[tuple[str, bytes]] = field(default_factory=list)


@dataclass
class RasterPrelude:
    texts: list[Text] = field(default_factory=list)
    images: list[PreparedImage] = field(default_factory=list)
    debug: PreludeDebug = field(default_factory=PreludeDebug)


def prepare_images(
    images: list[Image], page: Page, *, compute=None, ocr_fns: "dict | None" = None,
    keep_debug: bool = False, on_debug_layer: "OnDebugLayer | None" = None,
    on_debug_image: "OnDebugImage | None" = None,
) -> RasterPrelude:
    """Steps 1-4 (module docstring) on every image. `compute` is a Pool-2
    proxy for PaddleOCR; `ocr_fns` optionally overrides `text_ocr.run_ocr`'s
    `detect_many`/`recognize`/`recognize_raw` (tests inject fakes).
    `keep_debug` keeps the full-resolution debug rasters for
    `prelude_debug_layers`; `on_debug_layer` receives the prelude's layers
    right away. With neither, no debug raster is built at all.
    `on_debug_image` (`core.pipeline`'s `(folder_key, name, make_image)`
    hook) is offered each image's histogram under `HISTOGRAM_FOLDER`. `Text`
    seqnos run 0..n-1 -- callers continue numbering after them."""
    meta = page.meta
    prelude = RasterPrelude()
    sink = _RasterSink(stream=on_debug_layer is not None, keep=keep_debug, page_meta=meta)
    t_start = time.perf_counter()
    for i, image in enumerate(tqdm(images, desc="Raster prelude", unit="img",
                                   leave=False, disable=len(images) <= 1)):
        prepared = _prepare_one(image, meta.index, i, compute, ocr_fns or {}, prelude, sink, on_debug_image)
        if prepared is not None:
            prelude.images.append(prepared)
    prelude.debug.rasters = sink.kept
    if on_debug_layer is not None:
        for layer in sink.finish_stream() + _vector_layers(meta, prelude):
            on_debug_layer(*layer)
    _LOG.info("Raster prelude: %d image(s), %d text(s) (%.1fs)", len(prelude.images), len(prelude.texts),
              time.perf_counter() - t_start)
    return prelude


def prelude_debug_layers(page_meta, prelude: "RasterPrelude | None") -> "list[DebugLayer]":
    """Batch counterpart of the streamed prelude layers -- identical layers,
    rebuilt from a `keep_debug=True` prelude."""
    if prelude is None:
        return []
    docs: dict[tuple[str, str], tuple[str, object]] = {}
    for it in prelude.debug.rasters:
        key = (it.stage, it.label)
        if key not in docs:
            docs[key] = (it.hexcolor, new_page_doc(page_meta))
        _place_raster(docs[key][1][1], it.array, it.image)
    raster = [(s, l, hexcolor, _finish_doc(doc, page_meta)) for (s, l), (hexcolor, doc) in docs.items()]
    return raster + _vector_layers(page_meta, prelude)


# ---------------------------------------------------------------------------
# Per-image flow
# ---------------------------------------------------------------------------
def _prepare_one(image: Image, page_index: int, image_index: int, compute, ocr_fns: dict,
                 prelude: RasterPrelude, sink: "_RasterSink",
                 on_debug_image: "OnDebugImage | None" = None) -> "PreparedImage | None":
    rgb = _as_rgb_copy(image.array)
    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        return None
    to_page = make_to_page(image, w, h)
    px_per_pt = image_px_per_pt(image, w, h)
    _LOG.info("Raster prelude image %d: %dx%d px, %.2f px/pt", image_index + 1, w, h, px_per_pt)

    # 1. OCR on the raw image (BGR view -- PaddleOCR is cv2/BGR); the deskew
    # border is filled with the image's median colour (its paper).
    t0 = time.perf_counter()
    bg_rgb = tuple(int(c) for c in _median_rgb(rgb))
    ocr = run_ocr(rgb[:, :, ::-1], px_per_pt, bg_rgb[::-1], compute=compute, **ocr_fns)
    _LOG.info("  OCR: %d tiles, %d tile boxes, %d merged, %d hits (%.1fs)", len(ocr.tiles),
              len(ocr.tile_boxes), len(ocr.merged), len(ocr.hits), time.perf_counter() - t0)

    # 2. colour CLAHE
    t0 = time.perf_counter()
    enhanced, lab = clahe_lab(rgb)
    if sink.active:
        sink.add("enhance", "enhanced colour image", _C_ENHANCED, enhanced.copy(), image)
    _LOG.info("  colour CLAHE (%.1fs)", time.perf_counter() - t0)

    # 3. colour layers
    t0 = time.perf_counter()
    layers = split_layers(lab, rgb)
    _LOG.info("  colour layers: %d %s, Otsu L %s (%.1fs)", layers.n_layers,
              [(_rgb_hex(c), k, n) for c, k, n in zip(layers.colors_rgb, layers.kinds, layers.pixel_counts)],
              layers.otsu_l, time.perf_counter() - t0)
    if sink.active:
        chroma = chroma_of(lab)
        sink.add("ink", "ink mask", _C_INK, _mask_rgba(layers.labels >= 0, (0, 0, 0)), image)
        sink.add("ink", "chroma map", _C_CHROMA, _chroma_rgb(chroma), image)
        sink.add("color_layers", "all layers", _C_LAYERS, recolor(layers), image)
        for k in layers.ink_layers():
            hexcolor = _rgb_hex(layers.colors_rgb[k])
            sink.add("color_layers", f"layer {k} {hexcolor}", hexcolor,
                     _mask_rgba(layers.labels == k, tuple(int(c) for c in layers.colors_rgb[k])), image)
        del chroma
    if sink.active or on_debug_image is not None:
        _offer_histogram(lab, layers, page_index, image_index, prelude, sink, on_debug_image)
    del lab

    # 4. text removal + Text output
    t0 = time.perf_counter()
    inks, erased = erase_text(enhanced, layers, ocr.hits)
    n_texts = len(prelude.texts)
    _collect_ocr(prelude, ocr, inks, layers, to_page, page_index)
    if sink.active:
        sink.add("text_removal", "cleaned image", _C_CLEANED, enhanced.copy(), image)
        sink.add("text_removal", "erased pixels", _C_ERASED, _mask_rgba(erased, _ERASED_RGB), image)
    _LOG.info("  text removal: %d text(s), %d px erased (%.1fs)", len(prelude.texts) - n_texts,
              int(erased.sum()), time.perf_counter() - t0)
    return PreparedImage(image=image, to_page=to_page, px_per_pt=px_per_pt,
                         enhanced_rgb=enhanced, layers=layers)


def _offer_histogram(lab, layers: ColorLayers, page_index: int, image_index: int,
                     prelude: RasterPrelude, sink: "_RasterSink", on_debug_image) -> None:
    """Render the histogram PNG at most once: always when debugging (kept on
    `prelude.debug.histograms`), else only if `on_debug_image` keeps it."""
    cache: list[bytes] = []

    def png() -> bytes:
        if not cache:
            cache.append(_histogram_png(lab, layers))
        return cache[0]

    if sink.active:
        prelude.debug.histograms.append((f"p{page_index}_img{image_index}.png", png()))
    if on_debug_image is not None:
        from PIL import Image as PILImage

        on_debug_image(HISTOGRAM_FOLDER, f"img{image_index}.png",
                       lambda: PILImage.open(io.BytesIO(png())))


def _collect_ocr(prelude: RasterPrelude, ocr, inks, layers: ColorLayers, to_page: Mapper,
                 page_index: int) -> None:
    bx, qd = prelude.debug.boxes, prelude.debug.quads
    for x0, y0, x1, y1 in ocr.tiles:
        bx["tile"].append(_bbox_of([to_page((x0, y0)), to_page((x1, y1))]))
    for tb in ocr.tile_boxes:
        x0, y0, x1, y1 = tb.bbox
        bx["tile_detect"].append(_bbox_of([to_page((x0, y0)), to_page((x1, y1))]))
        if tb.quad is not None:
            qd["tile_detect"].append(_quad_to_page(tb.quad, to_page))
    for x0, y0, x1, y1 in ocr.merged:
        bx["merged"].append(_bbox_of([to_page((x0, y0)), to_page((x1, y1))]))
    for hit, ink in zip(ocr.hits, inks):
        quad = _quad_to_page(hit.quad, to_page)
        bbox = _bbox_of(list(quad))
        if not hit.text:
            bx["refined_failed"].append(bbox)
            qd["refined_failed"].append(quad)
            continue
        bx["refined_passed"].append(bbox)
        qd["refined_passed"].append(quad)
        direction = _page_direction(transform_direction((1.0, 0.0), hit.angle_deg or 0.0), hit.quad, to_page)
        color = None
        if ink is not None:
            r, g, b = (int(c) for c in layers.colors_rgb[ink])
            color = (r << 16) | (g << 8) | b
        prelude.texts.append(Text(
            text=hit.text, bbox=bbox, direction=direction, origin=compute_origin(bbox, direction),
            font="", font_size=0.0, color=color, flags=0, ascender=None, descender=None, wmode=0,
            block_no=0, line_no=0, word_no=0, page_index=page_index, seqno=len(prelude.texts),
            confidence=hit.confidence, source="ocr", orientation_source="ocr",
            quad_points=quad,
        ))


# ---------------------------------------------------------------------------
# Geometry helpers (backends use `PreparedImage.to_page` / `.px_per_pt`)
# ---------------------------------------------------------------------------
def _as_rgb_copy(array) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2).astype(np.uint8)
    return np.array(arr[:, :, :3], dtype=np.uint8, copy=True)


def _median_rgb(rgb: np.ndarray) -> np.ndarray:
    flat = rgb.reshape(-1, 3)
    if len(flat) > 200_000:
        flat = flat[:: len(flat) // 200_000 + 1]
    return np.median(flat, axis=0)


def make_to_page(image: Image, w: int, h: int) -> Mapper:
    """Pixel `(px, py)` -> unit square `(px / w, py / h)` -> the placement's
    own image matrix (`Image.transform`), so rotated/flipped placements land
    correctly; without a transform, an upright fill of `Image.bbox`."""
    if image.transform is not None:
        a, b, c, d, e, f = image.transform

        def to_page(pt: tuple[float, float]) -> tuple[float, float]:
            fx, fy = pt[0] / w, pt[1] / h
            return (fx * a + fy * c + e, fx * b + fy * d + f)
        return to_page

    x0, y0, x1, y1 = image.bbox

    def to_page_bbox(pt: tuple[float, float]) -> tuple[float, float]:
        return (x0 + pt[0] / w * (x1 - x0), y0 + pt[1] / h * (y1 - y0))
    return to_page_bbox


def image_px_per_pt(image: Image, w: int, h: int) -> float:
    """Image pixels per page point (geometric mean over both image axes)."""
    if image.transform is not None:
        a, b, c, d, _e, _f = image.transform
        len_x, len_y = float(np.hypot(a, b)), float(np.hypot(c, d))
    else:
        x0, y0, x1, y1 = image.bbox
        len_x, len_y = x1 - x0, y1 - y0
    if len_x <= 0 or len_y <= 0:
        return max(image.dpi, 1.0) / 72.0
    return float(np.sqrt((w / len_x) * (h / len_y)))


def _quad_to_page(quad, to_page: Mapper) -> tuple:
    """A pixel-space quad's corners through the placement map, page space."""
    return tuple(to_page((float(x), float(y))) for x, y in np.asarray(quad).reshape(-1, 2))


def _page_direction(d_px: tuple[float, float], quad: np.ndarray, to_page: Mapper) -> tuple[float, float]:
    """A pixel-space direction through the placement's linear map, unit length."""
    cx, cy = (float(v) for v in np.asarray(quad).mean(axis=0))
    p0 = to_page((cx, cy))
    p1 = to_page((cx + d_px[0], cy + d_px[1]))
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    n = float(np.hypot(dx, dy))
    return (dx / n, dy / n) if n > 0 else (1.0, 0.0)


def _bbox_of(points) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


# ---------------------------------------------------------------------------
# Debug rendering
# ---------------------------------------------------------------------------
_C_ENHANCED = "#475569"
_C_INK = "#111827"
_C_CHROMA = "#c026d3"
_C_LAYERS = "#7c3aed"
_C_CLEANED = "#0f766e"
_C_ERASED = "#dc2626"
_ERASED_RGB = (220, 38, 38)
_C_TILE = "#94a3b8"
_C_TILE_DETECT = "#f59e0b"
_C_MERGED = "#ea580c"
_C_OCR_PASS = "#16a34a"
_C_OCR_FAIL = "#9333ea"
# chroma at which the heatmap saturates (8-bit LAB units)
_CHROMA_MAP_MAX = 60.0


class _RasterSink:
    """Full-resolution debug rasters. Streaming: drawn into one open
    single-page PDF per (stage, label) right away, the array dropped. Keep:
    retained for `prelude_debug_layers`. Neither: `active` is False and the
    caller builds nothing."""

    def __init__(self, *, stream: bool, keep: bool, page_meta) -> None:
        self.stream = stream
        self.keep = keep
        self.page_meta = page_meta
        self.kept: list[_RasterItem] = []
        self._docs: dict[tuple[str, str], tuple[str, object]] = {}

    @property
    def active(self) -> bool:
        return self.stream or self.keep

    def add(self, stage: str, label: str, hexcolor: str, array: np.ndarray, image: Image) -> None:
        if self.keep:
            self.kept.append(_RasterItem(stage, label, hexcolor, array, image))
        if self.stream:
            key = (stage, label)
            if key not in self._docs:
                self._docs[key] = (hexcolor, new_page_doc(self.page_meta))
            _place_raster(self._docs[key][1][1], array, image)

    def finish_stream(self) -> "list[DebugLayer]":
        out = [(s, l, hexcolor, _finish_doc(doc, self.page_meta))
               for (s, l), (hexcolor, doc) in self._docs.items()]
        self._docs.clear()
        return out


def _hex_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def _rgb_hex(rgb) -> str:
    return "#" + "".join(f"{int(c):02x}" for c in rgb[:3])


def _finish_doc(doc_page, page_meta) -> bytes:
    """Close out a `draw.new_page_doc` page: `/Rotate` last, so every raster
    was placed in unrotated page space."""
    doc, page = doc_page
    try:
        page.set_rotation(page_meta.rotation)
        return doc.tobytes(deflate=True)
    finally:
        doc.close()


def _place_raster(page, array: np.ndarray, image: Image) -> None:
    draw_image(page, image_spec(array, image.bbox, transform=image.transform))


def _mask_rgba(mask: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    out = np.empty(mask.shape + (4,), dtype=np.uint8)
    out[..., :3] = rgb
    np.multiply(mask, 255, out=out[..., 3], casting="unsafe")
    return out


def _chroma_rgb(chroma: np.ndarray) -> np.ndarray:
    scaled = np.clip(chroma * (255.0 / _CHROMA_MAP_MAX), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)


def _vector_layers(page_meta, prelude: RasterPrelude) -> "list[DebugLayer]":
    from rastervec.commons.renderer import render_text_pdf

    bx, qd = prelude.debug.boxes, prelude.debug.quads

    def boxes(label, hexcolor, items):
        rgb = _hex_rgb(hexcolor)
        return ("ocr", label, hexcolor, render_specs_pdf(page_meta, [bbox_spec(b, rgb) for b in items]))

    def quads(label, hexcolor, items):
        rgb = _hex_rgb(hexcolor)
        return ("ocr", label, hexcolor, render_specs_pdf(page_meta, [quad_spec(q, rgb) for q in items]))

    return [
        boxes("tiles", _C_TILE, bx["tile"]),
        quads("tile detect bbox", _C_TILE_DETECT, qd["tile_detect"]),
        boxes("merged bbox", _C_MERGED, bx["merged"]),
        quads("passed bbox", _C_OCR_PASS, qd["refined_passed"]),
        quads("failed bbox", _C_OCR_FAIL, qd["refined_failed"]),
        ("ocr", "passed text", _C_OCR_PASS, render_text_pdf(
            page_meta, prelude.texts, color_of=lambda _t: _hex_rgb(_C_OCR_PASS),
        )),
    ]


def _histogram_png(lab: np.ndarray, layers: ColorLayers) -> bytes:
    """Two panels: the L histogram with its Otsu threshold, and the
    chromatic ink's chroma-weighted hue histogram (raw + smoothed) with each
    hue layer's peak and arc in the layer's colour."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 6.5), dpi=100)
    ax_l, ax_h = fig.subplots(2, 1)
    # binned with numpy, drawn as one step polygon -- `hist`/`bar` draw one
    # artist per bin, seconds on a full page
    counts = np.bincount(lab[:, :, 0].ravel(), minlength=256)[:256]
    ax_l.stairs(counts, np.arange(257), color="#64748b", fill=True)
    ax_l.set_yscale("log")
    if layers.otsu_l is not None:
        ax_l.axvline(layers.otsu_l, color="#dc2626", label=f"Otsu L = {layers.otsu_l:.0f}")
        ax_l.legend(loc="upper left")
    else:
        ax_l.set_title("Otsu on L not used (classes too close)", fontsize=9)
    ax_l.set_xlabel("CLAHE'd L (8-bit LAB)")

    deg = np.linspace(0, 360, len(layers.hist), endpoint=False) if layers.hist is not None else np.zeros(0)
    if layers.hist is not None and layers.hist.any():
        ax_h.stairs(layers.hist, np.linspace(0, 360, len(layers.hist) + 1), color="#cbd5e1", fill=True,
                    label="chroma-weighted")
        ax_h.plot(deg, layers.hist_smooth, color="#0f172a", lw=1.2, label="smoothed")
        # log scale: one dominant colour would otherwise flatten every
        # smaller peak to the axis
        ax_h.set_yscale("log")
        top = float(max(layers.hist.max(), layers.hist_smooth.max()))
        ax_h.set_ylim(top * 1e-5, top * 2)
    for k in layers.ink_layers():
        color = tuple(int(c) / 255 for c in layers.colors_rgb[k])
        if layers.kinds[k] != "hue":
            ax_h.plot([], [], "s", color=color,
                      label=f"layer {k} achromatic, {layers.pixel_counts[k]} px")
            continue
        start, end = layers.hue_ranges[k]
        spans = [(start, end)] if start < end else [(start, 360.0), (0.0, end)]
        for a, b in spans:
            ax_h.axvspan(a, b, color=color, alpha=0.2)
        ax_h.axvline(layers.peaks_deg[k], color=color, lw=2,
                     label=f"layer {k} {_rgb_hex(layers.colors_rgb[k])}, {layers.pixel_counts[k]} px")
    ax_h.set_xlim(0, 360)
    ax_h.set_xlabel(f"LAB hue (deg) -- ink chroma > {INK_CHROMA_MIN:g}, "
                    f"achromatic <= {ACHROMATIC_CHROMA_MAX:g}")
    if layers.n_layers:
        ax_h.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()
