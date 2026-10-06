"""DeepVectoriser's Phase2Backend entrypoint. Every embedded `Image` Phase 1
hands over goes through, in order,

  1. color separation      DBSCAN over HSV-cone colors (`color_separation.py`)
  2. tiled OCR             960 px / 20 % tiles -> detect -> cross-tile merge ->
                           pad -> detect again -> pad -> deskew -> recognize
                           (`text_ocr.py`, own `paddle_engine.py` copy)
  3. text removal          each recognized text's ink layer erased to the
                           background color (`text_removal.py`)
  4. learned vectorizing   per ink color layer: the layer's binary mask
                           (ink 0, everything else 255 -- color independent,
                           so a pale line is as visible as a black one),
                           resampled to the canonical `TARGET_PX_PER_PT`,
                           tiled at `TILE_PX` and run through Liu et al.'s
                           (AAAI-22) stroke encoder + auto-regressive Bezier
                           vectorizer, tiles merged (`inference.py`,
                           `geometry.py`, `model/`)
  5. diff (debug only)     the strokes rasterized and compared back against
                           their own layer's ink (`diff.py`)

Steps 1-3 (and 5) are Junction's, *copied* (sibling P2 backends share no
code). Junction's CLAHE/unsharp enhancement is not: the model reads binary
layer masks, which enhancement can't change.

Output: one `Vector` per stroke -- `"c"` items, `"l"` where a piece is
flat (`FLAT_TOL_PX`), in its layer's color, width estimated from the ink's
distance transform -- plus OCR `Text`s, all in page space.

Coordinate mapping: canonical pixel -> native pixel (`/ scale`) -> unit
square -> the placement's own image matrix (`Image.transform`), so rotated/
flipped placements land correctly; an image without a transform falls back
to an upright fill of `Image.bbox`. Every map is affine, so Bezier control
points map exactly."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
from tqdm import tqdm

from rastervec.commons.helpers.geometry import compute_origin, transform_direction
from rastervec.commons.logging_setup import get_logger
from rastervec.commons.models import Image, Page, Text, Vector
from rastervec.commons.renderer.draw import (
    bbox_spec, dot_spec, draw_image, image_spec, new_page_doc, polyline_spec, quad_spec,
    render_specs_pdf,
)
from rastervec.P2_Raster_To_Vec.DeepVectoriser import diff as diff_mod
from rastervec.P2_Raster_To_Vec.DeepVectoriser import geometry as geo
from rastervec.P2_Raster_To_Vec.DeepVectoriser.color_separation import recolor, separate_colors
from rastervec.P2_Raster_To_Vec.DeepVectoriser.config import (
    DIFF_TOLERANCE_PX, FLAT_TOL_PX, INK_GRAY_THRESHOLD, TARGET_PX_PER_PT,
)
from rastervec.P2_Raster_To_Vec.DeepVectoriser.inference import vectorize_layer
from rastervec.P2_Raster_To_Vec.DeepVectoriser.text_ocr import run_ocr
from rastervec.P2_Raster_To_Vec.DeepVectoriser.text_removal import erase_text

DebugLayer = "tuple[str, str, str, bytes]"
OnDebugLayer = "Callable[[str, str, str, bytes], None]"
Mapper = Callable[[tuple[float, float]], tuple[float, float]]

_LOG = get_logger("P2.DeepVectoriser")


def extract(
    images: list[Image], page: Page, *, compute=None,
    debug_out: "dict | None" = None, on_debug_layer: "OnDebugLayer | None" = None,
    ocr_fns: "dict | None" = None, model_fn=None, keep_debug_arrays: bool = True,
) -> tuple[list[Vector], list[Text]]:
    """Runs the full flow (module docstring) on every image. `compute` is a
    Pool-2 proxy for PaddleOCR + model calls (forwarded by `core.pipeline`
    by signature). `ocr_fns` optionally overrides `text_ocr.run_ocr`'s
    `detect_many`/`recognize`/`recognize_raw`, and `model_fn` the model
    (`list[tile uint8] -> list[inference.TilePrediction]`) -- tests inject
    fakes. Without `model_fn`, missing weights raise `FileNotFoundError`."""
    state = _PageState()
    sink = _RasterSink(
        stream=on_debug_layer is not None,
        keep=debug_out is not None and keep_debug_arrays,
    )
    t_start = time.perf_counter()
    _LOG.info("DeepVectoriser: page %d, %d embedded image(s)", page.meta.index, len(images))
    for i, image in enumerate(tqdm(images, desc="DeepVectoriser images", unit="img",
                                   leave=False, disable=len(images) <= 1)):
        _LOG.info("DeepVectoriser image %d/%d", i + 1, len(images))
        _process_image(image, page, compute, state, sink, ocr_fns or {}, model_fn)

    if on_debug_layer is not None:
        t0 = time.perf_counter()
        layers_out = _page_layers(page.meta, sink.finish_stream(page.meta), state)
        for layer in layers_out:
            on_debug_layer(*layer)
        _LOG.info("DeepVectoriser debug layers: %d emitted (%.1fs)", len(layers_out),
                  time.perf_counter() - t0)
    if debug_out is not None:
        debug_out["raster_images"] = sink.kept
        debug_out["state"] = state
        debug_out["vectors"] = state.vectors
        debug_out["texts"] = state.texts
    _LOG.info("DeepVectoriser done: %d vectors, %d texts (%.1fs)", len(state.vectors), len(state.texts),
              time.perf_counter() - t_start)
    return state.vectors, state.texts


def render_debug(debug_out: "dict | None", page_meta) -> "list[DebugLayer]":
    """Batch counterpart of the `on_debug_layer` path -- identical layers,
    rebuilt from `debug_out`'s kept raw arrays."""
    if not debug_out:
        return []
    raster = _compose_raster_layers(page_meta, debug_out.get("raster_images") or [])
    return _page_layers(page_meta, raster, debug_out.get("state") or _PageState())


# ---------------------------------------------------------------------------
# Per-page accumulation
# ---------------------------------------------------------------------------
@dataclass
class _PageState:
    vectors: list[Vector] = field(default_factory=list)
    texts: list[Text] = field(default_factory=list)
    seqno: int = 0
    boxes: dict = field(default_factory=lambda: {
        "tile": [], "tile_detect": [], "merged": [], "refined_passed": [], "refined_failed": [],
        "vec_tile": [], "vec_resplit": [],
    })
    quads: dict = field(default_factory=lambda: {
        "tile_detect": [], "refined_passed": [], "refined_failed": [],
    })
    # page-space stroke geometry for the debug layers
    geom: dict = field(default_factory=lambda: {"raw": [], "raw_ends": [], "merged_ends": []})

    def next_seqno(self) -> int:
        self.seqno += 1
        return self.seqno - 1


@dataclass
class _RasterItem:
    stage: str
    label: str
    hexcolor: str
    array: np.ndarray
    image: Image
    shape: tuple[int, int]


class _RasterSink:
    """Receives full-resolution debug rasters (copied from Junction).
    Streaming: PNG-encode into one open single-page PDF per (stage, label)
    immediately, drop the array. Keep: retain `_RasterItem`s for
    `debug_out`. Neither: ignore (`active` tells the caller not to build
    the array at all)."""

    def __init__(self, *, stream: bool, keep: bool) -> None:
        self.stream = stream
        self.keep = keep
        self.kept: list[_RasterItem] = []
        self._docs: dict[tuple[str, str], tuple[str, object]] = {}

    @property
    def active(self) -> bool:
        return self.stream or self.keep

    def add(self, stage: str, label: str, hexcolor: str, array: np.ndarray, image: Image, page_meta) -> None:
        if self.keep:
            self.kept.append(_RasterItem(stage, label, hexcolor, array, image, array.shape[:2]))
        if self.stream:
            key = (stage, label)
            if key not in self._docs:
                self._docs[key] = (hexcolor, new_page_doc(page_meta))
            _place_raster(self._docs[key][1][1], array, image)

    def finish_stream(self, page_meta) -> "list[DebugLayer]":
        out: "list[DebugLayer]" = []
        for (stage, label), (hexcolor, doc) in self._docs.items():
            out.append((stage, label, hexcolor, _finish_doc(doc, page_meta)))
        self._docs.clear()
        return out


def _compose_raster_layers(page_meta, items: list[_RasterItem]) -> "list[DebugLayer]":
    docs: dict[tuple[str, str], tuple[str, object]] = {}
    for it in items:
        key = (it.stage, it.label)
        if key not in docs:
            docs[key] = (it.hexcolor, new_page_doc(page_meta))
        _place_raster(docs[key][1][1], it.array, it.image)
    return [(s, l, hexcolor, _finish_doc(doc, page_meta)) for (s, l), (hexcolor, doc) in docs.items()]


# ---------------------------------------------------------------------------
# Per-image flow
# ---------------------------------------------------------------------------
def _process_image(image: Image, page: Page, compute, state: _PageState, sink: _RasterSink,
                   ocr_fns: dict, model_fn) -> None:
    rgb = _as_rgb_copy(image.array)
    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        return
    meta = page.meta
    to_page = _make_to_page(image, w, h)
    px_per_pt = _px_per_pt(image, w, h)
    scale = TARGET_PX_PER_PT / px_per_pt  # native px -> canonical px
    _LOG.info("  %dx%d px, %.2f px/pt (x%.3f to canonical)", w, h, px_per_pt, scale)

    # 1. color separation
    t0 = time.perf_counter()
    layers = separate_colors(rgb)
    _LOG.info("  color separation: %d layer(s) (%.1fs)", layers.n_layers, time.perf_counter() - t0)
    if layers.n_layers == 0:
        return
    if sink.active:
        sink.add("color_separation", "clusters", _C_CLUSTERS, recolor(layers), image, meta)
    bg_rgb = tuple(int(c) for c in layers.centroids_rgb[layers.background])

    # 2. OCR (BGR view -- PaddleOCR is cv2/BGR)
    t0 = time.perf_counter()
    ocr = run_ocr(rgb[:, :, ::-1], px_per_pt, bg_rgb[::-1], compute=compute, **ocr_fns)
    _LOG.info("  OCR: %d tiles, %d hits (%.1fs)", len(ocr.tiles), len(ocr.hits), time.perf_counter() - t0)

    # 3. text removal + Text output
    t0 = time.perf_counter()
    inks = erase_text(rgb, layers, ocr.hits)
    n_texts = len(state.texts)
    _collect_ocr(state, ocr, inks, layers, to_page, meta.index)
    if sink.active:
        sink.add("text_removal", "cleaned image", _C_CLEANED, rgb.copy(), image, meta)
    _LOG.info("  text removal: %d text(s) erased (%.1fs)", len(state.texts) - n_texts,
              time.perf_counter() - t0)

    del rgb

    # 4. per-layer learned vectorizing (+ 5. diff when debugging)
    def canon_to_page(pt) -> tuple[float, float]:
        return to_page((float(pt[0]) / scale, float(pt[1]) / scale))

    total_codes = np.zeros((h, w), dtype=np.uint8) if sink.active else None
    for layer in layers.ink_layers():
        color_rgb = layers.centroids_rgb[layer]
        color = tuple(float(c) / 255.0 for c in color_rgb)
        hexcolor = _rgb_hex(color_rgb)
        t0 = time.perf_counter()
        layer_gray = layer_image(layers.labels, layer, scale)
        res = vectorize_layer(layer_gray, model_fn=model_fn, compute=compute,
                              desc=f"DeepVectoriser {hexcolor}")
        dist = cv2.distanceTransform((layer_gray < INK_GRAY_THRESHOLD).astype(np.uint8), cv2.DIST_L2, 3)
        for stroke in res.strokes:
            width_pt = stroke_width_px(stroke, dist) / TARGET_PX_PER_PT
            state.vectors.append(stroke_to_vector(
                stroke, canon_to_page, meta.index, state.next_seqno(), color, width_pt,
            ))
        _accumulate_debug(state, res, canon_to_page)
        _LOG.info("  vectorize layer %s: %d tile(s) (%d re-split) -> %d stroke(s) (%.1fs)", hexcolor,
                  len(res.tiles), len(res.resplit), len(res.strokes), time.perf_counter() - t0)
        if sink.active:
            polylines = [_DiffPolyline(geo.sample_stroke(s, 8), 1.0) for s in res.strokes]
            codes, _ = diff_mod.diff_codes(layer_gray < INK_GRAY_THRESHOLD, polylines, DIFF_TOLERANCE_PX)
            codes = codes[:layer_gray.shape[0], :layer_gray.shape[1]]
            native = cv2.resize(codes, (w, h), interpolation=cv2.INTER_NEAREST)
            np.copyto(total_codes, native, where=native > 0)
    if sink.active:
        sink.add("vector_diff", "total", _C_DIFF, diff_mod.codes_to_rgba(total_codes), image, meta)


@dataclass
class _DiffPolyline:
    points: np.ndarray
    width: float


def layer_image(labels: np.ndarray, layer: int, scale: float) -> np.ndarray:
    """One color layer as a gray mask (ink 0, everything else 255),
    resampled to the canonical scale (area-averaged when shrinking, so thin
    lines stay as gray rather than vanishing). Shared by inference and
    `prep_dataset.py` so the model always sees the same kind of input."""
    gray = np.where(labels == layer, 0, 255).astype(np.uint8)
    if abs(scale - 1.0) < 0.02:
        return gray
    h, w = gray.shape
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(gray, size, interpolation=interp)


def stroke_width_px(stroke: np.ndarray, dist: np.ndarray) -> float:
    """2 x the median distance-to-background along the stroke (>= 1 px)."""
    pts = np.round(geo.sample_stroke(stroke, 4) - 0.5).astype(int)
    h, w = dist.shape
    ok = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    if not ok.any():
        return 1.0
    vals = dist[pts[ok, 1], pts[ok, 0]]
    vals = vals[vals > 0]
    return max(1.0, 2.0 * float(np.median(vals))) if len(vals) else 1.0


def stroke_to_vector(stroke: np.ndarray, to_page: Mapper, page_index: int, seqno: int, color,
                     width_pt: float) -> Vector:
    """One `Vector` per stroke: per Bezier piece an `"l"` (flat, judged in
    canonical px) or a `"c"` item, mapped to page space."""
    items = []
    pts: list[tuple[float, float]] = []
    for piece in stroke:
        p = [to_page(q) for q in piece]
        if geo.piece_is_flat(piece, FLAT_TOL_PX):
            items.append(("l", p[0], p[3]))
        else:
            items.append(("c", p[0], p[1], p[2], p[3]))
        pts.extend(p)
    return Vector(
        type="s", items=items, width=width_pt, rect=_bbox_of(pts),
        **_base_vector_kwargs(page_index, seqno, color),
    )


def _accumulate_debug(state: _PageState, res, canon_to_page: Mapper) -> None:
    for x0, y0, s in res.tiles:
        state.boxes["vec_tile"].append(_bbox_of([canon_to_page((x0, y0)), canon_to_page((x0 + s, y0 + s))]))
    for x0, y0, s in res.resplit:
        state.boxes["vec_resplit"].append(_bbox_of([canon_to_page((x0, y0)), canon_to_page((x0 + s, y0 + s))]))
    for s in res.raw:
        state.geom["raw"].append([canon_to_page(p) for p in geo.sample_stroke(s, 6)])
        state.geom["raw_ends"].extend([canon_to_page(s[0, 0]), canon_to_page(s[-1, 3])])
    for s in res.strokes:
        state.geom["merged_ends"].extend([canon_to_page(s[0, 0]), canon_to_page(s[-1, 3])])


def _collect_ocr(state: _PageState, ocr, inks, layers, to_page: Mapper, page_index: int) -> None:
    for x0, y0, x1, y1 in ocr.tiles:
        state.boxes["tile"].append(_bbox_of([to_page((x0, y0)), to_page((x1, y1))]))
    for tb in ocr.tile_boxes:
        x0, y0, x1, y1 = tb.bbox
        state.boxes["tile_detect"].append(_bbox_of([to_page((x0, y0)), to_page((x1, y1))]))
        if tb.quad is not None:
            state.quads["tile_detect"].append(_quad_to_page(tb.quad, to_page))
    for x0, y0, x1, y1 in ocr.merged:
        state.boxes["merged"].append(_bbox_of([to_page((x0, y0)), to_page((x1, y1))]))
    for hit, ink in zip(ocr.hits, inks):
        quad = _quad_to_page(hit.quad, to_page)
        bbox = _bbox_of(list(quad))
        if not hit.text:
            state.boxes["refined_failed"].append(bbox)
            state.quads["refined_failed"].append(quad)
            continue
        state.boxes["refined_passed"].append(bbox)
        state.quads["refined_passed"].append(quad)
        direction = _page_direction(transform_direction((1.0, 0.0), hit.angle_deg or 0.0), hit.quad, to_page)
        color = None
        if ink is not None:
            r, g, b = (int(c) for c in layers.centroids_rgb[ink])
            color = (r << 16) | (g << 8) | b
        state.texts.append(Text(
            text=hit.text, bbox=bbox, direction=direction, origin=compute_origin(bbox, direction),
            font="", font_size=0.0, color=color, flags=0, ascender=None, descender=None, wmode=0,
            block_no=0, line_no=0, word_no=0, page_index=page_index, seqno=state.next_seqno(),
            confidence=hit.confidence, source="ocr", orientation_source="ocr",
            quad_points=quad,
        ))


def _quad_to_page(quad, to_page: Mapper) -> tuple:
    return tuple(to_page((float(x), float(y))) for x, y in np.asarray(quad).reshape(-1, 2))


def _page_direction(d_px: tuple[float, float], quad: np.ndarray, to_page: Mapper) -> tuple[float, float]:
    cx, cy = (float(v) for v in np.asarray(quad).mean(axis=0))
    p0 = to_page((cx, cy))
    p1 = to_page((cx + d_px[0], cy + d_px[1]))
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    n = float(np.hypot(dx, dy))
    return (dx / n, dy / n) if n > 0 else (1.0, 0.0)


# ---------------------------------------------------------------------------
# Geometry helpers (copied from Junction)
# ---------------------------------------------------------------------------
def _as_rgb_copy(array) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2).astype(np.uint8)
    return np.array(arr[:, :, :3], dtype=np.uint8, copy=True)


def _make_to_page(image: Image, w: int, h: int) -> Mapper:
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


def _px_per_pt(image: Image, w: int, h: int) -> float:
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


def _base_vector_kwargs(page_index: int, seqno: int, color) -> dict:
    return dict(
        color=color, fill=None, dashes=None, closePath=False,
        lineCap=0, lineJoin=0, even_odd=False, stroke_opacity=1.0, fill_opacity=1.0,
        layer=None, scissor=None, blendmode=None, isolated=False, knockout=False,
        opacity=1.0, page_index=page_index, seqno=seqno,
    )


def _bbox_of(points) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


# ---------------------------------------------------------------------------
# Debug rendering -- this backend's own layers, built from `commons/renderer`
# primitives plus embedded-PNG raster pages.
# ---------------------------------------------------------------------------
_C_CLUSTERS = "#7c3aed"
_C_CLEANED = "#0f766e"
_C_DIFF = "#dc2626"
_C_TILE = "#94a3b8"
_C_TILE_DETECT = "#f59e0b"
_C_MERGED = "#ea580c"
_C_OCR_PASS = "#16a34a"
_C_OCR_FAIL = "#9333ea"
_C_VEC_TILE = "#2563eb"
_C_RESPLIT = "#e11d48"
_C_RAW = "#0d9488"
_C_ENDPOINT = "#ca8a04"
_C_FINAL = "#059669"


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


def _boxes_layer(page_meta, stage: str, label: str, hexcolor: str, boxes) -> DebugLayer:
    rgb = _hex_rgb(hexcolor)
    return (stage, label, hexcolor, render_specs_pdf(page_meta, [bbox_spec(b, rgb) for b in boxes]))


def _quads_layer(page_meta, stage: str, label: str, hexcolor: str, quads) -> DebugLayer:
    rgb = _hex_rgb(hexcolor)
    return (stage, label, hexcolor, render_specs_pdf(page_meta, [quad_spec(q, rgb) for q in quads]))


def _geometry_layer(page_meta, stage: str, label: str, hexcolor: str, *,
                    polylines=(), points=(), width: float = 0.6, radius: float = 0.8) -> DebugLayer:
    rgb = _hex_rgb(hexcolor)
    specs = [polyline_spec(pl, rgb, width=width) for pl in polylines if len(pl) >= 2]
    specs += [dot_spec(p, rgb, radius) for p in points]
    return (stage, label, hexcolor, render_specs_pdf(page_meta, specs))


def _page_layers(page_meta, raster_layers: "list[DebugLayer]", state: _PageState) -> "list[DebugLayer]":
    from rastervec.commons.renderer import render_text_pdf, render_vectors_pdf

    bx = state.boxes
    qd = state.quads
    geom = state.geom
    out: "list[DebugLayer]" = list(raster_layers)
    out += [
        _boxes_layer(page_meta, "ocr", "tiles", _C_TILE, bx["tile"]),
        _quads_layer(page_meta, "ocr", "tile detect bbox", _C_TILE_DETECT, qd["tile_detect"]),
        _boxes_layer(page_meta, "ocr", "merged bbox", _C_MERGED, bx["merged"]),
        _quads_layer(page_meta, "ocr", "passed bbox", _C_OCR_PASS, qd["refined_passed"]),
        _quads_layer(page_meta, "ocr", "failed bbox", _C_OCR_FAIL, qd["refined_failed"]),
        ("ocr", "passed text", _C_OCR_PASS, render_text_pdf(
            page_meta, state.texts, color_of=lambda _t: _hex_rgb(_C_OCR_PASS),
        )),
        _boxes_layer(page_meta, "tiles", "vectorizer tile grid", _C_VEC_TILE, bx["vec_tile"]),
        _boxes_layer(page_meta, "tiles", "re-split tiles", _C_RESPLIT, bx["vec_resplit"]),
        _geometry_layer(page_meta, "strokes", "raw tile strokes", _C_RAW, polylines=geom["raw"]),
        _geometry_layer(page_meta, "strokes", "raw endpoints", _C_ENDPOINT, points=geom["raw_ends"]),
        _geometry_layer(page_meta, "merge", "merged endpoints", _C_RESPLIT, points=geom["merged_ends"]),
        ("merge", "merged vectors", _C_FINAL, render_vectors_pdf(
            page_meta, state.vectors, color_of=lambda v: tuple(v.color or (0, 0, 0)),
        )),
    ]
    return out
