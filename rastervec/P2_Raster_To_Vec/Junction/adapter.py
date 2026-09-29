"""Junction's Phase2Backend entrypoint: every embedded `Image` Phase 1 hands
over goes through, in order,

  1. color separation      DBSCAN over HSV-cone colors (`color_separation.py`)
  2. tiled OCR             960 px / 20 % tiles -> detect -> cross-tile merge ->
                           pad -> detect again -> pad -> deskew -> recognize
                           (`text_ocr.py`, own `paddle_engine.py` copy)
  3. text removal          each recognized text's ink layer erased to the
                           background color (`text_removal.py`)
  4. enhancement           CLAHE + unsharp mask (`enhance.py`)
  5. tracing               per ink color layer, per 10 pt-tolerance spatial
                           component, one at a time (`components.py`), the
                           classical `junction_test.pipeline.run` at native
                           resolution
  6. diff (debug only)     traced vectors rasterized and compared back
                           against their own ink (`diff.py`)

and comes back as `Vector`s (colored by their layer) + OCR `Text`s, all in
page space.

Coordinate mapping: pixel `(px, py)` of an `h`x`w` image -> unit square
`(px / w, py / h)` -> the placement's own image matrix (`Image.transform`,
from `get_image_info`), so rotated/flipped placements land correctly; an
image without a transform falls back to an upright fill of `Image.bbox`.

Debug output (`on_debug_layer` streaming / `debug_out` + `render_debug`
batch, both through the same helpers): raster stages (color clusters,
cleaned image, enhanced image, rendered vectors, total + per-layer diff)
become full-resolution PNGs embedded in page-sized PDFs -- each layer label
appears once per page even with several images, the images composed onto
it; OCR and tracing stages become box / text layers. On the streaming path
each raster debug image is PNG-encoded into its layer's page right away and
the array dropped, so no full-size debug array outlives its own step;
`debug_out` instead keeps the raw arrays (incl. the uint8 diff-code
canvases, `diff.MATCHED/MISSED/SPURIOUS`) for later missed-line work."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from rastervec.commons.helpers.geometry import compute_origin, transform_direction
from rastervec.commons.models import Image, Page, Text, Vector
from rastervec.P2_Raster_To_Vec.Junction import diff as diff_mod
from rastervec.P2_Raster_To_Vec.Junction.color_separation import recolor, separate_colors
from rastervec.P2_Raster_To_Vec.Junction.components import iter_layer_components
from rastervec.P2_Raster_To_Vec.Junction.config import COMPONENT_TOLERANCE_PT, DIFF_TOLERANCE_PX
from rastervec.P2_Raster_To_Vec.Junction.enhance import enhance, to_gray
from rastervec.P2_Raster_To_Vec.Junction.junction_test.pipeline import Params, run
from rastervec.P2_Raster_To_Vec.Junction.junction_test.types_ import Arc, Segment
from rastervec.P2_Raster_To_Vec.Junction.text_ocr import run_ocr
from rastervec.P2_Raster_To_Vec.Junction.text_removal import erase_text

DebugLayer = "tuple[str, str, str, bytes]"
OnDebugLayer = "Callable[[str, str, str, bytes], None]"
Mapper = Callable[[tuple[float, float]], tuple[float, float]]


def extract(
    images: list[Image], page: Page, *, params: Params | None = None, compute=None,
    debug_out: "dict | None" = None, on_debug_layer: "OnDebugLayer | None" = None,
    ocr_fns: "dict | None" = None,
) -> tuple[list[Vector], list[Text]]:
    """Runs the full flow (module docstring) on every image. `compute` is a
    Pool-2 proxy for PaddleOCR calls (forwarded by `core.pipeline` by
    signature). `ocr_fns` optionally overrides `text_ocr.run_ocr`'s
    `detect_many`/`recognize`/`recognize_raw` (tests inject fakes)."""
    params = params or Params()
    state = _PageState()
    sink = _RasterSink(
        stream=on_debug_layer is not None, keep=debug_out is not None,
    )
    for image in images:
        _process_image(image, page, params, compute, state, sink, ocr_fns or {})

    if on_debug_layer is not None:
        for layer in _page_layers(page.meta, sink.finish_stream(page.meta), state):
            on_debug_layer(*layer)
    if debug_out is not None:
        debug_out["raster_images"] = sink.kept
        # (label, image, uint8 codes HxW): "total" + one per layer hex --
        # the raw missed/spurious maps, see diff.py.
        debug_out["diff_codes"] = [
            (it.label, it.image, it.array) for it in sink.kept if it.stage == "_codes"
        ]
        debug_out["state"] = state
        debug_out["vectors"] = state.vectors
        debug_out["texts"] = state.texts
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
        "component": [], "chain": [], "junction": [], "segment": [], "arc": [],
    })

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
    """Receives full-resolution debug rasters. Streaming: PNG-encode into one
    open single-page PDF per (stage, label) immediately, drop the array.
    Keep: retain `_RasterItem`s for `debug_out`. Neither: ignore (and
    `active` tells the caller not to build the array at all)."""

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
                self._docs[key] = (hexcolor, _new_blank_doc(page_meta))
            _place_raster(self._docs[key][1][0], array, image, array.shape[:2])

    def finish_stream(self, page_meta) -> "list[DebugLayer]":
        out: "list[DebugLayer]" = []
        for (stage, label), (hexcolor, doc) in self._docs.items():
            out.append((stage, label, hexcolor, _finish_doc(doc, page_meta)))
        self._docs.clear()
        return out


def _compose_raster_layers(page_meta, items: list[_RasterItem]) -> "list[DebugLayer]":
    docs: dict[tuple[str, str], tuple[str, object]] = {}
    for it in items:
        if it.stage.startswith("_"):  # raw diff-code canvases -- data, not a layer
            continue
        key = (it.stage, it.label)
        if key not in docs:
            docs[key] = (it.hexcolor, _new_blank_doc(page_meta))
        _place_raster(docs[key][1][0], it.array, it.image, it.shape)
    return [(s, l, hexcolor, _finish_doc(doc, page_meta)) for (s, l), (hexcolor, doc) in docs.items()]


# ---------------------------------------------------------------------------
# Per-image flow
# ---------------------------------------------------------------------------
def _process_image(
    image: Image, page: Page, params: Params, compute, state: _PageState,
    sink: _RasterSink, ocr_fns: dict,
) -> None:
    rgb = _as_rgb_copy(image.array)
    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        return
    meta = page.meta
    to_page = _make_to_page(image, w, h)
    px_per_pt = _px_per_pt(image, w, h)

    # 1. color separation
    layers = separate_colors(rgb)
    if layers.n_layers == 0:
        return
    if sink.active:
        sink.add("color_separation", "clusters", _C_CLUSTERS, recolor(layers), image, meta)
    bg_rgb = tuple(int(c) for c in layers.centroids_rgb[layers.background])

    # 2. OCR (BGR view -- PaddleOCR is cv2/BGR)
    ocr = run_ocr(rgb[:, :, ::-1], px_per_pt, bg_rgb[::-1], compute=compute, **ocr_fns)

    # 3. text removal + Text output
    inks = erase_text(rgb, layers, ocr.hits)
    _collect_ocr(state, ocr, inks, layers, to_page, meta.index)
    if sink.active:
        sink.add("text_removal", "cleaned image", _C_CLEANED, rgb.copy(), image, meta)

    # 4. enhancement
    enhanced = enhance(to_gray(rgb))
    del rgb
    if sink.active:
        sink.add("enhance", "enhanced image", _C_ENHANCED, enhanced, image, meta)

    # 5. per-layer, per-component tracing (+ 6. diff when debugging)
    tol_px = COMPONENT_TOLERANCE_PT * px_per_pt
    rendered_canvas = np.zeros((h, w), dtype=bool) if sink.active else None
    total_codes = np.zeros((h, w), dtype=np.uint8) if sink.active else None
    for layer in layers.ink_layers():
        color_rgb = layers.centroids_rgb[layer]
        color = tuple(float(c) / 255.0 for c in color_rgb)
        layer_codes = np.zeros((h, w), dtype=np.uint8) if sink.active else None
        for comp in iter_layer_components(layers.labels, enhanced, layer, tol_px):
            result = run(comp.gray, params)
            rh, rw = result.gray.shape[:2]
            sx = comp.gray.shape[1] / rw if rw else 1.0
            sy = comp.gray.shape[0] / rh if rh else 1.0

            def mapper(pt, _x0=comp.x0, _y0=comp.y0, _sx=sx, _sy=sy):
                return to_page((pt[0] * _sx + _x0, pt[1] * _sy + _y0))

            for seg in result.segments:
                state.vectors.append(_segment_to_vector(seg, mapper, meta.index, state.next_seqno(), color))
            for arc in result.arcs:
                state.vectors.append(_arc_to_vector(arc, mapper, meta.index, state.next_seqno(), color))
            _accumulate_trace_boxes(state.boxes, result, mapper, (comp.x0, comp.y0), comp.gray.shape, to_page)

            if sink.active:
                codes, rendered = diff_mod.diff_codes(
                    result.ink, result.segments, result.arcs, DIFF_TOLERANCE_PX,
                )
                if (rh, rw) != comp.gray.shape[:2]:  # only with an opt-in max_work_px
                    import cv2

                    size = (comp.gray.shape[1], comp.gray.shape[0])
                    codes = cv2.resize(codes, size, interpolation=cv2.INTER_NEAREST)
                    rendered = cv2.resize(rendered.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
                diff_mod.paste_codes(layer_codes, codes, comp.x0, comp.y0)
                diff_mod.paste_codes(total_codes, codes, comp.x0, comp.y0)
                region = rendered_canvas[comp.y0:comp.y0 + rendered.shape[0], comp.x0:comp.x0 + rendered.shape[1]]
                region |= rendered[:region.shape[0], :region.shape[1]]
        if sink.active:
            hexcolor = _rgb_hex(color_rgb)
            sink.add("vector_diff", f"layer {hexcolor}", hexcolor,
                     diff_mod.codes_to_rgba(layer_codes), image, meta)
            if sink.keep:
                sink.kept.append(_RasterItem("_codes", hexcolor, hexcolor, layer_codes, image, (h, w)))
            del layer_codes
    if sink.active:
        sink.add("vector_render", "rendered vectors", _C_SEGMENT,
                 diff_mod.rendered_to_rgba(rendered_canvas), image, meta)
        sink.add("vector_diff", "total", _C_DIFF, diff_mod.codes_to_rgba(total_codes), image, meta)
        if sink.keep:
            sink.kept.append(_RasterItem("_codes", "total", _C_DIFF, total_codes, image, (h, w)))


def _collect_ocr(state: _PageState, ocr, inks, layers, to_page: Mapper, page_index: int) -> None:
    for x0, y0, x1, y1 in ocr.tiles:
        state.boxes["tile"].append(_bbox_of([to_page((x0, y0)), to_page((x1, y1))]))
    for tb in ocr.tile_boxes:
        x0, y0, x1, y1 = tb.bbox
        state.boxes["tile_detect"].append(_bbox_of([to_page((x0, y0)), to_page((x1, y1))]))
    for x0, y0, x1, y1 in ocr.merged:
        state.boxes["merged"].append(_bbox_of([to_page((x0, y0)), to_page((x1, y1))]))
    for hit, ink in zip(ocr.hits, inks):
        bbox = _bbox_of([to_page((float(x), float(y))) for x, y in hit.quad])
        if not hit.text:
            state.boxes["refined_failed"].append(bbox)
            continue
        state.boxes["refined_passed"].append(bbox)
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
        ))


def _page_direction(d_px: tuple[float, float], quad: np.ndarray, to_page: Mapper) -> tuple[float, float]:
    """A pixel-space direction through the placement's linear map, unit length."""
    cx, cy = (float(v) for v in np.asarray(quad).mean(axis=0))
    p0 = to_page((cx, cy))
    p1 = to_page((cx + d_px[0], cy + d_px[1]))
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    n = float(np.hypot(dx, dy))
    return (dx / n, dy / n) if n > 0 else (1.0, 0.0)


# ---------------------------------------------------------------------------
# Geometry helpers
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


def _bbox_of(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _segment_to_vector(seg: Segment, to_page: Mapper, page_index: int, seqno: int, color) -> Vector:
    p0, p1 = to_page(seg.p0), to_page(seg.p1)
    return Vector(
        type="s", items=[("l", p0, p1)], width=seg.width, rect=_bbox_of([p0, p1]),
        **_base_vector_kwargs(page_index, seqno, color),
    )


def _arc_to_vector(arc: Arc, to_page: Mapper, page_index: int, seqno: int, color) -> Vector:
    points = [to_page(p) for p in arc.polyline] or [to_page(arc.center)]
    items = [("l", points[i], points[i + 1]) for i in range(len(points) - 1)]
    if not items:
        items = [("l", points[0], points[0])]
    return Vector(
        type="s", items=items, width=arc.width, rect=_bbox_of(points),
        **_base_vector_kwargs(page_index, seqno, color),
    )


def _accumulate_trace_boxes(boxes: dict, result, mapper: Mapper, offset, crop_shape, to_page: Mapper) -> None:
    """Only cheap page-space bbox tuples -- never `result`'s mask arrays."""
    x0, y0 = offset
    ch, cw = crop_shape[:2]
    boxes["component"].append(_bbox_of([to_page((x0, y0)), to_page((x0 + cw, y0 + ch))]))
    for chain in result.graph.chains:
        if chain:
            boxes["chain"].append(_bbox_of([mapper(p) for p in chain]))
    for junction in result.junctions:
        cx, cy = mapper(junction.xy)
        boxes["junction"].append((cx - 2.0, cy - 2.0, cx + 2.0, cy + 2.0))
    for seg in result.segments:
        boxes["segment"].append(_bbox_of([mapper(seg.p0), mapper(seg.p1)]))
    for arc in result.arcs:
        boxes["arc"].append(_bbox_of([mapper(p) for p in arc.polyline] or [mapper(arc.center)]))


# ---------------------------------------------------------------------------
# Debug rendering -- this backend's own layers, built from `commons/renderer`
# primitives (boxes/text/vectors) plus embedded-PNG raster pages. Nothing
# here is shared with any P3 backend or P2_Raster_To_Vec/Stub.
# ---------------------------------------------------------------------------
_C_CLUSTERS = "#7c3aed"
_C_CLEANED = "#0f766e"
_C_ENHANCED = "#475569"
_C_DIFF = "#dc2626"
_C_TILE = "#94a3b8"
_C_TILE_DETECT = "#f59e0b"
_C_MERGED = "#ea580c"
_C_OCR_PASS = "#16a34a"
_C_OCR_FAIL = "#9333ea"
_C_COMPONENT = "#2563eb"
_C_GRAPH_CHAIN = "#0d9488"
_C_JUNCTION = "#dc2626"
_C_SEGMENT = "#059669"
_C_ARC = "#c026d3"


def _hex_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def _rgb_hex(rgb) -> str:
    return "#" + "".join(f"{int(c):02x}" for c in rgb[:3])


def _new_blank_doc(page_meta):
    """`(page, doc)` -- a fresh page in *unrotated* MediaBox space; rotation
    is only applied in `_finish_doc`, after every raster is placed, so
    placement rects are never reinterpreted by `/Rotate`."""
    import pymupdf as fitz

    doc = fitz.open()
    page = doc.new_page(width=page_meta.width, height=page_meta.height)
    return page, doc


def _finish_doc(page_doc, page_meta) -> bytes:
    page, doc = page_doc
    try:
        page.set_rotation(page_meta.rotation)
        return doc.tobytes(deflate=True)
    finally:
        doc.close()


def _oriented_for_page(array: np.ndarray, image: Image) -> np.ndarray:
    """Reorient `array` (image pixel frame) so rows run down and columns run
    right in page space -- handles quarter-turn/flip placements exactly; a
    general skew is approximated by its nearest quarter-turn."""
    if image.transform is None:
        return array
    a, b, c, d, _e, _f = image.transform
    if abs(a) >= abs(b):  # image x -> page x
        out = array
        if a < 0:
            out = out[:, ::-1]
        if d < 0:
            out = out[::-1]
    else:                 # image x -> page y (quarter turn)
        out = np.swapaxes(array, 0, 1)
        if b < 0:
            out = out[::-1]
        if c < 0:
            out = out[:, ::-1]
    return np.ascontiguousarray(out)


def _place_raster(page, array: np.ndarray, image: Image, _shape) -> None:
    import pymupdf as fitz
    from PIL import Image as PILImage

    arr = np.asarray(_oriented_for_page(array, image), dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    mode = "L" if arr.ndim == 2 else {3: "RGB", 4: "RGBA"}[arr.shape[2]]
    buf = io.BytesIO()
    PILImage.fromarray(arr, mode=mode).save(buf, format="PNG", compress_level=6)
    page.insert_image(fitz.Rect(*image.bbox), stream=buf.getvalue(), keep_proportion=False)


def _boxes_layer(page_meta, stage: str, label: str, hexcolor: str, boxes) -> DebugLayer:
    from rastervec.commons.renderer import render_boxes_pdf

    rgb = _hex_rgb(hexcolor)
    return (stage, label, hexcolor, render_boxes_pdf(page_meta, [(b, rgb) for b in boxes]))


def _page_layers(page_meta, raster_layers: "list[DebugLayer]", state: _PageState) -> "list[DebugLayer]":
    from rastervec.commons.renderer import render_text_pdf, render_vectors_pdf

    bx = state.boxes
    out: "list[DebugLayer]" = [layer for layer in raster_layers if not layer[0].startswith("_")]
    out += [
        _boxes_layer(page_meta, "ocr", "tiles", _C_TILE, bx["tile"]),
        _boxes_layer(page_meta, "ocr", "tile detect bbox", _C_TILE_DETECT, bx["tile_detect"]),
        _boxes_layer(page_meta, "ocr", "merged bbox", _C_MERGED, bx["merged"]),
        _boxes_layer(page_meta, "ocr", "passed bbox", _C_OCR_PASS, bx["refined_passed"]),
        _boxes_layer(page_meta, "ocr", "failed bbox", _C_OCR_FAIL, bx["refined_failed"]),
        ("ocr", "passed text", _C_OCR_PASS, render_text_pdf(
            page_meta, state.texts, color_of=lambda _t: _hex_rgb(_C_OCR_PASS),
        )),
        _boxes_layer(page_meta, "components", "component bbox", _C_COMPONENT, bx["component"]),
        _boxes_layer(page_meta, "graph_build", "chain bbox", _C_GRAPH_CHAIN, bx["chain"]),
        _boxes_layer(page_meta, "graph_build", "junction", _C_JUNCTION, bx["junction"]),
        _boxes_layer(page_meta, "polyline_fit", "segment bbox", _C_SEGMENT, bx["segment"]),
        _boxes_layer(page_meta, "arc_detect", "arc bbox", _C_ARC, bx["arc"]),
        ("final_vectors", "vectors", _C_SEGMENT, render_vectors_pdf(
            page_meta, state.vectors, color_of=lambda v: tuple(v.color or (0, 0, 0)),
        )),
    ]
    return out
