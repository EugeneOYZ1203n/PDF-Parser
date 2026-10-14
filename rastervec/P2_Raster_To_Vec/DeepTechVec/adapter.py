"""DeepTechVec's Phase2Backend entrypoint. Every embedded `Image` Phase 1
hands over goes through, in order,

  1-3. raster prelude     the shared `RasterText.raster_text.prepare_images`:
                           tiled OCR on the raw image -> colour CLAHE (LAB L)
                           -> simple colour layers (achromatic + hue peaks)
                           -> each recognized word's ink layer erased
  4. learned vectorizing   per ink color layer: the layer's binary mask (ink
                           0, everything else 255), resampled to the canonical
                           `TARGET_PX_PER_PT`, through Egiazarian et al.'s
                           (ECCV 2020) pipeline -- 64 px patches -> primitive
                           network -> energy refinement -> merging
                           (`inference.py`, `refine.py`, `merge.py`, `model/`)
  5. diff (debug only)     the primitives rasterized and compared back against
                           their own layer's ink (`diff.py`)

Steps 1-3 are the shared raster prelude (the one module P2 backends
share); step 5's diff is Junction's, *copied*. The paper's step (1), a
cleaning U-Net turning a degraded scan into a clean black-and-white drawing,
is not run: its input here is already the clean binary mask of one of the
prelude's colour layers (the same input DeepVectoriser reads), and the
shared dataset has no degraded/clean pairs to train it on.

Output: one `Vector` per merged primitive -- a line as one `"l"` item, a
quadratic Bezier as one exact cubic `"c"` item (`"l"` when flat,
`FLAT_TOL_PX`) -- in its layer's color, with the network's *predicted*
width, plus OCR `Text`s, all in page space.

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

from rastervec.commons.logging_setup import get_logger
from rastervec.commons.models import Image, Page, Text, Vector
from rastervec.commons.renderer.draw import (
    bbox_spec, dot_spec, draw_image, image_spec, new_page_doc, polyline_spec, render_specs_pdf,
)
from rastervec.P2_Raster_To_Vec.DeepTechVec import diff as diff_mod
from rastervec.P2_Raster_To_Vec.DeepTechVec import geometry as geo
from rastervec.P2_Raster_To_Vec.DeepTechVec.config import (
    DIFF_TOLERANCE_PX, FLAT_TOL_PX, INK_GRAY_THRESHOLD, TARGET_PX_PER_PT,
)
from rastervec.P2_Raster_To_Vec.DeepTechVec.inference import vectorize_layer
from rastervec.P2_Raster_To_Vec.DeepTechVec.masks import layer_image  # noqa: F401 -- also adapter.layer_image
from rastervec.P2_Raster_To_Vec.RasterText.raster_text import (
    PreparedImage, prelude_debug_layers, prepare_images,
)

DebugLayer = "tuple[str, str, str, bytes]"
OnDebugLayer = "Callable[[str, str, str, bytes], None]"
Mapper = Callable[[tuple[float, float]], tuple[float, float]]

_LOG = get_logger("P2.DeepTechVec")


def extract(
    images: list[Image], page: Page, *, compute=None,
    debug_out: "dict | None" = None, on_debug_layer: "OnDebugLayer | None" = None, on_debug_image=None,
    ocr_fns: "dict | None" = None, model_fn=None, keep_debug_arrays: bool = True,
    refine_iters: "int | None" = None,
) -> tuple[list[Vector], list[Text]]:
    """Runs the full flow (module docstring) on every image. `compute` is a
    Pool-2 proxy for PaddleOCR + model calls (forwarded by `core.pipeline`
    by signature). `ocr_fns` optionally overrides the prelude's `run_ocr`
    `detect_many`/`recognize`/`recognize_raw`, and `model_fn` the model
    (`list[tile uint8] -> list[inference.TilePrediction]`) -- tests inject
    fakes; `refine_iters` overrides `config.REFINE_ITERS`. Without
    `model_fn`, missing weights raise `FileNotFoundError`."""
    state = _PageState()
    sink = _RasterSink(
        stream=on_debug_layer is not None,
        keep=debug_out is not None and keep_debug_arrays,
    )
    t_start = time.perf_counter()
    _LOG.info("DeepTechVec: page %d, %d embedded image(s)", page.meta.index, len(images))
    prelude = prepare_images(images, page, compute=compute, ocr_fns=ocr_fns,
                             keep_debug=sink.keep, on_debug_layer=on_debug_layer,
                             on_debug_image=on_debug_image)
    state.texts = prelude.texts
    state.seqno = len(prelude.texts)
    for i, prepared in enumerate(tqdm(prelude.images, desc="DeepTechVec images", unit="img",
                                      leave=False, disable=len(prelude.images) <= 1)):
        _LOG.info("DeepTechVec image %d/%d", i + 1, len(prelude.images))
        _process_image(prepared, page, compute, state, sink, model_fn, refine_iters)

    if on_debug_layer is not None:
        t0 = time.perf_counter()
        layers_out = _page_layers(page.meta, sink.finish_stream(page.meta), state)
        for layer in layers_out:
            on_debug_layer(*layer)
        _LOG.info("DeepTechVec debug layers: %d emitted (%.1fs)", len(layers_out),
                  time.perf_counter() - t0)
    if debug_out is not None:
        debug_out["raster_prelude"] = prelude
        debug_out["raster_images"] = sink.kept
        debug_out["state"] = state
        debug_out["vectors"] = state.vectors
        debug_out["texts"] = state.texts
    _LOG.info("DeepTechVec done: %d vectors, %d texts (%.1fs)", len(state.vectors), len(state.texts),
              time.perf_counter() - t_start)
    return state.vectors, state.texts


def render_debug(debug_out: "dict | None", page_meta) -> "list[DebugLayer]":
    """Batch counterpart of the `on_debug_layer` path -- identical layers,
    rebuilt from `debug_out`'s kept raw arrays."""
    if not debug_out:
        return []
    raster = _compose_raster_layers(page_meta, debug_out.get("raster_images") or [])
    return (prelude_debug_layers(page_meta, debug_out.get("raster_prelude"))
            + _page_layers(page_meta, raster, debug_out.get("state") or _PageState()))


# ---------------------------------------------------------------------------
# Per-page accumulation
# ---------------------------------------------------------------------------
@dataclass
class _PageState:
    vectors: list[Vector] = field(default_factory=list)
    texts: list[Text] = field(default_factory=list)
    seqno: int = 0
    boxes: dict = field(default_factory=lambda: {
        "patch": [],
    })
    # page-space primitive geometry for the debug layers
    geom: dict = field(default_factory=lambda: {"raw": [], "raw_ends": [], "refined": [], "merged_ends": []})

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
def _process_image(prepared: PreparedImage, page: Page, compute, state: _PageState, sink: _RasterSink,
                   model_fn, refine_iters: "int | None") -> None:
    image = prepared.image
    layers = prepared.layers
    h, w = layers.labels.shape[:2]
    meta = page.meta
    to_page = prepared.to_page
    px_per_pt = prepared.px_per_pt
    scale = TARGET_PX_PER_PT / px_per_pt  # native px -> canonical px
    _LOG.info("  %dx%d px, %.2f px/pt (x%.3f to canonical), %d colour layer(s)", w, h, px_per_pt,
              scale, layers.n_layers)

    # 4. per-layer learned vectorizing (+ 5. diff when debugging)
    def canon_to_page(pt) -> tuple[float, float]:
        return to_page((float(pt[0]) / scale, float(pt[1]) / scale))

    total_codes = np.zeros((h, w), dtype=np.uint8) if sink.active else None
    extra = {} if refine_iters is None else {"refine_iters": refine_iters}
    for layer in layers.ink_layers():
        color_rgb = layers.colors_rgb[layer]
        color = tuple(float(c) / 255.0 for c in color_rgb)
        hexcolor = _rgb_hex(color_rgb)
        t0 = time.perf_counter()
        layer_gray = layer_image(layers.labels, layer, scale)
        res = vectorize_layer(layer_gray, model_fn=model_fn, compute=compute,
                              desc=f"DeepTechVec {hexcolor}", **extra)
        for prim in res.prims:
            state.vectors.append(prim_to_vector(
                prim, canon_to_page, meta.index, state.next_seqno(), color, prim.width / TARGET_PX_PER_PT,
            ))
        _accumulate_debug(state, res, canon_to_page)
        _LOG.info("  vectorize layer %s: %d patch(es) -> %d raw / %d refined -> %d merged primitive(s) (%.1fs)",
                  hexcolor, len(res.patches), len(res.raw), len(res.refined), len(res.prims),
                  time.perf_counter() - t0)
        if sink.active:
            polylines = [_DiffPolyline(geo.prim_points(p, 8), 1.0) for p in res.prims]
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


def prim_to_vector(prim: geo.Prim, to_page: Mapper, page_index: int, seqno: int, color,
                   width_pt: float) -> Vector:
    """One `Vector` per primitive: a line -> one `"l"` item; a quadratic ->
    its exact cubic as one `"c"` item, or an `"l"` when flat (judged in
    canonical px). Mapped to page space."""
    if prim.is_line or geo.quad_is_flat(prim.pts, FLAT_TOL_PX):
        a, b = to_page(prim.pts[0]), to_page(prim.pts[-1])
        items = [("l", a, b)]
        pts = [a, b]
    else:
        pts = [to_page(q) for q in geo.quad_to_cubic(prim.pts)]
        items = [("c", *pts)]
    return Vector(
        type="s", items=items, width=max(width_pt, 1e-3), rect=_bbox_of(pts),
        **_base_vector_kwargs(page_index, seqno, color),
    )


def _accumulate_debug(state: _PageState, res, canon_to_page: Mapper) -> None:
    for x0, y0, s in res.patches:
        state.boxes["patch"].append(_bbox_of([canon_to_page((x0, y0)), canon_to_page((x0 + s, y0 + s))]))
    for p in res.raw:
        state.geom["raw"].append([canon_to_page(q) for q in geo.prim_points(p, 6)])
        state.geom["raw_ends"].extend([canon_to_page(p.start), canon_to_page(p.end)])
    for p in res.refined:
        state.geom["refined"].append([canon_to_page(q) for q in geo.prim_points(p, 6)])
    for p in res.prims:
        state.geom["merged_ends"].extend([canon_to_page(p.start), canon_to_page(p.end)])


# ---------------------------------------------------------------------------
# Geometry helpers (copied from Junction)
# ---------------------------------------------------------------------------
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
_C_DIFF = "#dc2626"
_C_PATCH = "#2563eb"
_C_MERGED_END = "#e11d48"
_C_RAW = "#0d9488"
_C_REFINED = "#db2777"
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


def _geometry_layer(page_meta, stage: str, label: str, hexcolor: str, *,
                    polylines=(), points=(), width: float = 0.6, radius: float = 0.8) -> DebugLayer:
    rgb = _hex_rgb(hexcolor)
    specs = [polyline_spec(pl, rgb, width=width) for pl in polylines if len(pl) >= 2]
    specs += [dot_spec(p, rgb, radius) for p in points]
    return (stage, label, hexcolor, render_specs_pdf(page_meta, specs))


def _page_layers(page_meta, raster_layers: "list[DebugLayer]", state: _PageState) -> "list[DebugLayer]":
    from rastervec.commons.renderer import render_vectors_pdf

    bx = state.boxes
    geom = state.geom
    out: "list[DebugLayer]" = list(raster_layers)
    out += [
        _boxes_layer(page_meta, "tiles", "patch grid", _C_PATCH, bx["patch"]),
        _geometry_layer(page_meta, "primitives", "raw (network)", _C_RAW, polylines=geom["raw"]),
        _geometry_layer(page_meta, "primitives", "raw endpoints", _C_ENDPOINT, points=geom["raw_ends"]),
        _geometry_layer(page_meta, "refine", "refined", _C_REFINED, polylines=geom["refined"]),
        _geometry_layer(page_meta, "merge", "merged endpoints", _C_MERGED_END, points=geom["merged_ends"]),
        ("merge", "merged vectors", _C_FINAL, render_vectors_pdf(
            page_meta, state.vectors, color_of=lambda v: tuple(v.color or (0, 0, 0)),
        )),
    ]
    return out
