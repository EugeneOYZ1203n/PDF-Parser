"""ImplicitSketchVec's Phase2Backend entrypoint. Every embedded `Image` Phase 1
hands over goes through, in order,

  1-3. raster prelude     the shared `commons.raster_text.raster_text.prepare_images`:
                           tiled OCR on the raw image -> colour CLAHE (LAB L)
                           -> simple colour layers (achromatic + hue peaks)
                           -> each recognized word's ink layer erased
  4. learned vectorizing   per ink color layer: the layer's binary mask (ink
                           0, everything else 255), resampled to the canonical
                           `TARGET_PX_PER_PT`, through Yan et al.'s (SIGGRAPH
                           2024) pipeline -- Distance Field Prediction (six
                           UDFs at 2x) -> Line Reconstruction (2D Neural Dual
                           Contouring) -> post-processing in the DC domain ->
                           curve fitting (`inference.py`, `postprocess.py`,
                           `model/`)
  5. diff (debug only)     the strokes rasterized and compared back against
                           their own layer's ink (`diff.py`)

Steps 1-3 are the shared raster prelude (`commons/raster_text`); step 5's diff uses the stroke backends' shared diff codes
(`commons/raster_text/debug_tools.py`).

Output: one `Vector` per grouped stroke -- `"c"` items, `"l"` where a piece is
flat (`FLAT_TOL_PX`) -- in its layer's color, width estimated from the ink's
distance transform, plus OCR `Text`s, all in page space.

Coordinate mapping: canonical pixel -> native pixel (`/ scale`) -> unit
square -> the placement's own image matrix (`Image.transform`), so rotated/
flipped placements land correctly; an image without a transform falls back
to an upright fill of `Image.bbox`. Every map is affine, so Bezier control
points map exactly."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable

import cv2
import numpy as np
from tqdm import tqdm

from raster_parser.models.implicit_sketch_vec import ImplicitSketchVecPageState, DiffPolyline
from raster_parser.commons.logging_setup import get_logger
from raster_parser.models.image import Image
from raster_parser.models.page import Page
from raster_parser.models.text import Text
from raster_parser.models.vector import Vector
from raster_parser.commons.renderer.draw import bbox_spec, render_specs_pdf
from raster_parser.phase_raster_vectorize.implicit_sketch_vec import diff as diff_mod
from raster_parser.phase_raster_vectorize.implicit_sketch_vec import geometry as geo
from raster_parser.phase_raster_vectorize.implicit_sketch_vec.config import (
    DIFF_TOLERANCE_PX,
    INK_GRAY_THRESHOLD,
    TARGET_PX_PER_PT,
)
from raster_parser.phase_raster_vectorize.implicit_sketch_vec.inference import vectorize_layer
from raster_parser.phase_raster_vectorize.implicit_sketch_vec.masks import layer_image  # noqa: F401 -- also adapter.layer_image
from raster_parser.commons.raster_text.raster_text import (
    PreparedImage,
    prelude_debug_layers,
    prepare_images,
)
from raster_parser.commons.raster_text.strokes import stroke_to_vector, stroke_width_px
from raster_parser.commons.raster_text.debug_tools import (
    RasterSink,
    compose_raster_layers,
    geometry_layer,
)
from raster_parser.commons.renderer.draw import hex_to_rgb01
from raster_parser.commons.helpers.geometry import points_bbox, rgb255_to_hex

DebugLayer = "tuple[str, str, str, bytes]"
OnDebugLayer = "Callable[[str, str, str, bytes], None]"
Mapper = Callable[[tuple[float, float]], tuple[float, float]]

if TYPE_CHECKING:
    from raster_parser.models.settings import RunSettings

_LOG = get_logger("implicit_sketch_vec")


# Debug rasters: drawn pixels are at least this opaque (0..255).
_MIN_ALPHA = 160
# `cv2.polylines(..., shift=_SUBPIXEL_SHIFT)` reads fixed-point coordinates: x 2**shift.
_SUBPIXEL_SHIFT = 4
_SUBPIXEL = 1 << _SUBPIXEL_SHIFT


def extract(
    images: list[Image],
    page: Page,
    settings: "RunSettings | None" = None,
    *,
    compute=None,
    debug_out: "dict | None" = None,
    on_debug_layer: "OnDebugLayer | None" = None,
    on_debug_image=None,
    ocr_fns: "dict | None" = None,
    model_fn=None,
    keep_debug_arrays: bool = True,
) -> tuple[list[Vector], list[Text]]:
    """Runs the full flow (module docstring) on every image. `compute` is a
    Pool-2 proxy for PaddleOCR + model calls (forwarded by `core.pipeline`
    by signature). `ocr_fns` optionally overrides the prelude's `run_ocr`
    `detect_many`/`recognize`/`recognize_raw`, and `model_fn` the model
    (`list[tile uint8] -> list[inference.TilePrediction]`) -- tests inject
    fakes. Without `model_fn`, missing weights raise `FileNotFoundError`."""
    if settings is not None:
        on_debug_layer = on_debug_layer or settings.on_debug_layer
        on_debug_image = on_debug_image or settings.on_debug_image
    state = ImplicitSketchVecPageState()
    sink = RasterSink(
        stream=on_debug_layer is not None,
        keep=debug_out is not None and keep_debug_arrays,
    )
    t_start = time.perf_counter()
    _LOG.info("ImplicitSketchVec: page %d, %d embedded image(s)", page.meta.index, len(images))
    prelude = prepare_images(
        images,
        page,
        compute=compute,
        ocr_fns=ocr_fns,
        keep_debug=sink.keep,
        on_debug_layer=on_debug_layer,
        on_debug_image=on_debug_image,
    )
    state.texts = prelude.texts
    state.seqno = len(prelude.texts)
    n_images = len(prelude.images)
    for i, prepared in enumerate(
        tqdm(
            prelude.consume_images(),
            total=n_images,
            desc="ImplicitSketchVec images",
            unit="img",
            leave=False,
            disable=n_images <= 1,
        )
    ):
        _LOG.info("ImplicitSketchVec image %d/%d", i + 1, n_images)
        _process_image(prepared, page, compute, state, sink, model_fn)
        del prepared  # free this image's arrays before the next one

    if on_debug_layer is not None:
        t0 = time.perf_counter()
        layers_out = _page_layers(page.meta, sink.finish_stream(page.meta), state)
        for layer in layers_out:
            on_debug_layer(*layer)
        _LOG.info(
            "ImplicitSketchVec debug layers: %d emitted (%.1fs)",
            len(layers_out),
            time.perf_counter() - t0,
        )
    if debug_out is not None:
        debug_out["raster_prelude"] = prelude
        debug_out["raster_images"] = sink.kept
        debug_out["state"] = state
        debug_out["vectors"] = state.vectors
        debug_out["texts"] = state.texts
    _LOG.info(
        "ImplicitSketchVec done: %d vectors, %d texts (%.1fs)",
        len(state.vectors),
        len(state.texts),
        time.perf_counter() - t_start,
    )
    return state.vectors, state.texts


def render_debug(debug_out: "dict | None", page_meta) -> "list[DebugLayer]":
    """Batch counterpart of the `on_debug_layer` path -- identical layers,
    rebuilt from `debug_out`'s kept raw arrays."""
    if not debug_out:
        return []
    raster = compose_raster_layers(page_meta, debug_out.get("raster_images") or [])
    return prelude_debug_layers(page_meta, debug_out.get("raster_prelude")) + _page_layers(
        page_meta, raster, debug_out.get("state") or ImplicitSketchVecPageState()
    )


# ---------------------------------------------------------------------------
# Per-page accumulation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-image flow
# ---------------------------------------------------------------------------
def _process_image(
    prepared: PreparedImage,
    page: Page,
    compute,
    state: ImplicitSketchVecPageState,
    sink: RasterSink,
    model_fn,
) -> None:
    image = prepared.image
    layers = prepared.layers
    h, w = layers.labels.shape[:2]
    meta = page.meta
    to_page = prepared.to_page
    px_per_pt = prepared.px_per_pt
    scale = TARGET_PX_PER_PT / px_per_pt  # native px -> canonical px
    _LOG.info(
        "  %dx%d px, %.2f px/pt (x%.3f to canonical), %d colour layer(s)",
        w,
        h,
        px_per_pt,
        scale,
        layers.n_layers,
    )

    # 4. per-layer learned vectorizing (+ 5. diff when debugging)
    def canon_to_page(pt) -> tuple[float, float]:
        return to_page((float(pt[0]) / scale, float(pt[1]) / scale))

    total_codes = np.zeros((h, w), dtype=np.uint8) if sink.active else None
    for layer in layers.ink_layers():
        color_rgb = layers.colors_rgb[layer]
        color = tuple(float(c) / 255.0 for c in color_rgb)
        hexcolor = rgb255_to_hex(color_rgb)
        t0 = time.perf_counter()
        layer_gray = layer_image(layers.labels, layer, scale)
        res = vectorize_layer(
            layer_gray,
            model_fn=model_fn,
            compute=compute,
            debug=sink.active,
            desc=f"ImplicitSketchVec {hexcolor}",
        )
        dist = cv2.distanceTransform(
            (layer_gray < INK_GRAY_THRESHOLD).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_3
        )
        for stroke in res.strokes:
            width_pt = stroke_width_px(stroke, dist) / TARGET_PX_PER_PT
            state.vectors.append(
                stroke_to_vector(
                    stroke,
                    canon_to_page,
                    meta.index,
                    state.next_seqno(),
                    color,
                    width_pt,
                )
            )
        _accumulate_debug(state, res, canon_to_page)
        _LOG.info(
            "  vectorize layer %s: %d tile(s), %d keypoint(s), %d USM region(s) -> %d stroke(s) (%.1fs)",
            hexcolor,
            len(res.tiles),
            sum(len(v) for v in res.keypoints.values()),
            len(res.regions),
            len(res.strokes),
            time.perf_counter() - t0,
        )
        if sink.active:
            heat = cv2.resize(res.heat, (w, h), interpolation=cv2.INTER_AREA)
            rgba = np.zeros((h, w, 4), np.uint8)
            rgba[..., :3] = _HEAT_RGB
            rgba[..., 3] = heat
            sink.add("dfp", "udf centerline", _C_HEAT, rgba, image, meta)
            # the dense DC-domain layers (up to one mark per 0.5 px cell) as
            # rasters: as vector paths they cost a PDF path each
            shape = layer_gray.shape
            sink.add(
                "dfp",
                "under-sampling map",
                _C_USM,
                _mask_rgba(_cells_mask(res.usm_cells, shape), (w, h), _C_USM),
                image,
                meta,
            )
            sink.add(
                "ndc",
                "raw edges",
                _C_RAW,
                _mask_rgba(_segments_mask(res.raw_segments, shape), (w, h), _C_RAW),
                image,
                meta,
            )
            sink.add(
                "refine",
                "refined edges",
                _C_GRAPH,
                _mask_rgba(_segments_mask(res.graph_segments, shape), (w, h), _C_GRAPH),
                image,
                meta,
            )
            polylines = [DiffPolyline(geo.sample_stroke(s, 8), 1.0) for s in res.strokes]
            codes, _ = diff_mod.diff_codes(
                layer_gray < INK_GRAY_THRESHOLD, polylines, DIFF_TOLERANCE_PX
            )
            codes = codes[: layer_gray.shape[0], : layer_gray.shape[1]]
            native = cv2.resize(codes, (w, h), interpolation=cv2.INTER_NEAREST)
            np.copyto(total_codes, native, where=native > 0)
    if sink.active:
        sink.add("vector_diff", "total", _C_DIFF, diff_mod.codes_to_rgba(total_codes), image, meta)


def _segments_mask(segments, shape) -> np.ndarray:
    """Segments (canonical px) drawn 1 px wide into a `shape` mask."""
    m = np.zeros(shape, np.uint8)
    if segments:
        lines = np.round(np.array([[p, q] for p, q in segments], float) * _SUBPIXEL).astype(
            np.int32
        )
        cv2.polylines(
            m, list(lines.reshape(-1, 2, 1, 2)), False, 255, 1, cv2.LINE_8, shift=_SUBPIXEL_SHIFT
        )
    return m


def _cells_mask(cells, shape) -> np.ndarray:
    """Under-sampled 1/SR px cells -> the canonical pixels they fall in."""
    from .config import SR

    m = np.zeros(shape, np.uint8)
    if cells:
        c = np.array(sorted(cells), np.int64) // SR
        ok = (c[:, 0] >= 0) & (c[:, 0] < shape[1]) & (c[:, 1] >= 0) & (c[:, 1] < shape[0])
        m[c[ok, 1], c[ok, 0]] = 255
    return m


def _mask_rgba(mask: np.ndarray, size: tuple[int, int], hexcolor: str) -> np.ndarray:
    """A canonical-px mask -> an RGBA overlay at the image's native size."""
    native = cv2.resize(
        mask, size, interpolation=cv2.INTER_AREA if mask.shape[1] > size[0] else cv2.INTER_NEAREST
    )
    rgba = np.zeros((size[1], size[0], 4), np.uint8)
    rgba[..., :3] = [int(c * 255) for c in hex_to_rgb01(hexcolor)]
    rgba[..., 3] = np.where(native > 0, np.maximum(native, _MIN_ALPHA), 0)
    return rgba


def _accumulate_debug(state: ImplicitSketchVecPageState, res, canon_to_page: Mapper) -> None:
    from .config import SR

    for x0, y0, s in res.tiles:
        state.boxes["vec_tile"].append(
            points_bbox([canon_to_page((x0, y0)), canon_to_page((x0 + s, y0 + s))])
        )
    for name, pts in res.keypoints.items():
        state.geom[name].extend(canon_to_page(p) for p in pts)

    for region in res.regions:
        cells = np.array(sorted(region), float)
        lo, hi = cells.min(0) / SR, (cells.max(0) + 1) / SR
        state.boxes["region"].append(points_bbox([canon_to_page(lo), canon_to_page(hi)]))
    for pl in res.polylines:
        state.geom["grouped"].append([canon_to_page(p) for p in pl])
    for s in res.strokes:
        state.geom["ends"].extend([canon_to_page(s[0, 0]), canon_to_page(s[-1, 3])])


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Debug rendering -- this backend's own layers, built from `commons/renderer`
# primitives plus embedded-PNG raster pages.
# ---------------------------------------------------------------------------
_C_DIFF = "#dc2626"
_C_VEC_TILE = "#2563eb"
_C_RESPLIT = "#e11d48"
_C_RAW = "#0d9488"
_C_HEAT = "#db2777"
_HEAT_RGB = (219, 39, 119)
_C_KP = {"end": "#16a34a", "sharp": "#f59e0b", "junc": "#dc2626"}
_C_USM = "#7c3aed"
_C_GRAPH = "#0891b2"
_C_ENDPOINT = "#ca8a04"
_C_FINAL = "#059669"


def _boxes_layer(page_meta, stage: str, label: str, hexcolor: str, boxes) -> DebugLayer:
    rgb = hex_to_rgb01(hexcolor)
    return (stage, label, hexcolor, render_specs_pdf(page_meta, [bbox_spec(b, rgb) for b in boxes]))


def _page_layers(
    page_meta, raster_layers: "list[DebugLayer]", state: ImplicitSketchVecPageState
) -> "list[DebugLayer]":
    from raster_parser.commons.renderer.pdf import render_vectors_pdf

    bx = state.boxes
    geom = state.geom
    out: "list[DebugLayer]" = list(raster_layers)
    out += [
        _boxes_layer(page_meta, "tiles", "tile grid", _C_VEC_TILE, bx["vec_tile"]),
        geometry_layer(page_meta, "dfp", "keypoints end", _C_KP["end"], points=geom["end"]),
        geometry_layer(page_meta, "dfp", "keypoints sharp", _C_KP["sharp"], points=geom["sharp"]),
        geometry_layer(page_meta, "dfp", "keypoints junc", _C_KP["junc"], points=geom["junc"]),
        _boxes_layer(page_meta, "topology", "refined regions", _C_RESPLIT, bx["region"]),
        geometry_layer(page_meta, "group", "strokes", _C_ENDPOINT, polylines=geom["grouped"]),
        geometry_layer(page_meta, "fit", "stroke endpoints", _C_RESPLIT, points=geom["ends"]),
        (
            "fit",
            "fitted vectors",
            _C_FINAL,
            render_vectors_pdf(
                page_meta,
                state.vectors,
                color_of=lambda v: tuple(v.color or (0, 0, 0)),
            ),
        ),
    ]
    return out
