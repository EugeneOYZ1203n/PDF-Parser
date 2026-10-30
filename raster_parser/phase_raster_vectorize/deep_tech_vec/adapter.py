"""DeepTechVec's Phase2Backend entrypoint. Every embedded `Image` Phase 1
hands over goes through, in order,

  1-3. raster prelude     the shared `commons.raster_text.raster_text.prepare_images`:
                           tiled OCR on the raw image -> colour CLAHE (LAB L)
                           -> simple colour layers (achromatic + hue peaks)
                           -> each recognized word's ink layer erased
  4. learned vectorizing   per ink color layer: the layer's binary mask (ink
                           0, everything else 255), resampled to the canonical
                           `TARGET_PX_PER_PT`, through Egiazarian et al.'s
                           (ECCV 2020) pipeline -- 64 px patches -> the
                           authors' pretrained primitive network -> energy
                           refinement -> merging
                           (`inference.py`, `refine.py`, `merge.py`, `model/`)
  5. diff (debug only)     the primitives rasterized and compared back against
                           their own layer's ink (`diff.py`)

Steps 1-3 are the shared raster prelude (`commons/raster_text`); step 5's diff uses the stroke backends' shared diff codes
(`commons/raster_text/debug_tools.py`). The paper's step (1), a
cleaning U-Net turning a degraded scan into a clean black-and-white drawing,
is not run: its input here is already the clean binary mask of one of the
prelude's colour layers (the same input DeepVectoriser reads), and the
authors released no cleaning weights.

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
from typing import TYPE_CHECKING, Callable

import cv2
import numpy as np
from tqdm import tqdm

from raster_parser.models.deep_tech_vec import DeepTechVecPageState, DiffPolyline
from raster_parser.commons.logging_setup import get_logger
from raster_parser.models.image import Image
from raster_parser.models.page import Page
from raster_parser.models.text import Text
from raster_parser.models.vector import Vector
from raster_parser.commons.renderer.draw import bbox_spec, render_specs_pdf
from raster_parser.phase_raster_vectorize.deep_tech_vec import diff as diff_mod
from raster_parser.phase_raster_vectorize.deep_tech_vec import geometry as geo
from raster_parser.phase_raster_vectorize.deep_tech_vec.config import (
    DIFF_TOLERANCE_PX,
    FLAT_TOL_PX,
    INK_GRAY_THRESHOLD,
    TARGET_PX_PER_PT,
)
from raster_parser.phase_raster_vectorize.deep_tech_vec.inference import vectorize_layer
from raster_parser.phase_raster_vectorize.deep_tech_vec.masks import layer_image  # noqa: F401 -- also adapter.layer_image
from raster_parser.commons.raster_text.raster_text import (
    PreparedImage,
    prelude_debug_layers,
    prepare_images,
)
from raster_parser.commons.raster_text.debug_tools import (
    RasterSink,
    compose_raster_layers,
    geometry_layer,
    traced_vector_kwargs,
)
from raster_parser.commons.renderer.draw import hex_to_rgb01
from raster_parser.commons.helpers.geometry import points_bbox, rgb255_to_hex

DebugLayer = "tuple[str, str, str, bytes]"
OnDebugLayer = "Callable[[str, str, str, bytes], None]"
Mapper = Callable[[tuple[float, float]], tuple[float, float]]

if TYPE_CHECKING:
    from raster_parser.models.settings import RunSettings

_LOG = get_logger("deep_tech_vec")


# A predicted width is floored here so a zero-width primitive still draws (page points).
_MIN_WIDTH_PT = 1e-3


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
    refine_iters: "int | None" = None,
) -> tuple[list[Vector], list[Text]]:
    """Runs the full flow (module docstring) on every image. `compute` is a
    Pool-2 proxy for PaddleOCR + model calls (forwarded by `core.pipeline`
    by signature). `ocr_fns` optionally overrides the prelude's `run_ocr`
    `detect_many`/`recognize`/`recognize_raw`, and `model_fn` the model
    (`list[tile uint8] -> list[inference.TilePrediction]`) -- tests inject
    fakes; `refine_iters` overrides `config.REFINE_ITERS`. Without
    `model_fn`, missing weights raise `FileNotFoundError`."""
    if settings is not None:
        on_debug_layer = on_debug_layer or settings.on_debug_layer
        on_debug_image = on_debug_image or settings.on_debug_image
    state = DeepTechVecPageState()
    sink = RasterSink(
        stream=on_debug_layer is not None,
        keep=debug_out is not None and keep_debug_arrays,
    )
    t_start = time.perf_counter()
    _LOG.info("DeepTechVec: page %d, %d embedded image(s)", page.meta.index, len(images))
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
            desc="DeepTechVec images",
            unit="img",
            leave=False,
            disable=n_images <= 1,
        )
    ):
        _LOG.info("DeepTechVec image %d/%d", i + 1, n_images)
        _process_image(prepared, page, compute, state, sink, model_fn, refine_iters)
        del prepared  # free this image's arrays before the next one

    if on_debug_layer is not None:
        t0 = time.perf_counter()
        layers_out = _page_layers(page.meta, sink.finish_stream(page.meta), state)
        for layer in layers_out:
            on_debug_layer(*layer)
        _LOG.info(
            "DeepTechVec debug layers: %d emitted (%.1fs)",
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
        "DeepTechVec done: %d vectors, %d texts (%.1fs)",
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
        page_meta, raster, debug_out.get("state") or DeepTechVecPageState()
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
    state: DeepTechVecPageState,
    sink: RasterSink,
    model_fn,
    refine_iters: "int | None",
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
    extra = {} if refine_iters is None else {"refine_iters": refine_iters}
    for layer in layers.ink_layers():
        color_rgb = layers.colors_rgb[layer]
        color = tuple(float(c) / 255.0 for c in color_rgb)
        hexcolor = rgb255_to_hex(color_rgb)
        t0 = time.perf_counter()
        layer_gray = layer_image(layers.labels, layer, scale)
        res = vectorize_layer(
            layer_gray, model_fn=model_fn, compute=compute, desc=f"DeepTechVec {hexcolor}", **extra
        )
        for prim in res.prims:
            state.vectors.append(
                prim_to_vector(
                    prim,
                    canon_to_page,
                    meta.index,
                    state.next_seqno(),
                    color,
                    prim.width / TARGET_PX_PER_PT,
                )
            )
        _accumulate_debug(state, res, canon_to_page)
        _LOG.info(
            "  vectorize layer %s: %d patch(es) -> %d raw / %d refined -> %d merged primitive(s) (%.1fs)",
            hexcolor,
            len(res.patches),
            len(res.raw),
            len(res.refined),
            len(res.prims),
            time.perf_counter() - t0,
        )
        if sink.active:
            polylines = [DiffPolyline(geo.prim_points(p, 8), 1.0) for p in res.prims]
            codes, _ = diff_mod.diff_codes(
                layer_gray < INK_GRAY_THRESHOLD, polylines, DIFF_TOLERANCE_PX
            )
            codes = codes[: layer_gray.shape[0], : layer_gray.shape[1]]
            native = cv2.resize(codes, (w, h), interpolation=cv2.INTER_NEAREST)
            np.copyto(total_codes, native, where=native > 0)
    if sink.active:
        sink.add("vector_diff", "total", _C_DIFF, diff_mod.codes_to_rgba(total_codes), image, meta)


def prim_to_vector(
    prim: geo.Prim, to_page: Mapper, page_index: int, seqno: int, color, width_pt: float
) -> Vector:
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
        type="s",
        items=items,
        width=max(width_pt, _MIN_WIDTH_PT),
        rect=points_bbox(pts),
        **traced_vector_kwargs(page_index, seqno, color),
    )


def _accumulate_debug(state: DeepTechVecPageState, res, canon_to_page: Mapper) -> None:
    for x0, y0, s in res.patches:
        state.boxes["patch"].append(
            points_bbox([canon_to_page((x0, y0)), canon_to_page((x0 + s, y0 + s))])
        )
    for p in res.raw:
        state.geom["raw"].append([canon_to_page(q) for q in geo.prim_points(p, 6)])
        state.geom["raw_ends"].extend([canon_to_page(p.start), canon_to_page(p.end)])
    for p in res.refined:
        state.geom["refined"].append([canon_to_page(q) for q in geo.prim_points(p, 6)])
    for p in res.prims:
        state.geom["merged_ends"].extend([canon_to_page(p.start), canon_to_page(p.end)])


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
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


def _boxes_layer(page_meta, stage: str, label: str, hexcolor: str, boxes) -> DebugLayer:
    rgb = hex_to_rgb01(hexcolor)
    return (stage, label, hexcolor, render_specs_pdf(page_meta, [bbox_spec(b, rgb) for b in boxes]))


def _page_layers(
    page_meta, raster_layers: "list[DebugLayer]", state: DeepTechVecPageState
) -> "list[DebugLayer]":
    from raster_parser.commons.renderer.pdf import render_vectors_pdf

    bx = state.boxes
    geom = state.geom
    out: "list[DebugLayer]" = list(raster_layers)
    out += [
        _boxes_layer(page_meta, "tiles", "patch grid", _C_PATCH, bx["patch"]),
        geometry_layer(page_meta, "primitives", "raw (network)", _C_RAW, polylines=geom["raw"]),
        geometry_layer(
            page_meta, "primitives", "raw endpoints", _C_ENDPOINT, points=geom["raw_ends"]
        ),
        geometry_layer(page_meta, "refine", "refined", _C_REFINED, polylines=geom["refined"]),
        geometry_layer(
            page_meta, "merge", "merged endpoints", _C_MERGED_END, points=geom["merged_ends"]
        ),
        (
            "merge",
            "merged vectors",
            _C_FINAL,
            render_vectors_pdf(
                page_meta,
                state.vectors,
                color_of=lambda v: tuple(v.color or (0, 0, 0)),
            ),
        ),
    ]
    return out
