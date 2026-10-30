"""Junction's raster_vectorize entry point (`RasterVectorizer`): every embedded `Image` native_read hands
over goes through, in order,
  1-4. raster prelude      the shared `commons.raster_text.prepare_images`:
                           tiled OCR on the raw image -> colour CLAHE (LAB L)
                           -> simple colour layers (achromatic + hue peaks)
                           -> each recognized word's ink layer erased
  5. tracing               per ink color layer, per 10 pt-tolerance spatial
                           component, one at a time (`components.py`),
                           `tracer.pipeline.run` at native resolution:
                           binarize -> solid fills (ink >= FILL_MIN_WIDTH_PT
                           wide) split off and outlined -> skeleton of the
                           rest -> node/edge graph (junction clusters, barb
                           pruning, junction-zone collapse) -> strokes joined
                           straight through junctions -> Douglas-Peucker,
                           width 1.0
  6. diff (debug only)     traced vectors rasterized and compared back
                           against their own ink (`diff.py`)

and comes back as `Vector`s (colored by their layer) + OCR `Text`s, all in
page space.

Coordinate mapping: pixel `(px, py)` of an `h`x`w` image -> unit square
`(px / w, py / h)` -> the placement's own image matrix (`Image.transform`,
from `get_image_info`), so rotated/flipped placements land correctly; an
image without a transform falls back to an upright fill of `Image.bbox`.

Debug output (`on_debug_layer` streaming / `debug_out` + `render_debug`
batch, both through the same helpers): the prelude's own layers (enhanced
colour image, ink, colour layers, text removal, ocr -- see `raster_text.py`),
then Junction's raster stages (rendered vectors, total + per-layer diff)
become full-resolution PNGs embedded in page-sized PDFs -- each layer label
appears once per page even with several images, the images composed onto
it; OCR and component stages become box / text layers, and the tracing
stages (skeleton chains, junction / end nodes, output polylines, fill
outlines; the fill mask is a raster layer) become
point / line layers drawn on the real geometry. On the streaming path
each raster debug image is PNG-encoded into its layer's page right away and
the array dropped, so no full-size debug array outlives its own step;
`debug_out` instead keeps the raw arrays (incl. the uint8 diff-code
canvases, `diff.MATCHED/MISSED/SPURIOUS`) for later missed-line work."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from raster_parser.models.raster_text import RasterItem, Mapper
from raster_parser.models.junction import JunctionPageState, Outline, Polyline
from raster_parser.commons.logging_setup import get_logger
from raster_parser.models.image import Image
from raster_parser.models.page import Page
from raster_parser.models.text import Text
from raster_parser.models.vector import Vector
from raster_parser.commons.renderer.draw import (
    bbox_spec,
    dot_batch_spec,
    hex_to_rgb01,
    new_page_doc,
    render_specs_pdf,
    stroke_batch_spec,
)
from raster_parser.phase_raster_vectorize.junction import diff as diff_mod
from raster_parser.phase_raster_vectorize.junction.components import layer_components
from raster_parser.phase_raster_vectorize.junction.config import (
    COMPONENT_TOLERANCE_PT,
    DIFF_TOLERANCE_PX,
    FILL_MIN_WIDTH_FLOOR_PX,
    FILL_MIN_WIDTH_PT,
)
from raster_parser.phase_raster_vectorize.junction.tracer.pipeline import Params, run
from raster_parser.phase_raster_vectorize.junction.tracer.simplify import fit_polyline
from raster_parser.commons.raster_text.enhance import to_gray
from raster_parser.commons.raster_text.raster_text import (
    PreparedImage,
    prelude_debug_layers,
    prepare_images,
)
from raster_parser.commons.helpers.geometry import rgb255_to_hex
from raster_parser.commons.raster_text.raster_text import place_raster
from raster_parser.commons.raster_text.debug_tools import (  # noqa: F401 -- shared with the other raster backends
    traced_vector_kwargs as _base_vector_kwargs,
    finish_doc as _finish_doc,
)
from raster_parser.commons.helpers.geometry import points_bbox

DebugLayer = "tuple[str, str, str, bytes]"
OnDebugLayer = "Callable[[str, str, str, bytes], None]"


if TYPE_CHECKING:
    from raster_parser.models.settings import RunSettings

_LOG = get_logger("junction")


def extract(
    images: list[Image],
    page: Page,
    settings: "RunSettings | None" = None,
    *,
    params: Params | None = None,
    compute=None,
    debug_out: "dict | None" = None,
    on_debug_layer: "OnDebugLayer | None" = None,
    on_debug_image=None,
    ocr_fns: "dict | None" = None,
    keep_debug_arrays: bool = True,
) -> tuple[list[Vector], list[Text]]:
    """Runs the full flow (module docstring) on every image -- the `RasterVectorizer` contract
    (`core/interfaces.py`); `settings` supplies the debug hooks unless they are passed directly.
    `compute` is a process-pool proxy for PaddleOCR calls. `ocr_fns` optionally overrides the prelude's
    `text_ocr.run_ocr` `detect_many`/`recognize`/`recognize_raw` (tests
    inject fakes).
    `keep_debug_arrays=False` keeps the full-resolution raster arrays out of
    `debug_out` -- with no `on_debug_layer` either, the debug-only raster
    work (recolor, image copies, per-component diff) is skipped entirely."""
    if settings is not None:
        on_debug_layer = on_debug_layer or settings.on_debug_layer
        on_debug_image = on_debug_image or settings.on_debug_image
    params = params or Params()
    state = JunctionPageState()
    sink = _DebugSink(
        stream=on_debug_layer is not None,
        keep=debug_out is not None and keep_debug_arrays,
        debug=on_debug_layer is not None or debug_out is not None,
    )
    t_start = time.perf_counter()
    _LOG.info("Junction: page %d, %d embedded image(s)", page.meta.index, len(images))
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
            desc="Junction images",
            unit="img",
            leave=False,
            disable=n_images <= 1,
        )
    ):
        _LOG.info("Junction image %d/%d", i + 1, n_images)
        _process_image(prepared, page, params, state, sink)
        del prepared  # free this image's arrays before the next one

    if on_debug_layer is not None:
        t0 = time.perf_counter()
        _LOG.debug("Junction debug layers: rendering")
        layers_out = _page_layers(page.meta, sink.finish_stream(page.meta), state)
        for layer in layers_out:
            on_debug_layer(*layer)
        _LOG.info(
            "Junction debug layers: %d emitted (%.1fs)", len(layers_out), time.perf_counter() - t0
        )
    if debug_out is not None:
        debug_out["raster_prelude"] = prelude
        debug_out["raster_images"] = sink.kept
        # (label, image, uint8 codes HxW): "total" + one per layer hex --
        # the raw missed/spurious maps, see diff.py.
        debug_out["diff_codes"] = [
            (it.label, it.image, it.array) for it in sink.kept if it.stage == "_codes"
        ]
        debug_out["state"] = state
        debug_out["vectors"] = state.vectors
        debug_out["display_vectors"] = state.display_vectors
        debug_out["texts"] = state.texts
    _LOG.info(
        "Junction done: %d vectors, %d texts (%.1fs)",
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
    raster = _compose_raster_layers(page_meta, debug_out.get("raster_images") or [])
    return prelude_debug_layers(page_meta, debug_out.get("raster_prelude")) + _page_layers(
        page_meta, raster, debug_out.get("state") or JunctionPageState()
    )


# ---------------------------------------------------------------------------
# Per-page accumulation
# ---------------------------------------------------------------------------


class _RasterSink:
    """Receives full-resolution debug rasters. Streaming: PNG-encode into one
    open single-page PDF per (stage, label) immediately, drop the array.
    Keep: retain `RasterItem`s for `debug_out`. Neither: ignore (and
    `active` tells the caller not to build the array at all)."""

    def __init__(self, *, stream: bool, keep: bool) -> None:
        self.stream = stream
        self.keep = keep
        self.kept: list[RasterItem] = []
        self._docs: dict[tuple[str, str], tuple[str, object]] = {}
        self.seconds = 0.0  # time spent placing / finishing streamed rasters

    @property
    def active(self) -> bool:
        return self.stream or self.keep


class _DebugSink(_RasterSink):
    """`_RasterSink` plus `debug`: whether any debug consumer exists at all
    (`debug_out` given, even without its raster arrays)."""

    def __init__(self, *, stream: bool, keep: bool, debug: bool) -> None:
        super().__init__(stream=stream, keep=keep)
        self.debug = debug

    def add(
        self, stage: str, label: str, hexcolor: str, array: np.ndarray, image: Image, page_meta
    ) -> None:
        if self.keep:
            self.kept.append(RasterItem(stage, label, hexcolor, array, image, array.shape[:2]))
        if self.stream:
            t0 = time.perf_counter()
            key = (stage, label)
            if key not in self._docs:
                self._docs[key] = (hexcolor, new_page_doc(page_meta))
            place_raster(self._docs[key][1][1], array, image)
            self.seconds += time.perf_counter() - t0

    def finish_stream(self, page_meta) -> "list[DebugLayer]":
        t0 = time.perf_counter()
        out: "list[DebugLayer]" = []
        for (stage, label), (hexcolor, doc) in self._docs.items():
            out.append((stage, label, hexcolor, _finish_doc(doc, page_meta)))
        self._docs.clear()
        self.seconds += time.perf_counter() - t0
        _LOG.info("Junction raster debug layers: %d (%.1fs encoding)", len(out), self.seconds)
        return out


def _compose_raster_layers(page_meta, items: list[RasterItem]) -> "list[DebugLayer]":
    docs: dict[tuple[str, str], tuple[str, object]] = {}
    for it in items:
        if it.stage.startswith("_"):  # raw diff-code canvases -- data, not a layer
            continue
        key = (it.stage, it.label)
        if key not in docs:
            docs[key] = (it.hexcolor, new_page_doc(page_meta))
        place_raster(docs[key][1][1], it.array, it.image)
    return [
        (s, l, hexcolor, _finish_doc(doc, page_meta)) for (s, l), (hexcolor, doc) in docs.items()
    ]


# ---------------------------------------------------------------------------
# Per-image flow
# ---------------------------------------------------------------------------
def _process_image(
    prepared: PreparedImage, page: Page, params: Params, state: JunctionPageState, sink: _DebugSink
) -> None:
    image = prepared.image
    layers = prepared.layers
    h, w = layers.labels.shape[:2]
    meta = page.meta
    to_page = prepared.to_page
    px_per_pt = prepared.px_per_pt
    _LOG.info("  %dx%d px, %.2f px/pt, %d colour layer(s)", w, h, px_per_pt, layers.n_layers)
    enhanced = to_gray(prepared.enhanced_rgb)
    if params.fill_min_width_px is None:
        params = replace(
            params, fill_min_width_px=max(FILL_MIN_WIDTH_PT * px_per_pt, FILL_MIN_WIDTH_FLOOR_PX)
        )

    # 5. per-layer, per-component tracing (+ 6. diff when debugging)
    tol_px = COMPONENT_TOLERANCE_PT * px_per_pt
    rendered_canvas = np.zeros((h, w), dtype=bool) if sink.active else None
    total_codes = np.zeros((h, w), dtype=np.uint8) if sink.active else None
    fill_canvas = np.zeros((h, w), dtype=bool) if sink.active else None
    for layer in layers.ink_layers():
        color_rgb = layers.colors_rgb[layer]
        color = tuple(float(c) / 255.0 for c in color_rgb)
        layer_codes = np.zeros((h, w), dtype=np.uint8) if sink.active else None
        hexcolor = rgb255_to_hex(color_rgb)
        t0 = time.perf_counter()
        n_vec = len(state.vectors)
        n_comp, comps = layer_components(layers.labels, enhanced, layer, tol_px)
        _LOG.debug("  trace layer %s: %d component(s)", hexcolor, n_comp)
        bar = tqdm(comps, total=n_comp, desc=f"Junction trace {hexcolor}", unit="comp", leave=False)
        for comp in bar:
            result = run(comp.gray, params)

            def mapper(pt, _x0=comp.x0, _y0=comp.y0):
                return to_page((pt[0] + _x0, pt[1] + _y0))

            first_seqno = state.seqno
            for pl in result.polylines:
                state.vectors.append(
                    _polyline_to_vector(pl, mapper, meta.index, state.next_seqno(), color)
                )
            for outline in result.outlines:
                state.vectors.append(
                    _outline_to_vector(outline, mapper, meta.index, state.next_seqno(), color)
                )
            if sink.debug:
                _accumulate_trace_geometry(
                    state,
                    result,
                    mapper,
                    (comp.x0, comp.y0),
                    comp.gray.shape,
                    to_page,
                    meta.index,
                    color,
                    first_seqno,
                )

            if sink.active:
                if result.fill is not None:
                    fr = fill_canvas[
                        comp.y0 : comp.y0 + result.fill.shape[0],
                        comp.x0 : comp.x0 + result.fill.shape[1],
                    ]
                    fr |= result.fill[: fr.shape[0], : fr.shape[1]]
                codes, rendered = diff_mod.diff_codes(
                    result.ink, result.polylines, DIFF_TOLERANCE_PX, result.outlines
                )
                diff_mod.paste_codes(layer_codes, codes, comp.x0, comp.y0)
                diff_mod.paste_codes(total_codes, codes, comp.x0, comp.y0)
                region = rendered_canvas[
                    comp.y0 : comp.y0 + rendered.shape[0], comp.x0 : comp.x0 + rendered.shape[1]
                ]
                region |= rendered[: region.shape[0], : region.shape[1]]
            bar.set_postfix(polylines=len(state.vectors) - n_vec, refresh=False)
        bar.close()
        _LOG.info(
            "  trace layer %s: %d component(s) -> %d polyline(s) (%.1fs)",
            hexcolor,
            n_comp,
            len(state.vectors) - n_vec,
            time.perf_counter() - t0,
        )
        if sink.active:
            sink.add(
                "vector_diff",
                f"layer {hexcolor}",
                hexcolor,
                diff_mod.codes_to_rgba(layer_codes),
                image,
                meta,
            )
            if sink.keep:
                sink.kept.append(
                    RasterItem("_codes", hexcolor, hexcolor, layer_codes, image, (h, w))
                )
            del layer_codes
    if sink.active:
        sink.add(
            "vector_render",
            "rendered vectors",
            _C_SEGMENT,
            diff_mod.rendered_to_rgba(rendered_canvas),
            image,
            meta,
        )
        sink.add("vector_diff", "total", _C_DIFF, diff_mod.codes_to_rgba(total_codes), image, meta)
        if fill_canvas.any():
            sink.add(
                "fills", "filled regions", _C_FILL, _mask_to_rgba(fill_canvas, _C_FILL), image, meta
            )
        if sink.keep:
            sink.kept.append(RasterItem("_codes", "total", _C_DIFF, total_codes, image, (h, w)))


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _polyline_to_vector(
    pl: Polyline, to_page: Mapper, page_index: int, seqno: int, color
) -> Vector:
    """One `Vector` per polyline: an `"l"` item per consecutive vertex pair."""
    pts = [to_page(p) for p in pl.points]
    return Vector(
        type="s",
        items=[("l", a, b) for a, b in zip(pts, pts[1:])],
        width=pl.width,
        rect=points_bbox(pts),
        **_base_vector_kwargs(page_index, seqno, color),
    )


def _outline_to_vector(
    outline: Outline, to_page: Mapper, page_index: int, seqno: int, color
) -> Vector:
    """One stroked (unfilled) `Vector` per fill region: every ring as its own
    closed run of `"l"` items (each ring repeats its first point, so no
    `closePath` is needed -- which would join the last ring to the first)."""
    items, allpts = [], []
    for ring in outline.rings:
        pts = [to_page(p) for p in ring]
        items.extend(("l", a, b) for a, b in zip(pts, pts[1:]))
        allpts.extend(pts)
    return Vector(
        type="s",
        items=items,
        width=1.0,
        rect=points_bbox(allpts),
        **_base_vector_kwargs(page_index, seqno, color),
    )


def _item_runs(v: Vector) -> list[list[tuple[float, float]]]:
    """A `Vector`'s `"l"` items as polylines, split wherever one item does not
    start where the previous one ended (a fill outline's separate rings)."""
    runs: list[list[tuple[float, float]]] = []
    for it in v.items:
        a, b = it[1], it[-1]
        if runs and runs[-1][-1] == a:
            runs[-1].append(b)
        else:
            runs.append([a, b])
    return runs


_CHAIN_DISPLAY_EPS_PX = 0.5


def _accumulate_trace_geometry(
    state: JunctionPageState,
    result,
    mapper: Mapper,
    offset,
    crop_shape,
    to_page: Mapper,
    page_index: int,
    color,
    first_seqno: int,
) -> None:
    """Debug only. Only cheap page-space tuples -- never `result`'s mask
    arrays. Graph nodes are the junctions; a chain end on no node is an
    endpoint. Raw chains are thinned with a sub-pixel Douglas-Peucker for
    display (PyMuPDF's `Shape` costs ~40-170 us per vertex). Output
    polylines are already fitted; they feed both the `vectorize` layer and
    `state.display_vectors` (same seqnos as the real vectors), followed by
    the fill outlines."""
    x0, y0 = offset
    ch, cw = crop_shape[:2]
    state.boxes["component"].append(points_bbox([to_page((x0, y0)), to_page((x0 + cw, y0 + ch))]))
    geom = state.geom
    chains = result.graph.chains
    for chain in chains:
        if len(chain) >= 2:
            geom["chains"].append([mapper(p) for p in fit_polyline(chain, _CHAIN_DISPLAY_EPS_PX)])
    geom["junctions"].extend(mapper(node) for node in result.graph.nodes)
    for chain, (a, b) in zip(chains, result.graph.ends):
        if len(chain) < 2 or chain[0] == chain[-1]:
            continue
        if a < 0:
            geom["endpoints"].append(mapper(chain[0]))
        if b < 0:
            geom["endpoints"].append(mapper(chain[-1]))
    pl_ends: set = set()
    for k, pl in enumerate(result.polylines):
        vec = _polyline_to_vector(pl, mapper, page_index, first_seqno + k, color)
        state.display_vectors.append(vec)
        geom["polylines"].append(
            [vec.items[0][1]] + [it[2] for it in vec.items] if vec.items else []
        )
        pl_ends.update((pl.points[0], pl.points[-1]))
    geom["polyline_ends"].extend(mapper(p) for p in pl_ends)
    for k, outline in enumerate(result.outlines):
        state.display_vectors.append(
            _outline_to_vector(
                outline, mapper, page_index, first_seqno + len(result.polylines) + k, color
            )
        )
        geom["outlines"].extend([mapper(p) for p in ring] for ring in outline.rings)


# ---------------------------------------------------------------------------
# Debug rendering -- this backend's own layers, built from `commons/renderer`
# primitives (boxes/text/vectors) plus embedded-PNG raster pages.
# ---------------------------------------------------------------------------
_C_DIFF = "#dc2626"
_C_COMPONENT = "#2563eb"
_C_GRAPH_CHAIN = "#0d9488"
_C_SEGMENT = "#059669"
_C_JUNCTION = "#e11d48"
_C_ENDPOINT = "#ca8a04"
_C_SEGMENT_END = "#1d4ed8"
_C_FILL = "#7c3aed"
_MASK_ALPHA = 140  # debug mask opacity, 0..255


def _mask_to_rgba(mask: np.ndarray, hexcolor: str) -> np.ndarray:
    out = np.zeros((*mask.shape, 4), dtype=np.uint8)
    out[mask, :3] = [int(c * 255) for c in hex_to_rgb01(hexcolor)]
    out[mask, 3] = _MASK_ALPHA
    return out


def _boxes_layer(page_meta, stage: str, label: str, hexcolor: str, boxes) -> DebugLayer:
    rgb = hex_to_rgb01(hexcolor)
    return (
        stage,
        label,
        hexcolor,
        render_specs_pdf(page_meta, [bbox_spec(b, rgb) for b in boxes], deflate=True),
    )


def _geometry_layer(
    page_meta,
    stage: str,
    label: str,
    hexcolor: str,
    *,
    polylines=(),
    points=(),
    width: float = 0.6,
    radius: float = 0.8,
) -> DebugLayer:
    """Polylines (stroked) and points (small filled dots), as stroke
    batches -- there are tens of thousands of them on a real page."""
    rgb = hex_to_rgb01(hexcolor)
    specs = [
        stroke_batch_spec([pl for pl in polylines if len(pl) >= 2], rgb, width=width),
        dot_batch_spec(points, rgb, radius=radius),
    ]
    return (stage, label, hexcolor, render_specs_pdf(page_meta, specs, deflate=True))


def vectors_layer_pdf(page_meta, vectors: "list[Vector]", *, width: float = 1.0) -> bytes:
    """Junction's own (all-`"l"`) vectors stroked in their own colours, one
    stroke batch per colour -- `render_vectors_pdf` replays them through a
    PyMuPDF `Shape`, which takes minutes on a full page of traced chains."""
    by_color: dict = {}
    for v in vectors:
        if v.items:
            by_color.setdefault(tuple(v.color or (0.0, 0.0, 0.0)), []).extend(_item_runs(v))
    return render_specs_pdf(
        page_meta,
        [stroke_batch_spec(pls, color, width=width) for color, pls in by_color.items()],
        deflate=True,
    )


def _page_layers(
    page_meta, raster_layers: "list[DebugLayer]", state: JunctionPageState
) -> "list[DebugLayer]":
    bx = state.boxes
    geom = state.geom
    out: "list[DebugLayer]" = [layer for layer in raster_layers if not layer[0].startswith("_")]
    builders = [
        lambda: _boxes_layer(
            page_meta, "components", "component bbox", _C_COMPONENT, bx["component"]
        ),
        lambda: _geometry_layer(
            page_meta, "graph_build", "chains", _C_GRAPH_CHAIN, polylines=geom["chains"]
        ),
        lambda: _geometry_layer(
            page_meta, "graph_build", "junctions", _C_JUNCTION, points=geom["junctions"]
        ),
        lambda: _geometry_layer(
            page_meta, "graph_build", "endpoints", _C_ENDPOINT, points=geom["endpoints"]
        ),
        lambda: _geometry_layer(
            page_meta, "vectorize", "polylines", _C_SEGMENT, polylines=geom["polylines"]
        ),
        lambda: _geometry_layer(
            page_meta,
            "vectorize",
            "polyline endpoints",
            _C_SEGMENT_END,
            points=geom["polyline_ends"],
        ),
        lambda: _geometry_layer(
            page_meta, "fills", "outlines", _C_FILL, polylines=geom["outlines"]
        ),
        lambda: (
            "final_vectors",
            "vectors",
            _C_SEGMENT,
            vectors_layer_pdf(page_meta, state.display_vectors or state.vectors),
        ),
    ]
    for build in builders:
        t0 = time.perf_counter()
        layer = build()
        _LOG.info("  debug layer %s/%s: %.2fs", layer[0], layer[1], time.perf_counter() - t0)
        out.append(layer)
    return out
