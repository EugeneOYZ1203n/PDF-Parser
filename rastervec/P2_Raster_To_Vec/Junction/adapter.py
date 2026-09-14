"""Junction's Phase2Backend entrypoint -- wraps the ported classical
(no-CNN) raster-to-vector pipeline (`junction_test/pipeline.py::run`, from
branch `junction-classification`) for every `Image` Phase 1 hands over,
translating its pixel-space `Segment`/`Arc` primitives into
`commons.models.Vector`s in real page space.

Coordinate mapping: `pipeline.run` may internally downscale `gray` (see its
own `Params.max_work_px`) before analysing it, so segment/arc coordinates
live in `result.gray`'s own pixel frame, not the input `Image.array`'s.
Mapping back to page space uses `result.gray.shape` against the `Image`'s
own `bbox` directly (a fraction-of-frame mapping) -- dpi-agnostic, and
correct regardless of whatever internal downscale ratio the pipeline chose.

OCR text-box extraction inside `junction_test` (`Params.run_ocr`) stays
off -- Phase 3 OCRs the merged vector set for real; wiring that dead path
up is explicitly deferred (see the repo's refactor plan)."""
from __future__ import annotations

from rastervec.commons.models import Image, Page, Text, Vector
from rastervec.P2_Raster_To_Vec.Junction.junction_test.pipeline import Params, run
from rastervec.P2_Raster_To_Vec.Junction.junction_test.types_ import Arc, Segment

_ITEM_COLOR = (0.0, 0.0, 0.0)


def extract(
    images: list[Image], page: Page, *, params: Params | None = None,
    debug_out: "dict | None" = None,
) -> tuple[list[Vector], list[Text]]:
    """Runs the classical raster-to-vector pipeline on every given `Image`
    and returns every resulting line/arc primitive as a `Vector`. Returns
    no `Text` -- Phase 3 backends OCR the combined vector pool themselves.
    When `debug_out` is given (a plain dict), each image's own
    `junction_test.types_.PipelineResult` is stashed (with its `to_page`
    mapper) for `render_debug` to read back and visualise."""
    params = params or Params(run_ocr=False)
    vectors: list[Vector] = []
    seqno = 0
    per_image: "list[tuple[object, object]]" = []
    for image in images:
        gray = _to_gray(image.array)
        if gray.size == 0:
            continue
        result = run(gray, params)
        h_result, w_result = result.gray.shape[:2]
        if h_result == 0 or w_result == 0:
            continue

        def to_page(pt: tuple[float, float], _bbox=image.bbox, _h=h_result, _w=w_result) -> tuple[float, float]:
            x0, y0, x1, y1 = _bbox
            px, py = pt
            fx, fy = px / _w, py / _h
            return (x0 + fx * (x1 - x0), y0 + fy * (y1 - y0))

        for seg in result.segments:
            vectors.append(_segment_to_vector(seg, to_page, page.meta.index, seqno))
            seqno += 1
        for arc in result.arcs:
            vectors.append(_arc_to_vector(arc, to_page, page.meta.index, seqno))
            seqno += 1

        if debug_out is not None:
            per_image.append((result, to_page))

    if debug_out is not None:
        debug_out["per_image"] = per_image
        debug_out["vectors"] = vectors

    return vectors, []


def _to_gray(array) -> "object":
    import numpy as np

    arr = np.asarray(array)
    if arr.ndim == 2:
        return arr.astype("uint8")
    # simple luminance -- avoids any extra dependency for a plain RGB->gray
    return (arr[..., :3].astype("float32") @ [0.299, 0.587, 0.114]).astype("uint8")


def _base_vector_kwargs(page_index: int, seqno: int) -> dict:
    return dict(
        color=_ITEM_COLOR, fill=None, dashes=None, closePath=False,
        lineCap=0, lineJoin=0, even_odd=False, stroke_opacity=1.0, fill_opacity=1.0,
        layer=None, scissor=None, blendmode=None, isolated=False, knockout=False,
        opacity=1.0, page_index=page_index, seqno=seqno,
    )


def _bbox_of(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _segment_to_vector(seg: Segment, to_page, page_index: int, seqno: int) -> Vector:
    p0, p1 = to_page(seg.p0), to_page(seg.p1)
    return Vector(
        type="s", items=[("l", p0, p1)], width=seg.width, rect=_bbox_of([p0, p1]),
        **_base_vector_kwargs(page_index, seqno),
    )


def _arc_to_vector(arc: Arc, to_page, page_index: int, seqno: int) -> Vector:
    points = [to_page(p) for p in arc.polyline] or [to_page(arc.center)]
    items = [("l", points[i], points[i + 1]) for i in range(len(points) - 1)]
    if not items:
        items = [("l", points[0], points[0])]
    return Vector(
        type="s", items=items, width=arc.width, rect=_bbox_of(points),
        **_base_vector_kwargs(page_index, seqno),
    )


# ---------------------------------------------------------------------------
# Debug rendering -- this backend's own render function over its own
# `debug_out` shape, built only from `commons/renderer`'s generic box
# primitive (the earliest raster-processing stages here are numpy masks,
# not vector/text geometry, so they're rendered coarsely as their own
# ink-bbox box -- deliberately coarse per this task's scope; graph_stages
# onward render each chain/junction's own bbox, still box-based rather than
# true polylines to keep this module's own render function simple). Nothing
# here is shared with any P3 backend or P2_Raster_To_Vec/Stub.
# ---------------------------------------------------------------------------
_C_TEXT_MASK = "#f97316"
_C_GRAPHICS_MASK = "#2563eb"
_C_SKELETON = "#7c3aed"
_C_GRAPH_CHAIN = "#0d9488"
_C_JUNCTION = "#dc2626"
_C_SEGMENT = "#059669"
_C_ARC = "#c026d3"


def _hex_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def _mask_bbox(mask, to_page) -> "tuple[float, float, float, float] | None":
    import numpy as np

    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    p0 = to_page((float(xs.min()), float(ys.min())))
    p1 = to_page((float(xs.max()), float(ys.max())))
    return _bbox_of([p0, p1])


def render_debug(debug_out: "dict | None", page_meta) -> "list[tuple[str, str, str, bytes]]":
    """One (stage, label, hex, pdf_bytes) tuple per debug layer, pooling
    across every image `extract()` processed for this page, built straight
    from `parse()`'s own `debug_out` stash."""
    from rastervec.commons.renderer import render_boxes_pdf, render_vectors_pdf

    out: "list[tuple[str, str, str, bytes]]" = []
    if not debug_out:
        return out

    per_image = debug_out.get("per_image") or []

    text_boxes, graphics_boxes, skeleton_boxes = [], [], []
    chain_boxes, junction_boxes, segment_boxes, arc_boxes = [], [], [], []
    for result, to_page in per_image:
        b = _mask_bbox(result.text_mask, to_page)
        if b is not None:
            text_boxes.append(b)
        b = _mask_bbox(result.graphics_mask, to_page)
        if b is not None:
            graphics_boxes.append(b)
        b = _mask_bbox(result.skeleton, to_page)
        if b is not None:
            skeleton_boxes.append(b)
        for chain in result.graph.chains:
            if not chain:
                continue
            pts = [to_page(p) for p in chain]
            chain_boxes.append(_bbox_of(pts))
        for junction in getattr(result, "junctions", []):
            cx, cy = to_page(junction.xy)
            junction_boxes.append((cx - 2.0, cy - 2.0, cx + 2.0, cy + 2.0))
        for seg in result.segments:
            p0, p1 = to_page(seg.p0), to_page(seg.p1)
            segment_boxes.append(_bbox_of([p0, p1]))
        for arc in result.arcs:
            pts = [to_page(p) for p in arc.polyline] or [to_page(arc.center)]
            arc_boxes.append(_bbox_of(pts))

    out.append(("text_graphics_separation", "text mask bbox", _C_TEXT_MASK, render_boxes_pdf(
        page_meta, [(b, _hex_rgb(_C_TEXT_MASK)) for b in text_boxes],
    )))
    out.append(("text_graphics_separation", "graphics mask bbox", _C_GRAPHICS_MASK, render_boxes_pdf(
        page_meta, [(b, _hex_rgb(_C_GRAPHICS_MASK)) for b in graphics_boxes],
    )))
    out.append(("skeletonize", "skeleton bbox", _C_SKELETON, render_boxes_pdf(
        page_meta, [(b, _hex_rgb(_C_SKELETON)) for b in skeleton_boxes],
    )))
    out.append(("graph_build", "chain bbox", _C_GRAPH_CHAIN, render_boxes_pdf(
        page_meta, [(b, _hex_rgb(_C_GRAPH_CHAIN)) for b in chain_boxes],
    )))
    out.append(("graph_build", "junction", _C_JUNCTION, render_boxes_pdf(
        page_meta, [(b, _hex_rgb(_C_JUNCTION)) for b in junction_boxes],
    )))
    out.append(("polyline_fit", "segment bbox", _C_SEGMENT, render_boxes_pdf(
        page_meta, [(b, _hex_rgb(_C_SEGMENT)) for b in segment_boxes],
    )))
    out.append(("arc_detect", "arc bbox", _C_ARC, render_boxes_pdf(
        page_meta, [(b, _hex_rgb(_C_ARC)) for b in arc_boxes],
    )))

    vectors = debug_out.get("vectors") or []
    out.append(("final_vectors", "vectors", _C_SEGMENT, render_vectors_pdf(
        page_meta, vectors, color_of=lambda _v: _hex_rgb(_C_SEGMENT),
    )))

    return out
