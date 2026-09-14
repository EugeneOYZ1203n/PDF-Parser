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


def extract(images: list[Image], page: Page, *, params: Params | None = None) -> tuple[list[Vector], list[Text]]:
    """Runs the classical raster-to-vector pipeline on every given `Image`
    and returns every resulting line/arc primitive as a `Vector`. Returns
    no `Text` -- Phase 3 backends OCR the combined vector pool themselves."""
    params = params or Params(run_ocr=False)
    vectors: list[Vector] = []
    seqno = 0
    for image in images:
        gray = _to_gray(image.array)
        if gray.size == 0:
            continue
        result = run(gray, params)
        h_result, w_result = result.gray.shape[:2]
        if h_result == 0 or w_result == 0:
            continue

        def to_page(pt: tuple[float, float]) -> tuple[float, float]:
            x0, y0, x1, y1 = image.bbox
            px, py = pt
            fx, fy = px / w_result, py / h_result
            return (x0 + fx * (x1 - x0), y0 + fy * (y1 - y0))

        for seg in result.segments:
            vectors.append(_segment_to_vector(seg, to_page, page.meta.index, seqno))
            seqno += 1
        for arc in result.arcs:
            vectors.append(_arc_to_vector(arc, to_page, page.meta.index, seqno))
            seqno += 1

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
