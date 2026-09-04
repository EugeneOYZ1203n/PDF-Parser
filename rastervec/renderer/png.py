"""PNG / pixmap rendering of vector paths -- the pipeline's OCR and FAST
detection input.

`render_vector_cluster` isolates a text-candidate cluster onto its own
small PyMuPDF page and rasterizes it (OCR input); `render_page_paths` is
the whole-page counterpart (FAST detection input). Both replay each
drawing's items as one composite path via
`_shapes.replay_drawing_paths`, so multi-contour filled glyphs render with
their counters as holes rather than filled solid. `pixel_to_page_bbox`
inverts `render_vector_cluster`'s own isolated-canvas transform (shared via
`_cluster_frame`) to map a detected-in-pixel-space bbox back into PDF page
space.

`_cluster_frame`'s border around the cluster's own bbox is asymmetric --
tight vertically, generous horizontally -- ported from archive/raster_parser's
Type-2 full native-to-OCR pipeline (`parsing/parser.py::normalise_crop_for_ocr`,
used by `archive/scripts/type2_full_native_to_ocr_pipeline.py`), which padded
an already-rasterized crop with `cv2.copyMakeBorder` before resizing it for
PaddleOCR. Here the same ratios are applied to the render frame itself, in
PDF-point space, before the page is even rasterized -- the border is part of
the render, not a separate post-render pixel-fill step.

Coordinate space: everything here stays in unrotated MediaBox space, like
every other rastervec stage -- no page rotation is applied (see
`models.py`'s module docstring).
"""
from __future__ import annotations

import io

import pymupdf as fitz
from PIL import Image

from rastervec.config import (
    MIN_CLUSTER_PADDING,
    OCR_HORIZONTAL_PADDING_FRACTION,
    OCR_VERTICAL_PADDING_FRACTION,
)
from rastervec.helpers.geometry import PDF_POINTS_PER_INCH, union_bbox
from rastervec.models import PageMeta, VectorPath
from rastervec.renderer._shapes import replay_drawing_paths


def _cluster_frame(paths: list[VectorPath]) -> tuple[float, float, float, float]:
    """The isolated-canvas geometry shared by `render_vector_cluster` and
    `pixel_to_page_bbox`: `(x0, y0, pad_x, pad_y)` where `x0`/`y0` are the
    cluster's own bbox origin (page space) and `pad_x`/`pad_y` are the
    (asymmetric -- see module docstring) margins added around it, each
    >= `MIN_CLUSTER_PADDING` or a member's own stroke width if wider. A
    page-space point maps to canvas space via
    `(x - x0 + pad_x, y - y0 + pad_y)`."""
    x0, y0, _x1, y1 = union_bbox([p.bbox for p in paths])
    height = y1 - y0
    stroke_floor = max(MIN_CLUSTER_PADDING, max((p.stroke_width or 0.0) for p in paths))
    pad_y = max(stroke_floor, height * OCR_VERTICAL_PADDING_FRACTION)
    pad_x = max(stroke_floor, height * OCR_HORIZONTAL_PADDING_FRACTION)
    return x0, y0, pad_x, pad_y


def cluster_frame_size(paths: list[VectorPath]) -> tuple[float, float]:
    """(width, height) in PDF points of the isolated canvas
    `render_vector_cluster` would build for `paths` (bbox plus
    `_cluster_frame`'s asymmetric padding) -- lets a caller
    (`RenderOCR.ocr_cluster`) pick a dpi that keeps the rendered pixel size
    above some minimum without duplicating `_cluster_frame`'s own padding
    math."""
    x0, y0, x1, y1 = union_bbox([p.bbox for p in paths])
    _frame_x0, _frame_y0, pad_x, pad_y = _cluster_frame(paths)
    return (x1 - x0) + 2 * pad_x, (y1 - y0) + 2 * pad_y


def render_vector_cluster(paths: list[VectorPath], dpi: int) -> "Image.Image":
    """High-resolution render of an isolated vector path cluster, used as
    OCR input. Builds a fresh single-page PyMuPDF document sized to the
    cluster's own bbox (plus the asymmetric OCR border, via
    `_cluster_frame`), replays each drawing's items as one composite path
    via `_shapes.replay_drawing_paths`, then rasterizes at `dpi` -- reusing
    PyMuPDF's own rendering rather than re-implementing curve/fill
    rasterization by hand."""
    if not paths:
        raise ValueError("render_vector_cluster requires at least one path")

    x0, y0, x1, y1 = union_bbox([p.bbox for p in paths])
    frame_x0, frame_y0, pad_x, pad_y = _cluster_frame(paths)
    dx, dy = pad_x - frame_x0, pad_y - frame_y0
    width = (x1 - x0) + 2 * pad_x
    height = (y1 - y0) + 2 * pad_y

    doc = fitz.open()
    try:
        cluster_page = doc.new_page(width=width, height=height)
        shape = cluster_page.new_shape()
        replay_drawing_paths(shape, paths, dx=dx, dy=dy)
        shape.commit()

        zoom = dpi / PDF_POINTS_PER_INCH
        pixmap = cluster_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        image.load()  # decode now -- the backing BytesIO doesn't outlive this call
        return image
    finally:
        doc.close()


def pixel_to_page_bbox(
    paths: list[VectorPath],
    dpi: int,
    pixel_points: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    """Inverts `render_vector_cluster`'s own (dx, dy, zoom) transform to
    map a set of pixel-space points (e.g. Paddle's detected polygon
    corners, from a render of this exact `paths`/`dpi` pair) back into PDF
    page space, returning their bbox. Used by `RenderOCR.ocr_cluster` to
    compute a `TextVectorResult`'s ocr_bbox."""
    x0, y0, pad_x, pad_y = _cluster_frame(paths)
    zoom = dpi / PDF_POINTS_PER_INCH
    xs = [px / zoom - pad_x + x0 for px, _py in pixel_points]
    ys = [py / zoom - pad_y + y0 for _px, py in pixel_points]
    return (min(xs), min(ys), max(xs), max(ys))


def render_page_paths(
    paths: list[VectorPath], page_meta: PageMeta, dpi: int
) -> "Image.Image":
    """Renders a whole page-sized image containing only `paths` (no native
    text, no dropped/drawing content) -- used as FAST's own detection
    input: `pipeline.py`'s fast_text_detect stage draws every surviving
    vector-classification cluster's paths onto one shared page-sized canvas
    via this function, so FAST scans the whole page in a single pass
    instead of many small per-cluster collages. Stays in unrotated MediaBox
    space like every other stage (no rotation applied), consistent with
    `render_vector_cluster`."""
    doc = fitz.open()
    try:
        page = doc.new_page(width=page_meta.width, height=page_meta.height)
        shape = page.new_shape()
        replay_drawing_paths(shape, paths)
        shape.commit()

        zoom = dpi / PDF_POINTS_PER_INCH
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        image.load()
        return image
    finally:
        doc.close()
