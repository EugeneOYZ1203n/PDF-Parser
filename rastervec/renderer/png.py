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

Coordinate space: everything here stays in unrotated MediaBox space, like
every other rastervec stage -- no page rotation is applied (see
`models.py`'s module docstring).
"""
from __future__ import annotations

import io

import pymupdf as fitz
from PIL import Image

from rastervec.helpers.geometry import union_bbox
from rastervec.models import PageMeta, VectorPath
from rastervec.renderer._shapes import replay_drawing_paths

# Minimum padding (in PDF points) added around a cluster's bbox before
# rendering, so thin strokes right at the bbox edge aren't clipped.
_MIN_CLUSTER_PADDING = 4.0


def _cluster_frame(paths: list[VectorPath]) -> tuple[float, float, float]:
    """The isolated-canvas geometry shared by `render_vector_cluster` and
    `pixel_to_page_bbox`: `(x0, y0, padding)` where `x0`/`y0` are the
    cluster's own bbox origin (page space) and `padding` is the margin
    added around it (>= `_MIN_CLUSTER_PADDING`, or a member's own stroke
    width if wider) so edge strokes aren't clipped. A page-space point
    maps to canvas space via `(x - x0 + padding, y - y0 + padding)`."""
    x0, y0, _x1, _y1 = union_bbox([p.bbox for p in paths])
    padding = max(_MIN_CLUSTER_PADDING, max((p.stroke_width or 0.0) for p in paths))
    return x0, y0, padding


def cluster_frame_size(paths: list[VectorPath]) -> tuple[float, float]:
    """(width, height) in PDF points of the isolated canvas
    `render_vector_cluster` would build for `paths` (bbox plus
    `_cluster_frame`'s padding) -- lets a caller (`RenderOCR.ocr_cluster`)
    pick a dpi that keeps the rendered pixel size above some minimum
    without duplicating `_cluster_frame`'s own padding math."""
    x0, y0, x1, y1 = union_bbox([p.bbox for p in paths])
    _frame_x0, _frame_y0, padding = _cluster_frame(paths)
    return (x1 - x0) + 2 * padding, (y1 - y0) + 2 * padding


def render_vector_cluster(paths: list[VectorPath], dpi: int) -> "Image.Image":
    """High-resolution render of an isolated vector path cluster, used as
    OCR input. Builds a fresh single-page PyMuPDF document sized to the
    cluster's own bbox (plus padding for stroke overflow, via
    `_cluster_frame`), replays each drawing's items as one composite path
    via `_shapes.replay_drawing_paths`, then rasterizes at `dpi` -- reusing
    PyMuPDF's own rendering rather than re-implementing curve/fill
    rasterization by hand."""
    if not paths:
        raise ValueError("render_vector_cluster requires at least one path")

    x0, y0, x1, y1 = union_bbox([p.bbox for p in paths])
    frame_x0, frame_y0, padding = _cluster_frame(paths)
    dx, dy = padding - frame_x0, padding - frame_y0
    width = (x1 - x0) + 2 * padding
    height = (y1 - y0) + 2 * padding

    doc = fitz.open()
    try:
        cluster_page = doc.new_page(width=width, height=height)
        shape = cluster_page.new_shape()
        replay_drawing_paths(shape, paths, dx=dx, dy=dy)
        shape.commit()

        zoom = dpi / 72.0
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
    x0, y0, padding = _cluster_frame(paths)
    zoom = dpi / 72.0
    xs = [px / zoom - padding + x0 for px, _py in pixel_points]
    ys = [py / zoom - padding + y0 for _px, py in pixel_points]
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

        zoom = dpi / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        image.load()
        return image
    finally:
        doc.close()
