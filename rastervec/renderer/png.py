"""PNG / pixmap rendering of vector paths -- the pipeline's OCR and FAST
detection input.

`render_vector_cluster` isolates a text-candidate cluster onto its own
small PyMuPDF page and rasterizes it (OCR input); `render_page_paths` is
the whole-page counterpart (FAST detection input). Both replay each
drawing's items as one composite path via
`_shapes.replay_drawing_paths`, so multi-contour filled glyphs render with
their counters as holes rather than filled solid. `pixel_to_page_bbox`
inverts `render_vector_cluster`'s own isolated-canvas transform to map a
detected-in-pixel-space bbox back into PDF page space.

**No padding happens here.** A cluster render's canvas is exactly the
members' `union_bbox` -- nothing added on any side. Every margin the OCR
path needs (breathing room around glyphs, clipping slack) is one explicit
pixel-space step in `OCR/radon.py::pad_image`, applied by `segment_clusters`
after this render, so a reader can see the padding happen instead of
inferring it from a renderer's internals. Don't reintroduce a border here:
`pixel_to_page_bbox`/`page_points_to_pixel` assume the bbox *is* the frame.

Coordinate space: everything here stays in unrotated MediaBox space, like
every other rastervec stage -- no page rotation is applied (see
`models.py`'s module docstring).
"""
from __future__ import annotations

import pymupdf as fitz
from PIL import Image

from rastervec.helpers.geometry import PDF_POINTS_PER_INCH, union_bbox
from rastervec.models import PageMeta, Vector
from rastervec.renderer._shapes import replay_drawing_paths

# One `fitz.Document` reused across every render_vector_cluster/
# render_page_paths call in this process, instead of a fresh fitz.open()
# per call -- render_vector_cluster in particular is called once per
# text-candidate cluster (Radon segmentation) and again per deduped unique
# segment (OCR), so a real page can mean hundreds of calls; open/close and
# font-table setup on a brand-new document each time is pure overhead. Each
# call adds exactly one page, renders it, then deletes it (never closes the
# document), so the doc never holds more than one page at a time. Per-process
# (like `FastDetector`'s/`PaddleRecBackend`'s own model caches), safe under
# Pool-2 multiprocessing since each worker process gets its own.
_render_doc: "fitz.Document | None" = None

# Floor (PDF points) on a cluster canvas's own width/height -- see
# `render_vector_cluster`. Not padding: it only ever applies to an axis
# whose extent is genuinely zero.
_MIN_CANVAS_SIDE = 1.0


def _get_render_doc() -> "fitz.Document":
    global _render_doc
    if _render_doc is None:
        _render_doc = fitz.open()
    return _render_doc


def _rasterize(page: "fitz.Page", dpi: int) -> "Image.Image":
    """Renders `page` at `dpi` directly from its pixmap's raw samples --
    no PNG encode/decode round-trip -- and detaches the result via
    `.copy()` so the returned image outlives the pixmap/page (which the
    caller's `finally` immediately deletes from the shared render doc)."""
    zoom = dpi / PDF_POINTS_PER_INCH
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.frombuffer(
        "RGB", (pixmap.width, pixmap.height), pixmap.samples, "raw", "RGB", 0, 1,
    ).copy()


def render_vector_cluster(vectors: list[Vector], dpi: int) -> "Image.Image":
    """High-resolution render of an isolated vector cluster, used as OCR
    input. Adds one page -- sized to the members' `union_bbox` exactly, no
    border of any kind (see the module docstring) -- to this process's
    shared render document (see `_get_render_doc`: called at real-pipeline
    scale, once per text-candidate cluster and again per deduped unique
    segment, so a fresh `fitz.open()` per call would be pure overhead),
    replays each Vector's items as one composite path via
    `_shapes.replay_drawing_paths`, then rasterizes at `dpi` -- reusing
    PyMuPDF's own rendering rather than re-implementing curve/fill
    rasterization by hand."""
    if not vectors:
        raise ValueError("render_vector_cluster requires at least one vector")

    x0, y0, x1, y1 = union_bbox([v.bbox for v in vectors])
    dx, dy = -x0, -y0
    # Degeneracy guard, not padding: a flat cluster (a single horizontal
    # rule) has zero extent on one axis, and a 0-pt page rasterizes to a
    # 0-px pixmap that `Image.frombuffer` cannot build.
    width = max(x1 - x0, _MIN_CANVAS_SIDE)
    height = max(y1 - y0, _MIN_CANVAS_SIDE)

    doc = _get_render_doc()
    cluster_page = doc.new_page(width=width, height=height)
    try:
        replay_drawing_paths(cluster_page, vectors, dx=dx, dy=dy)
        return _rasterize(cluster_page, dpi)
    finally:
        doc.delete_page(cluster_page.number)


def pixel_to_page_bbox(
    vectors: list[Vector],
    dpi: int,
    pixel_points: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    """Inverts `render_vector_cluster`'s own (dx, dy, zoom) transform to
    map a set of pixel-space points (e.g. a segmented word's corners, from
    a render of this exact `vectors`/`dpi` pair) back into PDF page space,
    returning their bbox. Pixel `(0, 0)` is the cluster's own bbox origin,
    since the render carries no border.

    A caller working in a *padded* copy of that render (`OCR/radon.py::
    segment_clusters`) must subtract `pad_image`'s own returned pixel
    offset before calling this."""
    x0, y0, _x1, _y1 = union_bbox([v.bbox for v in vectors])
    zoom = dpi / PDF_POINTS_PER_INCH
    xs = [px / zoom + x0 for px, _py in pixel_points]
    ys = [py / zoom + y0 for _px, py in pixel_points]
    return (min(xs), min(ys), max(xs), max(ys))


def page_points_to_pixel(
    vectors: list[Vector],
    dpi: int,
    page_points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Map page-space points into the pixel space of
    `render_vector_cluster(vectors, dpi)` -- the exact inverse of
    `pixel_to_page_bbox` (same bbox origin + `zoom`). Lets the
    visualization notebook draw OCR's returned page-space boxes back
    onto the rendered cluster image the backend actually saw."""
    x0, y0, _x1, _y1 = union_bbox([v.bbox for v in vectors])
    zoom = dpi / PDF_POINTS_PER_INCH
    return [((px - x0) * zoom, (py - y0) * zoom) for px, py in page_points]


def render_page_paths(
    vectors: list[Vector], page_meta: PageMeta, dpi: int
) -> "Image.Image":
    """Renders a whole page-sized image containing only `vectors` (no
    native text, no dropped/drawing content) -- used as FAST's own
    detection input: the `fast` pipeline step draws every surviving
    vector-classification Vector onto one shared page-sized canvas via this
    function, so FAST scans the whole page in a single pass instead of many
    small per-cluster collages. Stays in unrotated MediaBox space like
    every other stage (no rotation applied), consistent with
    `render_vector_cluster`."""
    doc = _get_render_doc()
    page = doc.new_page(width=page_meta.width, height=page_meta.height)
    try:
        replay_drawing_paths(page, vectors)
        return _rasterize(page, dpi)
    finally:
        doc.delete_page(page.number)
