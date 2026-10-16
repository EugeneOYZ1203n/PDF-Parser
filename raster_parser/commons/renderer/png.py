"""Raster rendering of vector paths -- OCR input.

`render_vector_cluster` isolates a cluster of `Vector`s onto its own small PyMuPDF page and
rasterizes it. Each drawing's items are replayed as one composite path via
`_shapes.replay_drawing_paths`, so multi-contour filled glyphs keep their counters as holes.
The render is **black-and-white** (`monochrome=True`: every stroke / fill solid black, no
opacity or blend) with the real stroke widths -- colour belongs in the final page
(`P4_Output_Organization.render_output_pdf`), never in a detector's input.

No padding by default: the canvas is exactly the members' `union_bbox`, expanded only by an
explicit `padding` (PDF points, every side). Coordinates stay in unrotated page space; no page
rotation is applied (see `docs/ARCHITECTURE.md`, "Coordinate spaces").
"""
from __future__ import annotations

import pymupdf as fitz
from PIL import Image

from raster_parser.commons.helpers.geometry import PDF_POINTS_PER_INCH, union_bbox
from raster_parser.commons.models import Vector
from raster_parser.commons.renderer._shapes import replay_drawing_paths

# One `fitz.Document` reused across every render in this process instead of a fresh
# `fitz.open()` per call (a page can mean hundreds of cluster renders). Each call adds one page,
# renders it and deletes it, so the document never holds more than one page. Per process, so
# safe under a spawn pool.
_render_doc: "fitz.Document | None" = None

# `delete_page` only unlinks a page: its content stream and resources stay behind as orphaned
# xrefs. The shared document is therefore closed and reopened every `_RENDER_DOC_RECYCLE`
# renders, bounding what it can accumulate.
_RENDER_DOC_RECYCLE = 200
_render_doc_uses = 0

# Floor (PDF points) on a canvas side -- only ever applies to an axis whose extent is zero
# (a single horizontal rule); not padding.
_MIN_CANVAS_SIDE = 1.0


def _get_render_doc() -> "fitz.Document":
    """This process's shared, always-empty render document, recycled every
    `_RENDER_DOC_RECYCLE` calls."""
    global _render_doc, _render_doc_uses
    if _render_doc is not None and _render_doc_uses >= _RENDER_DOC_RECYCLE:
        _render_doc.close()
        _render_doc = None
    if _render_doc is None:
        _render_doc = fitz.open()
        _render_doc_uses = 0
    _render_doc_uses += 1
    return _render_doc


def _rasterize(page: "fitz.Page", dpi: int) -> "Image.Image":
    """`page` at `dpi` straight from the pixmap's samples, copied so the image outlives the
    page the caller deletes."""
    zoom = dpi / PDF_POINTS_PER_INCH
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.frombuffer(
        "RGB", (pixmap.width, pixmap.height), pixmap.samples, "raw", "RGB", 0, 1,
    ).copy()


def render_vector_cluster(
    vectors: list[Vector], dpi: int, padding: float = 0.0,
) -> "Image.Image":
    """Black-and-white render of `vectors` alone, on a canvas sized to their `union_bbox`
    plus `padding` points on every side, at `dpi`. Pixel (0, 0) is the canvas's top-left
    corner."""
    if not vectors:
        raise ValueError("render_vector_cluster requires at least one vector")

    x0, y0, x1, y1 = union_bbox([v.bbox for v in vectors])
    x0, y0, x1, y1 = x0 - padding, y0 - padding, x1 + padding, y1 + padding
    dx, dy = -x0, -y0
    width = max(x1 - x0, _MIN_CANVAS_SIDE)
    height = max(y1 - y0, _MIN_CANVAS_SIDE)

    doc = _get_render_doc()
    cluster_page = doc.new_page(width=width, height=height)
    try:
        replay_drawing_paths(cluster_page, vectors, dx=dx, dy=dy, monochrome=True)
        return _rasterize(cluster_page, dpi)
    finally:
        doc.delete_page(cluster_page.number)
