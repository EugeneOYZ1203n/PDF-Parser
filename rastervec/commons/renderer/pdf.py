"""One-layer debug-PDF wrappers over `draw.py`'s spec -> draw primitives.

Each function builds specs for one kind of mark and hands them to
`draw.render_specs_pdf` -- none of them place geometry themselves, so every
debug layer and the final output (`P4_Output_Organization/render.py`)
share one rotation/translation/scale implementation.

- `render_boxes_pdf` -- axis-aligned bbox outlines (`bbox_spec`).
- `render_quads_pdf` -- closed polygons, e.g. PaddleOCR's rotated detect
  quads (`quad_spec`).
- `render_text_pdf` -- `Text`s and/or `(text, bbox, rotation[, rgb])`
  boxes, rotated then scaled to fit their own box (`text_spec`).
- `render_vectors_pdf` -- `Vector`s replayed as recoloured strokes
  (`vector_spec`).
- `rasterize_pdf` -- page 0 of PDF bytes -> PIL image (previews).

The final reconstructed page (native + OCR text over drawing vectors in
their real paint) is not built here -- that is Phase 4's job
(`P4_Output_Organization.render_output_pdf`).
"""
from __future__ import annotations

import io

import pymupdf as fitz
from PIL import Image

from rastervec.commons.models import PageMeta, Text, Vector
from rastervec.commons.renderer.draw import (
    TextBox,
    bbox_spec,
    quad_spec,
    render_specs_pdf,
    text_spec,
    vector_spec,
)

# One box overlay: a page-space bbox, an (r, g, b) 0..1 outline color, and
# an optional PyMuPDF dash string (e.g. "[4 3] 0"; omit or None = solid).
BoxOverlay = (
    tuple[tuple[float, float, float, float], tuple[float, float, float]]
    | tuple[tuple[float, float, float, float], tuple[float, float, float], "str | None"]
)

__all__ = [
    "BoxOverlay",
    "TextBox",
    "rasterize_pdf",
    "render_boxes_pdf",
    "render_quads_pdf",
    "render_text_pdf",
    "render_vectors_pdf",
]


def render_boxes_pdf(page_meta: PageMeta, boxes: "list[BoxOverlay]", *, width: float = 1.5) -> bytes:
    """Each `(bbox, rgb[, dashes])` entry as an unfilled rectangle outline,
    on a fresh page sized/rotated to `page_meta`."""
    return render_specs_pdf(page_meta, [
        bbox_spec(spec[0], spec[1], width=width, dashes=spec[2] if len(spec) > 2 else None)
        for spec in boxes
    ])


def render_quads_pdf(page_meta: PageMeta, quads: list, *, width: float = 1.5) -> bytes:
    """Each `(points, rgb[, dashes])` entry as a closed, unfilled polygon --
    for genuinely rotated boxes (detect quads) an axis-aligned bbox can't
    show."""
    return render_specs_pdf(page_meta, [
        quad_spec(spec[0], spec[1], width=width, dashes=spec[2] if len(spec) > 2 else None)
        for spec in quads if spec[0] is not None and len(spec[0]) >= 3
    ])


def render_text_pdf(page_meta: PageMeta, texts: "list[Text | TextBox]", *, color_of=None) -> bytes:
    """Every item placed by `draw.text_spec` (rotated to its angle, scaled
    to its own box). `color_of(item) -> (r, g, b)` sets each item's colour;
    without it a `Text` draws black and a `TextBox` uses its own 4th
    element (default black)."""
    specs = []
    for item in texts:
        if color_of is not None:
            color = color_of(item)
        elif isinstance(item, tuple):
            color = None
        else:
            color = (0.0, 0.0, 0.0)
        specs.append(text_spec(item, color=color))
    return render_specs_pdf(page_meta, specs)


def render_vectors_pdf(
    page_meta: PageMeta, vectors: "list[Vector]", *, color_of=None, width: float = 1.0,
) -> bytes:
    """Every `Vector` replayed as a stroked path in `color_of(vector) ->
    (r, g, b)` (default black), its real fill/dashes/blend dropped so a
    stage colouring isn't drowned out."""
    return render_specs_pdf(page_meta, [
        vector_spec(v, recolor=color_of(v) if color_of is not None else (0.0, 0.0, 0.0), min_width=width)
        for v in vectors
    ])


def rasterize_pdf(pdf_bytes: bytes, *, zoom: float = 1.0) -> "Image.Image":
    """Page 0 of `pdf_bytes` rasterized at `zoom` (display space, `/Rotate`
    applied -- pixel-comparable with the source page's own pixmap)."""
    doc = fitz.open("pdf", pdf_bytes)
    try:
        pixmap = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        image.load()
        return image
    finally:
        doc.close()
