"""Conversion: turns a page's native text into vector-drawn text (real
`get_drawings()` path content, not embedded raster), so a PDF with known
native text becomes a known-answer test case for Vector_Classification --
the ground truth text is whatever the original native words were, and this
module's job is only to change how that text is *drawn*.

Verified API sequence (spike run against installed pymupdf 1.28.2, see
`tests/rastervec/Evaluation/Conversion/test_conversion.py`):

1. `fitz_page.get_svg_image()` renders the page to an SVG string. For a page
   with native text, PyMuPDF emits each glyph as an SVG `<path>` (filled
   glyph outline) referencing a `<defs>`-declared glyph path, never an SVG
   `<text>` element and never an embedded raster `<image>` -- confirmed by
   checking the returned SVG string for `<text` (absent) and `<path`
   (present), and the round-tripped PDF's `get_images()` (empty). This is
   exactly the "text-as-filled-vector-paths" shape this project's
   Vector_Classification pipeline is built to reconstruct.
2. `fitz.open(stream=svg.encode("utf-8"), filetype="svg")` opens that SVG
   string as a one-page pymupdf document.
3. `svg_doc.convert_to_pdf()` converts it to real PDF content bytes --
   confirmed the result's `get_drawings()` returns real drawing dicts (with
   `items`, `seqno`, `lineCap`, etc.) and `get_text()` returns "" (no native
   text objects survive the round-trip, only vector paths).
4. The converted single-page PDF's page rect already matches the source
   page's width/height exactly (confirmed empirically) since `get_svg_image`
   encodes the page's own pixel dimensions into the SVG `viewBox`/width/
   height attributes that `convert_to_pdf` reads back. This module still
   places it explicitly via `new_page` + `show_pdf_page` onto a page sized
   from the source's own `PageMeta.mediabox`/`rotation` rather than relying
   on that coincidence, so a page with a non-origin mediabox or a rotation
   still comes out correct.
"""
from __future__ import annotations

import pymupdf as fitz

from rastervec.logging_setup import get_logger
from rastervec.Reader.reader import Reader

_LOG = get_logger("conversion")


def convert_page_to_vector_text(
    pdf_path: str, page_index: int, output_path: str | None = None,
) -> bytes:
    """Renders `pdf_path`'s page `page_index` to SVG and converts it back
    into a new one-page PDF whose text is drawn as vector paths instead of
    native text objects. Returns the new PDF's bytes; also writes them to
    `output_path` if given."""
    with Reader(pdf_path) as reader:
        page = reader.get_page(page_index)
        meta = page.meta
        svg = page.fitz_page.get_svg_image()

    svg_doc = fitz.open(stream=svg.encode("utf-8"), filetype="svg")
    pdf_bytes = svg_doc.convert_to_pdf()
    converted = fitz.open("pdf", pdf_bytes)

    x0, y0, x1, y1 = meta.mediabox
    out = fitz.open()
    new_page = out.new_page(width=x1 - x0, height=y1 - y0)
    new_page.set_rotation(meta.rotation)
    new_page.show_pdf_page(new_page.rect, converted, 0)

    result = out.tobytes()
    _LOG.debug(
        "converted page %d of %s to vector text (%d bytes)",
        page_index, pdf_path, len(result),
    )

    if output_path is not None:
        with open(output_path, "wb") as f:
            f.write(result)

    return result
