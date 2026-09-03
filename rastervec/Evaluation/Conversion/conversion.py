"""Conversion: adds a page's native text back as vector-drawn text (real
`get_drawings()` path content, not embedded raster), so a PDF with known
native text becomes a known-answer test case for Vector_Classification --
the ground truth text is whatever the original native words were, and this
module's job is only to change how that text is *drawn*.

**Pre-existing vector content is preserved byte-for-byte.** An earlier
version of this module ran `get_svg_image()` over the *whole* page and
round-tripped everything through SVG -> PDF, which re-encoded any content
that was already vector paths (CAD text-as-filled-paths, drawing geometry --
exactly what a human clicks in `manual_label.py`): curves re-fitted, `seqno`
grouping changed, fills re-expressed, so manual `LabelEntry` bboxes no
longer lined up. Now the original page is copied verbatim (`insert_pdf`)
and only the native text is *added* as vectors on top.

Verified API sequence (installed pymupdf 1.28.2, see
`tests/rastervec/Evaluation/Conversion/test_conversion.py`):

1. `out.insert_pdf(src, from_page=n, to_page=n)` copies the source page
   exactly -- every drawing / image / path object carried over unchanged
   (object re-serialisation only, no geometric edit). `get_drawings()` on
   the result returns the same rects as on the source.
2. On that copy, `add_redact_annot(page.rect)` + `apply_redactions(text=
   PDF_REDACT_TEXT_REMOVE, graphics=PDF_REDACT_LINE_ART_NONE, images=
   PDF_REDACT_IMAGE_NONE)` deletes the native text objects while leaving
   line art and images untouched -- so `get_text()` comes back empty (as
   before) but `get_drawings()` is bit-identical.
3. A throwaway copy has line art + images stripped instead
   (`PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED` / `PDF_REDACT_IMAGE_REMOVE`,
   `text=PDF_REDACT_TEXT_NONE`), leaving a text-only page. Its
   `get_svg_image()` emits each glyph as a filled `<path>` (never an SVG
   `<text>` element, never a raster `<image>`); `fitz.open(filetype="svg")`
   + `convert_to_pdf()` turn that into real PDF path content.
4. Both the output page and the throwaway are forced to rotation 0 before
   any of the above, so the SVG and the overlay both live in the page's
   canonical **unrotated MediaBox space**; the original `/Rotate` is
   re-applied to the output page at the end. (The old code's rotated-page
   handling was untested and placed glyphs in the wrong quadrant for
   90/270 pages -- this does not.)
"""
from __future__ import annotations

import pymupdf as fitz

from rastervec.logging_setup import get_logger

_LOG = get_logger("conversion")


def _strip_to_text_only(page: fitz.Page) -> None:
    """In place: drop every line-art and image object, keep the text."""
    page.set_rotation(0)
    page.add_redact_annot(page.rect)
    page.apply_redactions(
        text=fitz.PDF_REDACT_TEXT_NONE,
        graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
        images=fitz.PDF_REDACT_IMAGE_REMOVE,
    )


def _strip_native_text(page: fitz.Page) -> None:
    """In place: delete the native text objects, leave line art + images
    (and their exact geometry) untouched."""
    page.set_rotation(0)
    page.add_redact_annot(page.rect)
    page.apply_redactions(
        text=fitz.PDF_REDACT_TEXT_REMOVE,
        graphics=fitz.PDF_REDACT_LINE_ART_NONE,
        images=fitz.PDF_REDACT_IMAGE_NONE,
    )


def convert_page_to_vector_text(
    pdf_path: str, page_index: int, output_path: str | None = None,
) -> bytes:
    """Copies `pdf_path`'s page `page_index` verbatim, removes its native
    text objects, and draws that same text back on as vector paths. Every
    pre-existing vector path keeps its exact original geometry. Returns the
    new one-page PDF's bytes; also writes them to `output_path` if given."""
    src = fitz.open(pdf_path)
    try:
        rotation = int(src[page_index].rotation) % 360
        out = fitz.open()
        out.insert_pdf(src, from_page=page_index, to_page=page_index)
        text_only = fitz.open()
        text_only.insert_pdf(src, from_page=page_index, to_page=page_index)
    finally:
        src.close()

    out_page = out[0]
    _strip_native_text(out_page)

    to_page = text_only[0]
    _strip_to_text_only(to_page)
    svg = to_page.get_svg_image()
    text_only.close()

    svg_doc = fitz.open(stream=svg.encode("utf-8"), filetype="svg")
    text_vectors = fitz.open("pdf", svg_doc.convert_to_pdf())
    out_page.show_pdf_page(out_page.rect, text_vectors, 0)
    out_page.set_rotation(rotation)

    result = out.tobytes()
    _LOG.debug(
        "converted page %d of %s to vector text (%d bytes)",
        page_index, pdf_path, len(result),
    )

    if output_path is not None:
        with open(output_path, "wb") as f:
            f.write(result)

    return result
