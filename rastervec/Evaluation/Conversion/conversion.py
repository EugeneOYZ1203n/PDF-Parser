"""Conversion: re-express a page's content as vector paths so a PDF with
known text/geometry becomes a known-answer test case for Vector_Classification.
Three modes, none of which ever rewrites pre-existing vector-path geometry:

- `convert_page_text_only` -- native text drawn as vector paths, **nothing
  else** (line art + images stripped). The auto-ground-truth benchmark input:
  the pipeline can only find native-text content.
- `convert_page_drawings_only` -- the original page with its native text
  objects removed, **every drawing kept byte-for-byte** (images removed too).
  The manual-ground-truth benchmark input: the pipeline can only find the
  CAD-vector-text a human labels.
- `convert_page_to_vector_text` -- both of the above overlaid (text-as-vectors
  on top of the untouched drawings). No longer used by the benchmark (which
  scores auto/manual from the two disjoint inputs above); kept as a general
  "how is this text drawn" utility and for its tests.

Verified API sequence (installed pymupdf 1.28.2, see
`tests/rastervec/Evaluation/Conversion/test_conversion.py`):

1. `out.insert_pdf(src, from_page=n, to_page=n)` copies the source page
   exactly -- every drawing / image / path object carried over unchanged
   (object re-serialisation only, no geometric edit). `get_drawings()` on
   the result returns the same rects as on the source.
2. `add_redact_annot(page.rect)` + `apply_redactions(...)`:
   - `text=PDF_REDACT_TEXT_REMOVE, graphics=PDF_REDACT_LINE_ART_NONE` deletes
     the native text objects while leaving line art bit-identical (used by
     `convert_page_drawings_only`, which also passes
     `images=PDF_REDACT_IMAGE_REMOVE`).
   - `text=PDF_REDACT_TEXT_NONE, graphics=PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
     images=PDF_REDACT_IMAGE_REMOVE` leaves a text-only page.
3. That text-only page's `get_svg_image()` emits each glyph as a filled
   `<path>` (never an SVG `<text>` element, never a raster `<image>`);
   `fitz.open(filetype="svg")` + `convert_to_pdf()` turn it into real PDF path
   content.
4. Everything is done at rotation 0 (SVG and overlay both live in the page's
   canonical unrotated MediaBox space); the original `/Rotate` is re-applied
   to the output page at the end.
"""
from __future__ import annotations

import pymupdf as fitz

from rastervec.logging_setup import get_logger

_LOG = get_logger("conversion")


def _page_geometry(pdf_path: str, page_index: int) -> tuple[int, float, float]:
    """`(rotation, mediabox width, mediabox height)` of one source page."""
    src = fitz.open(pdf_path)
    try:
        page = src[page_index]
        mb = page.mediabox
        return int(page.rotation) % 360, mb.width, mb.height
    finally:
        src.close()


def _one_page_copy(pdf_path: str, page_index: int) -> "fitz.Document":
    """A fresh one-page document that is a verbatim copy of the source page."""
    src = fitz.open(pdf_path)
    try:
        out = fitz.open()
        out.insert_pdf(src, from_page=page_index, to_page=page_index)
        return out
    finally:
        src.close()


def _strip_to_text_only(page: fitz.Page) -> None:
    """In place: drop every line-art and image object, keep the text."""
    page.set_rotation(0)
    page.add_redact_annot(page.rect)
    page.apply_redactions(
        text=fitz.PDF_REDACT_TEXT_NONE,
        graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_TOUCHED,
        images=fitz.PDF_REDACT_IMAGE_REMOVE,
    )


def _strip_native_text(page: fitz.Page, *, remove_images: bool = False) -> None:
    """In place: delete the native text objects, leave line art (and its exact
    geometry) untouched. `remove_images` also drops embedded images."""
    page.set_rotation(0)
    page.add_redact_annot(page.rect)
    page.apply_redactions(
        text=fitz.PDF_REDACT_TEXT_REMOVE,
        graphics=fitz.PDF_REDACT_LINE_ART_NONE,
        images=(
            fitz.PDF_REDACT_IMAGE_REMOVE if remove_images
            else fitz.PDF_REDACT_IMAGE_NONE
        ),
    )


def _text_vectors_doc(pdf_path: str, page_index: int) -> "fitz.Document":
    """A one-page PDF whose only content is the source page's native text,
    drawn as vector paths (line art + images stripped, then SVG round-tripped).
    In unrotated MediaBox space."""
    copy = _one_page_copy(pdf_path, page_index)
    try:
        page = copy[0]
        _strip_to_text_only(page)
        svg = page.get_svg_image()
    finally:
        copy.close()
    svg_doc = fitz.open(stream=svg.encode("utf-8"), filetype="svg")
    return fitz.open("pdf", svg_doc.convert_to_pdf())


def _emit(
    doc: "fitz.Document", pdf_path: str, page_index: int, output_path: str | None,
) -> bytes:
    result = doc.tobytes()
    doc.close()
    _LOG.debug(
        "converted page %d of %s (%d bytes)", page_index, pdf_path, len(result)
    )
    if output_path is not None:
        with open(output_path, "wb") as f:
            f.write(result)
    return result


def convert_page_text_only(
    pdf_path: str, page_index: int, output_path: str | None = None,
) -> bytes:
    """The source page's native text redrawn as vector paths, with **every
    drawing and image removed**. `get_drawings()` on the result holds only
    glyph paths; `get_text()` is empty."""
    rotation, width, height = _page_geometry(pdf_path, page_index)
    text_vectors = _text_vectors_doc(pdf_path, page_index)
    try:
        out = fitz.open()
        out_page = out.new_page(width=width, height=height)
        out_page.show_pdf_page(out_page.rect, text_vectors, 0)
        out_page.set_rotation(rotation)
    finally:
        text_vectors.close()
    return _emit(out, pdf_path, page_index, output_path)


def convert_page_drawings_only(
    pdf_path: str, page_index: int, output_path: str | None = None,
) -> bytes:
    """The source page copied verbatim, then its native text objects removed
    and images dropped -- **every drawing keeps its exact original geometry**.
    `get_drawings()` on the result equals the source's; `get_text()` is empty."""
    rotation, _w, _h = _page_geometry(pdf_path, page_index)
    out = _one_page_copy(pdf_path, page_index)
    out_page = out[0]
    _strip_native_text(out_page, remove_images=True)
    out_page.set_rotation(rotation)
    return _emit(out, pdf_path, page_index, output_path)


def convert_page_to_vector_text(
    pdf_path: str, page_index: int, output_path: str | None = None,
) -> bytes:
    """The source page copied verbatim (pre-existing vectors keep exact
    geometry), its native text objects removed, and that same text drawn back
    on top as vector paths. `get_drawings()` = original drawings + glyph paths;
    `get_text()` is empty. Not used by the benchmark -- see the module docstring."""
    rotation, _w, _h = _page_geometry(pdf_path, page_index)
    out = _one_page_copy(pdf_path, page_index)
    out_page = out[0]
    _strip_native_text(out_page)

    text_vectors = _text_vectors_doc(pdf_path, page_index)
    try:
        out_page.show_pdf_page(out_page.rect, text_vectors, 0)
        out_page.set_rotation(rotation)
    finally:
        text_vectors.close()
    return _emit(out, pdf_path, page_index, output_path)
