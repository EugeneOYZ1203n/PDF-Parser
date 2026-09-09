"""PDF-page reconstruction rendering.

`render_reconstructed_page` composes a fresh single-page PyMuPDF document
from whatever a pipeline stage has captured so far -- native text, drawing
vectors, OCR'd text, or plain (text, bbox, rotation) boxes -- and
rasterizes it, so the visualization notebook can show "does this look like
the original" for one stage's output at a time. `render_reconstructed_pdf`
builds the exact same page but hands back the PDF bytes instead of a
raster, for side-by-side ground-truth-vs-pipeline comparison files (see
`notebooks/benchmark_vector_classification.ipynb`). Both are rough
previews: font family isn't preserved (always the base14 "helv"). Native
words render at their extracted font size (`_place_word`); OCR words and
label boxes have no real size, so `_place_text` derives it from the box
height and then fills the box width by widening the gaps between words
(a single word with no gaps falls back to per-character spacing; a string
too long even at natural spacing is shrunk uniformly instead).

`render_boxes_pdf` is unrelated to reconstruction -- a generic "draw these
colored bbox outlines on a fresh page" primitive, used by
`Evaluation/Evaluate/metrics.py`'s `overlay_boxes_split` for the
benchmark's pred-vs-GT box-overlay PDF.
"""
from __future__ import annotations

import io

import pymupdf as fitz
from PIL import Image

from rastervec.models import PageMeta, Text, Vector
from rastervec.renderer._shapes import replay_drawing_paths

# One reconstructed text box: (text, page-space bbox, rotation in degrees).
TextBox = tuple[str, tuple[float, float, float, float], float]


def _text_color(color: int | None) -> tuple[float, float, float]:
    """Unpacks a PyMuPDF span-style packed sRGB int (as Text.color carries)
    into an (r, g, b) 0..1 tuple for insert_text's color param."""
    if color is None:
        return (0.0, 0.0, 0.0)
    return (
        ((color >> 16) & 255) / 255,
        ((color >> 8) & 255) / 255,
        (color & 255) / 255,
    )


def _build_reconstructed_doc(
    page_meta: PageMeta,
    *,
    native_words: list[Text] | None = None,
    drawing_vectors: list[Vector] | None = None,
    ocr_results: list[Text] | None = None,
    text_boxes: list[TextBox] | None = None,
) -> "fitz.Document":
    """Compose a fresh one-page document sized/rotated to `page_meta` from
    whatever has been captured so far. The caller owns the returned
    document and must `close()` it. Text reconstruction is approximate:
    font family isn't preserved (always the PyMuPDF base14 "helv"). Rotation
    is exact at any angle -- page.insert_text's own `rotate` param only
    accepts multiples of 90, so text is rotated instead via a `morph`
    transform (a (fixpoint, rotation-matrix) pair applied as a `cm` op
    before drawing, PyMuPDF's own mechanism for arbitrary-angle text)."""
    doc = fitz.open()
    page = doc.new_page(width=page_meta.width, height=page_meta.height)
    page.set_rotation(page_meta.rotation)

    if drawing_vectors:
        replay_drawing_paths(page, drawing_vectors)

    base_font = fitz.Font("helv")
    font_span = base_font.ascender - base_font.descender

    def _place_word(word: Text) -> None:
        if not word.text.strip():
            return
        bx0, by0, bx1, by1 = word.bbox
        # Rotate around the word's own bbox center, not the baseline
        # origin -- morph's fixpoint is what stays fixed under the
        # transform, so using origin as the fixpoint swings the text
        # around its own left edge instead of turning in place.
        center = fitz.Point((bx0 + bx1) / 2, (by0 + by1) / 2)
        # `angle()` is measured in PyMuPDF's get_text `dir` convention (y down);
        # `insert_text`'s morph rotation turns the other way in that frame, so
        # the angle is negated here -- without it a word whose direction has a
        # non-zero y component reconstructs mirrored about the x-axis (e.g.
        # text reading up comes out reading down).
        page.insert_text(
            word.origin, word.text,
            fontsize=max(word.font_size, 1.0),
            color=_text_color(word.color),
            rotate=0,
            morph=(center, fitz.Matrix(1, 1).prerotate(-word.angle())),
        )

    if native_words:
        for word in native_words:
            _place_word(word)

    def _place_text(
        text: str,
        bbox: tuple[float, float, float, float],
        rotation: float,
        *,
        color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Place `text` so it fills `bbox`: font size is derived from the
        box height, then the box *width* is filled by widening the gaps
        between words (justified-text style) rather than by scaling the
        glyphs. A single word with no gaps to widen is stretched
        horizontally instead (one draw call, so it stays selectable in
        `render_reconstructed_pdf`'s output). A string too long even at
        natural spacing is shrunk uniformly. Used for OCR'd words (no real
        font size) and ground-truth label boxes; native words keep their
        extracted size via `_place_word` instead."""
        if not text.strip():
            return
        x0, y0, x1, y1 = bbox
        bbox_width = x1 - x0
        # A font's em-square (fontsize) is taller than the rendered glyph
        # bbox by ascender - descender (both em-fractions); recover
        # fontsize from the bbox height via that ratio.
        fontsize = max((y1 - y0) / font_span, 1.0)
        # Rotate the whole placed string as a unit about the bbox centre
        # (insert_text's `rotate` only does multiples of 90, so use morph).
        # `rotation` is in the get_text `dir` convention (y down); morph turns
        # the other way in that frame, hence `-rotation` (see `_place_word`).
        center = fitz.Point((x0 + x1) / 2, (y0 + y1) / 2)
        morph = (center, fitz.Matrix(1, 1).prerotate(-rotation))

        natural = base_font.text_length(text, fontsize=fontsize)

        # Overflow: no room even at natural spacing -> shrink the whole
        # font uniformly so the text never spills past the box (and page).
        if bbox_width > 0 and natural > bbox_width:
            fontsize = max(fontsize * bbox_width / natural, 1.0)
            origin = (x0, y0 + base_font.ascender * fontsize)
            page.insert_text(
                origin, text, fontsize=fontsize, color=color, rotate=0, morph=morph,
            )
            return

        origin_y = y0 + base_font.ascender * fontsize

        # Already fits, or no width to fill -> one call at natural spacing.
        if bbox_width <= 0 or natural >= bbox_width - 1e-3:
            page.insert_text(
                (x0, origin_y), text, fontsize=fontsize, color=color, rotate=0, morph=morph,
            )
            return

        # Underflow: distribute the leftover width across the gaps between
        # words (letterforms + intra-word spacing untouched), one draw call
        # per word all sharing `morph` so the line turns as a unit.
        tokens = text.split()
        if len(tokens) >= 2:
            slack = bbox_width - natural
            gap = base_font.text_length(" ", fontsize=fontsize) + slack / (len(tokens) - 1)
            cursor = x0
            for token in tokens:
                page.insert_text(
                    (cursor, origin_y), token,
                    fontsize=fontsize, color=color, rotate=0, morph=morph,
                )
                cursor += base_font.text_length(token, fontsize=fontsize) + gap
            return

        # A single word: no gaps to widen, so stretch it horizontally to
        # the box width via a non-uniform scale in the morph transform
        # (kept to one draw call so the text stays selectable). Cap the
        # stretch so a very short token in a wide box isn't grotesque, and
        # pre-shift the origin so that after scaling about the box centre
        # the glyphs still start at the box's left edge.
        scale = min(bbox_width / natural, 3.0) if natural > 0 else 1.0
        start_x = center.x - (bbox_width / 2.0) / scale
        page.insert_text(
            (start_x, origin_y), text, fontsize=fontsize, color=color, rotate=0,
            morph=(center, fitz.Matrix(scale, 1.0).prerotate(-rotation)),
        )

    if ocr_results:
        for word in ocr_results:
            _place_text(
                word.text, word.bbox, word.angle(), color=_text_color(word.color),
            )

    if text_boxes:
        for text, bbox, rotation in text_boxes:
            _place_text(text, bbox, rotation)

    return doc


def render_reconstructed_page(
    page_meta: PageMeta,
    *,
    native_words: list[Text] | None = None,
    drawing_vectors: list[Vector] | None = None,
    ocr_results: list[Text] | None = None,
    text_boxes: list[TextBox] | None = None,
    zoom: float = 1.0,
) -> "Image.Image":
    """Notebook-only preview: redraws whatever has actually been captured
    so far -- one or more of native text, drawing vectors (each drawn from
    its own real member items, replayed per Vector so multi-contour fills
    keep their holes), OCR'd text, plain (text, bbox, rotation) boxes --
    onto a fresh blank page sized/rotated to match `page_meta`, then
    rasterizes at `zoom` the same way the notebook rasterizes the real page
    (so the two images are pixel-comparable at the same zoom level)."""
    doc = _build_reconstructed_doc(
        page_meta,
        native_words=native_words,
        drawing_vectors=drawing_vectors,
        ocr_results=ocr_results,
        text_boxes=text_boxes,
    )
    try:
        pixmap = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        image.load()
        return image
    finally:
        doc.close()


def render_reconstructed_pdf(
    page_meta: PageMeta,
    *,
    native_words: list[Text] | None = None,
    drawing_vectors: list[Vector] | None = None,
    ocr_results: list[Text] | None = None,
    text_boxes: list[TextBox] | None = None,
) -> bytes:
    """Same reconstructed page as `render_reconstructed_page`, returned as
    PDF bytes -- for writing a ground-truth-vs-pipeline comparison file
    that keeps the reconstructed text as real (selectable) PDF text rather
    than a raster."""
    doc = _build_reconstructed_doc(
        page_meta,
        native_words=native_words,
        drawing_vectors=drawing_vectors,
        ocr_results=ocr_results,
        text_boxes=text_boxes,
    )
    try:
        return doc.tobytes()
    finally:
        doc.close()


# One box overlay: a page-space bbox, an (r, g, b) 0..1 outline color, and
# an optional PyMuPDF dash string (e.g. "[4 3] 0"; omit or None = solid).
BoxOverlay = (
    tuple[tuple[float, float, float, float], tuple[float, float, float]]
    | tuple[tuple[float, float, float, float], tuple[float, float, float], "str | None"]
)


def render_boxes_pdf(
    page_meta: PageMeta,
    boxes: list[BoxOverlay],
    *,
    width: float = 1.5,
) -> bytes:
    """A fresh page sized/rotated to `page_meta`, with each of `boxes`'s
    (bbox, color[, dashes]) entries drawn as an unfilled rectangle outline
    (`page.draw_rect`, no fill -- so overlapping boxes stay legible). A third
    tuple element, when present, is a PyMuPDF dash string (`None` = solid).
    Used by `Evaluation/Evaluate/metrics.py`'s `overlay_boxes_split`
    (dashed = auto GT, solid = manual GT, dotted = prediction); generic
    otherwise."""
    doc = fitz.open()
    try:
        page = doc.new_page(width=page_meta.width, height=page_meta.height)
        page.set_rotation(page_meta.rotation)
        for spec in boxes:
            bbox, color = spec[0], spec[1]
            dashes = spec[2] if len(spec) > 2 else None
            page.draw_rect(fitz.Rect(*bbox), color=color, width=width, dashes=dashes)
        return doc.tobytes()
    finally:
        doc.close()
