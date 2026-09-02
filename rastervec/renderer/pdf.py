"""PDF-page reconstruction rendering.

`render_reconstructed_page` composes a fresh single-page PyMuPDF document
from whatever a pipeline stage has captured so far -- native text words,
drawing vectors, OCR'd vector-text results, or plain (text, bbox, rotation)
boxes -- and rasterizes it, so the visualization notebook can show "does
this look like the original" for one stage's output at a time.
`render_reconstructed_pdf` builds the exact same page but hands back the
PDF bytes instead of a raster, for side-by-side ground-truth-vs-pipeline
comparison files (see `notebooks/benchmark_vector_classification.ipynb`).
Both are rough previews, not `Evaluation/evaluation.py`'s real
(still-unbuilt) reconstruction stage: font family isn't preserved (always
the base14 "helv"), only size/baseline/rotation are approximated.

`render_boxes_pdf` is unrelated to reconstruction -- a generic "draw these
colored bbox outlines on a fresh page" primitive, used by
`Evaluation/Evaluate/evaluate.py`'s `render_evaluation_pdf` to visualize
matched/unmatched-label/unmatched-prediction bboxes.
"""
from __future__ import annotations

import io

import pymupdf as fitz
from PIL import Image

from rastervec.models import DrawingVector, PageMeta, TextVectorResult, TextWord
from rastervec.renderer._shapes import replay_drawing_paths

# One reconstructed text box: (text, page-space bbox, rotation in degrees).
TextBox = tuple[str, tuple[float, float, float, float], float]


def _text_color(color: int | None) -> tuple[float, float, float]:
    """Unpacks a PyMuPDF span-style packed sRGB int (as TextWord.color
    carries) into an (r, g, b) 0..1 tuple for insert_text's color param."""
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
    native_words: list[TextWord] | None = None,
    drawing_vectors: list[DrawingVector] | None = None,
    ocr_results: list[TextVectorResult] | None = None,
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
        shape = page.new_shape()
        for dv in drawing_vectors:
            replay_drawing_paths(shape, dv.paths)
        shape.commit()

    base_font = fitz.Font("helv")
    font_span = base_font.ascender - base_font.descender

    if native_words:
        for word in native_words:
            if not word.text.strip():
                continue
            if word.origin is not None:
                origin = word.origin
            else:
                # No real origin recorded -- approximate the baseline
                # from the bbox's top edge using the base14 font's own
                # ascender metric (a font's em-square is taller than
                # its bbox, and the baseline sits below the top edge by
                # roughly ascender * fontsize, not at the bbox's bottom
                # edge outright).
                x0, y0, _x1, _y1 = word.bbox
                origin = (x0, y0 + base_font.ascender * max(word.font_size, 1.0))
            # Rotate around the word's own bbox center, not the baseline
            # origin -- morph's fixpoint is what stays fixed under the
            # transform, so using origin as the fixpoint swings the text
            # around its own left edge instead of turning in place.
            bx0, by0, bx1, by1 = word.bbox
            center = fitz.Point((bx0 + bx1) / 2, (by0 + by1) / 2)
            page.insert_text(
                origin, word.text,
                fontsize=max(word.font_size, 1.0),
                color=_text_color(word.color),
                rotate=0,
                morph=(center, fitz.Matrix(1, 1).prerotate(word.angle)),
            )

    def _place_text(text: str, bbox: tuple[float, float, float, float], rotation: float) -> None:
        if not text.strip():
            return
        x0, y0, x1, y1 = bbox
        # A font's em-square (fontsize) is taller than the rendered
        # glyph bbox by ascender - descender (both em-fractions);
        # recover fontsize from the bbox height via that ratio, then
        # place the baseline ascender*fontsize below the bbox's top
        # edge, rather than treating the bbox height as the fontsize
        # and the bbox's bottom edge as the baseline outright.
        fontsize = max((y1 - y0) / font_span, 1.0)
        # Height alone doesn't guarantee the text actually fits within
        # its own bbox's width (e.g. a long OCR'd string in a narrow
        # cluster/word bbox) -- shrink fontsize further, uniformly, so
        # the rendered text_length never exceeds the bbox width it was
        # read from.
        bbox_width = x1 - x0
        if bbox_width > 0:
            text_width = base_font.text_length(text, fontsize=fontsize)
            if text_width > bbox_width:
                fontsize = max(fontsize * bbox_width / text_width, 1.0)
        origin = (x0, y0 + base_font.ascender * fontsize)
        # Rotate around the bbox's own center, not the baseline origin.
        center = fitz.Point((x0 + x1) / 2, (y0 + y1) / 2)
        page.insert_text(
            origin, text,
            fontsize=fontsize,
            rotate=0,
            morph=(center, fitz.Matrix(1, 1).prerotate(rotation)),
        )

    if ocr_results:
        for result in ocr_results:
            if result.words:
                # Word-level boxes available -- place/scale each word
                # into its own detected bbox instead of stretching one
                # string across the whole cluster bbox.
                for word in result.words:
                    _place_text(word.text, word.bbox, result.rotation_used)
            else:
                _place_text(result.text, result.bbox, result.rotation_used)

    if text_boxes:
        for text, bbox, rotation in text_boxes:
            _place_text(text, bbox, rotation)

    return doc


def render_reconstructed_page(
    page_meta: PageMeta,
    *,
    native_words: list[TextWord] | None = None,
    drawing_vectors: list[DrawingVector] | None = None,
    ocr_results: list[TextVectorResult] | None = None,
    text_boxes: list[TextBox] | None = None,
    zoom: float = 1.0,
) -> "Image.Image":
    """Notebook-only preview: redraws whatever has actually been captured
    so far -- one or more of native text words, drawing vectors (each drawn
    from its own real member VectorPaths, replayed per drawing so
    multi-contour fills keep their holes), OCR'd vector-text results, plain
    (text, bbox, rotation) boxes -- onto a fresh blank page sized/rotated
    to match `page_meta`, then rasterizes at `zoom` the same way the
    notebook rasterizes the real page (so the two images are
    pixel-comparable at the same zoom level)."""
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
    native_words: list[TextWord] | None = None,
    drawing_vectors: list[DrawingVector] | None = None,
    ocr_results: list[TextVectorResult] | None = None,
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


# One box overlay: a page-space bbox plus the (r, g, b) 0..1 color to
# outline it in.
BoxOverlay = tuple[tuple[float, float, float, float], tuple[float, float, float]]


def render_boxes_pdf(
    page_meta: PageMeta,
    boxes: list[BoxOverlay],
    *,
    width: float = 1.5,
) -> bytes:
    """A fresh page sized/rotated to `page_meta`, with each of `boxes`'s
    (bbox, color) pairs drawn as an unfilled rectangle outline
    (`page.draw_rect`, no fill -- so overlapping boxes stay legible). Used
    by `Evaluation/Evaluate/evaluate.py`'s `render_evaluation_pdf` to
    visualize one `evaluate_pipeline()` call's matched/unmatched-label/
    unmatched-prediction bboxes; generic otherwise -- it knows nothing
    about evaluation, just draws colored boxes."""
    doc = fitz.open()
    try:
        page = doc.new_page(width=page_meta.width, height=page_meta.height)
        page.set_rotation(page_meta.rotation)
        for bbox, color in boxes:
            page.draw_rect(fitz.Rect(*bbox), color=color, width=width)
        return doc.tobytes()
    finally:
        doc.close()
