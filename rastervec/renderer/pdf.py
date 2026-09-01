"""PDF-page reconstruction rendering.

`render_reconstructed_page` composes a fresh single-page PyMuPDF document
from whatever a pipeline stage has captured so far -- native text words,
drawing vectors, OCR'd vector-text results -- and rasterizes it, so the
visualization notebook can show "does this look like the original" for one
stage's output at a time. This is a rough preview, not
`Evaluation/evaluation.py`'s real (still-unbuilt) reconstruction stage:
font family isn't preserved (always the base14 "helv"), only size/baseline/
rotation are approximated.
"""
from __future__ import annotations

import io

import pymupdf as fitz
from PIL import Image

from rastervec.models import DrawingVector, PageMeta, TextVectorResult, TextWord
from rastervec.renderer._shapes import replay_drawing_paths


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


def render_reconstructed_page(
    page_meta: PageMeta,
    *,
    native_words: list[TextWord] | None = None,
    drawing_vectors: list[DrawingVector] | None = None,
    ocr_results: list[TextVectorResult] | None = None,
    zoom: float = 1.0,
) -> "Image.Image":
    """Notebook-only preview: redraws whatever has actually been captured
    so far -- one or more of native text words, drawing vectors (each drawn
    from its own real member VectorPaths, replayed per drawing so
    multi-contour fills keep their holes), OCR'd vector-text results --
    onto a fresh blank page sized/rotated to match `page_meta`, then
    rasterizes at `zoom` the same way the notebook rasterizes the real
    page (so the two images are pixel-comparable at the same zoom level).
    Text reconstruction is necessarily approximate: font family isn't
    preserved (always the PyMuPDF base14 "helv"). Rotation is exact, at any
    angle -- page.insert_text's own `rotate` param only accepts multiples
    of 90, so text is rotated instead via a `morph` transform (a
    (fixpoint, rotation-matrix) pair applied as a `cm` op before drawing,
    PyMuPDF's own mechanism for arbitrary-angle text)."""
    doc = fitz.open()
    try:
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

        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        image.load()
        return image
    finally:
        doc.close()
