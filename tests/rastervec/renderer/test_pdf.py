from __future__ import annotations

import pymupdf as fitz

from rastervec.models import DrawingVector, OcrWord, PageMeta, TextVectorResult, TextWord, VectorPath
from rastervec.renderer import render_reconstructed_page, render_reconstructed_pdf


def _make_path(
    *, kind="re", bbox=(0, 0, 10, 10), stroke_color=None, fill_color=None,
    stroke_width=None, dashes=None, closed=None,
) -> VectorPath:
    if kind == "l":
        points = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
    elif kind == "c":
        points = [
            (bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[1]), (bbox[2], bbox[3]),
        ]
    else:
        points = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
    return VectorPath(
        seq=0, item_index=0, kind=kind, fill_rule="", points=points, bbox=bbox,
        stroke_color=stroke_color, fill_color=fill_color,
        stroke_opacity=None, fill_opacity=None, stroke_width=stroke_width,
        dashes=dashes, closed=closed, layer=None, page_index=0,
    )


def _make_word(*, text="Hi", origin=(10, 20), font_size=10.0, angle=0.0, color=0x000000) -> TextWord:
    return TextWord(
        text=text, bbox=(origin[0], origin[1] - font_size, origin[0] + font_size, origin[1]),
        quad=((0, 0), (0, 0), (0, 0), (0, 0)), angle=angle, direction=(1, 0),
        font="Helvetica", font_size=font_size, color=color, flags=0, origin=origin,
        ascender=None, descender=None, orientation_source="text-span", page_index=0, seq=0,
    )


def test_render_reconstructed_page_size_matches_zoomed_page_meta():
    meta = PageMeta(index=0, number=1, mediabox=(0, 0, 200, 100), rotation=0, width=200, height=100)

    image = render_reconstructed_page(meta, zoom=2.0)

    assert image.size == (400, 200)


def test_render_reconstructed_page_draws_native_words():
    meta = PageMeta(index=0, number=1, mediabox=(0, 0, 200, 100), rotation=0, width=200, height=100)
    word = _make_word()

    blank = render_reconstructed_page(meta, zoom=2.0)
    with_text = render_reconstructed_page(meta, native_words=[word], zoom=2.0)

    assert blank.convert("L").getextrema() == (255, 255)
    darkest, _lightest = with_text.convert("L").getextrema()
    assert darkest < 255


def test_render_reconstructed_page_draws_drawing_vectors():
    meta = PageMeta(index=0, number=1, mediabox=(0, 0, 200, 100), rotation=0, width=200, height=100)
    path = _make_path(kind="l", bbox=(10, 10, 60, 60), stroke_color=(0, 0, 0), stroke_width=2)
    dv = DrawingVector(
        paths=[path], bbox=path.bbox, stroke_color=(0, 0, 0), fill_color=None,
        stroke_width=2, dashed=False, page_index=0,
    )

    image = render_reconstructed_page(meta, drawing_vectors=[dv], zoom=2.0)

    darkest, _lightest = image.convert("L").getextrema()
    assert darkest < 255


def test_render_reconstructed_page_draws_ocr_results():
    meta = PageMeta(index=0, number=1, mediabox=(0, 0, 200, 100), rotation=0, width=200, height=100)
    path = _make_path(kind="l", bbox=(10, 10, 60, 60), stroke_color=(0, 0, 0))
    result = TextVectorResult(
        paths=[path], text="Hello", confidence=0.9,
        bbox=(10, 10, 60, 30), ocr_bbox=None, rotation_used=0, page_index=0,
    )

    image = render_reconstructed_page(meta, ocr_results=[result], zoom=2.0)

    darkest, _lightest = image.convert("L").getextrema()
    assert darkest < 255


def test_render_reconstructed_page_shrinks_ocr_text_to_fit_narrow_bbox():
    # A long string in a narrow bbox must not raise or overflow the page --
    # width-fit should shrink the height-derived fontsize further.
    meta = PageMeta(index=0, number=1, mediabox=(0, 0, 200, 100), rotation=0, width=200, height=100)
    path = _make_path(kind="l", bbox=(10, 10, 30, 20), stroke_color=(0, 0, 0))
    result = TextVectorResult(
        paths=[path], text="A very long piece of OCR'd text", confidence=0.9,
        bbox=(10, 10, 30, 20), ocr_bbox=None, rotation_used=0, page_index=0,
    )

    image = render_reconstructed_page(meta, ocr_results=[result], zoom=2.0)

    darkest, _lightest = image.convert("L").getextrema()
    assert darkest < 255


def test_render_reconstructed_page_places_words_at_own_bboxes():
    # When TextVectorResult.words is populated, each word must be placed/
    # scaled into its own bbox instead of stretching result.text across the
    # whole cluster bbox.
    meta = PageMeta(index=0, number=1, mediabox=(0, 0, 200, 100), rotation=0, width=200, height=100)
    path = _make_path(kind="l", bbox=(10, 10, 90, 30), stroke_color=(0, 0, 0))
    result = TextVectorResult(
        paths=[path], text="HELLO WORLD", confidence=0.9,
        bbox=(10, 10, 90, 30), ocr_bbox=None, rotation_used=0, page_index=0,
        words=[
            OcrWord(text="HELLO", confidence=0.9, bbox=(10, 10, 45, 30)),
            OcrWord(text="WORLD", confidence=0.9, bbox=(50, 10, 90, 30)),
        ],
    )

    image = render_reconstructed_page(meta, ocr_results=[result], zoom=2.0)

    darkest, _lightest = image.convert("L").getextrema()
    assert darkest < 255


def test_render_reconstructed_page_arbitrary_angle_text_does_not_raise():
    meta = PageMeta(index=0, number=1, mediabox=(0, 0, 200, 100), rotation=0, width=200, height=100)
    word = _make_word(angle=37.0)

    image = render_reconstructed_page(meta, native_words=[word], zoom=2.0)

    darkest, _lightest = image.convert("L").getextrema()
    assert darkest < 255


def test_render_reconstructed_page_skips_blank_text():
    meta = PageMeta(index=0, number=1, mediabox=(0, 0, 200, 100), rotation=0, width=200, height=100)
    blank_word = _make_word(text="   ")

    image = render_reconstructed_page(meta, native_words=[blank_word], zoom=2.0)

    assert image.convert("L").getextrema() == (255, 255)


def test_render_reconstructed_page_draws_text_boxes():
    meta = PageMeta(index=0, number=1, mediabox=(0, 0, 200, 100), rotation=0, width=200, height=100)

    blank = render_reconstructed_page(meta, zoom=2.0)
    with_text = render_reconstructed_page(
        meta, text_boxes=[("Ground truth", (10, 10, 120, 30), 0.0)], zoom=2.0,
    )

    assert blank.convert("L").getextrema() == (255, 255)
    darkest, _lightest = with_text.convert("L").getextrema()
    assert darkest < 255


def test_render_reconstructed_pdf_returns_openable_pdf_with_text_and_drawings():
    meta = PageMeta(index=0, number=1, mediabox=(0, 0, 200, 100), rotation=0, width=200, height=100)
    path = _make_path(kind="l", bbox=(10, 40, 90, 40), stroke_color=(0, 0, 0), stroke_width=2)
    dv = DrawingVector(
        paths=[path], bbox=path.bbox, stroke_color=(0, 0, 0), fill_color=None,
        stroke_width=2, dashed=False, page_index=0,
    )

    pdf_bytes = render_reconstructed_pdf(
        meta,
        text_boxes=[("HELLO", (10, 10, 90, 30), 0.0)],
        drawing_vectors=[dv],
    )

    doc = fitz.open("pdf", pdf_bytes)
    try:
        assert doc.page_count == 1
        page = doc[0]
        assert "HELLO" in page.get_text()
        assert len(page.get_drawings()) >= 1
    finally:
        doc.close()


def test_render_reconstructed_pdf_page_size_matches_page_meta():
    meta = PageMeta(index=0, number=1, mediabox=(0, 0, 200, 100), rotation=0, width=200, height=100)

    doc = fitz.open("pdf", render_reconstructed_pdf(meta))
    try:
        assert (round(doc[0].rect.width), round(doc[0].rect.height)) == (200, 100)
    finally:
        doc.close()
