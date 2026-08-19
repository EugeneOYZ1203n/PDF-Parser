from __future__ import annotations

import pytest

from rastervec.models import DrawingVector, PageMeta, TextVectorResult, TextWord, VectorPath
from rastervec.renderer import Renderer


class _Meta:
    index = 0


class _Page:
    meta = _Meta()


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


def test_render_vector_cluster_requires_at_least_one_path():
    with pytest.raises(ValueError):
        Renderer().render_vector_cluster([], _Page(), 150)


def test_render_vector_cluster_draws_something():
    # A filled black rectangle should show up as non-white pixels in the render.
    path = _make_path(kind="re", bbox=(0, 0, 20, 10), fill_color=(0, 0, 0))
    image = Renderer().render_vector_cluster([path], _Page(), dpi=150)

    assert image.width > 0 and image.height > 0
    darkest, _lightest = image.convert("L").getextrema()
    assert darkest < 255  # something was actually drawn, not a blank white page


def test_render_vector_cluster_blank_stroke_and_fill_stays_blank():
    # No stroke_color and no fill_color -> nothing should render.
    path = _make_path(kind="re", bbox=(0, 0, 20, 10))
    image = Renderer().render_vector_cluster([path], _Page(), dpi=150)

    darkest, _lightest = image.convert("L").getextrema()
    assert darkest == 255


def test_render_vector_cluster_line_kind():
    path = _make_path(kind="l", bbox=(0, 0, 20, 20), stroke_color=(0, 0, 0), stroke_width=2)
    image = Renderer().render_vector_cluster([path], _Page(), dpi=150)

    darkest, _lightest = image.convert("L").getextrema()
    assert darkest < 255


def _make_word(*, text="Hi", origin=(10, 20), font_size=10.0, angle=0.0, color=0x000000) -> TextWord:
    return TextWord(
        text=text, bbox=(origin[0], origin[1] - font_size, origin[0] + font_size, origin[1]),
        quad=((0, 0), (0, 0), (0, 0), (0, 0)), angle=angle, direction=(1, 0),
        font="Helvetica", font_size=font_size, color=color, flags=0, origin=origin,
        ascender=None, descender=None, orientation_source="text-span", page_index=0, seq=0,
    )


def test_render_reconstructed_page_size_matches_zoomed_page_meta():
    meta = PageMeta(index=0, number=1, mediabox=(0, 0, 200, 100), rotation=0, width=200, height=100)

    image = Renderer().render_reconstructed_page(meta, zoom=2.0)

    assert image.size == (400, 200)


def test_render_reconstructed_page_draws_native_words():
    meta = PageMeta(index=0, number=1, mediabox=(0, 0, 200, 100), rotation=0, width=200, height=100)
    word = _make_word()

    blank = Renderer().render_reconstructed_page(meta, zoom=2.0)
    with_text = Renderer().render_reconstructed_page(meta, native_words=[word], zoom=2.0)

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

    image = Renderer().render_reconstructed_page(meta, drawing_vectors=[dv], zoom=2.0)

    darkest, _lightest = image.convert("L").getextrema()
    assert darkest < 255


def test_render_reconstructed_page_draws_ocr_results():
    meta = PageMeta(index=0, number=1, mediabox=(0, 0, 200, 100), rotation=0, width=200, height=100)
    path = _make_path(kind="l", bbox=(10, 10, 60, 60), stroke_color=(0, 0, 0))
    result = TextVectorResult(
        cluster_paths=[path], text="Hello", confidence=0.9,
        bbox=(10, 10, 60, 30), rotation_used=0, page_index=0,
    )

    image = Renderer().render_reconstructed_page(meta, ocr_results=[result], zoom=2.0)

    darkest, _lightest = image.convert("L").getextrema()
    assert darkest < 255


def test_render_reconstructed_page_skips_blank_text():
    meta = PageMeta(index=0, number=1, mediabox=(0, 0, 200, 100), rotation=0, width=200, height=100)
    blank_word = _make_word(text="   ")

    # Must not raise (insert_text with empty text can be finicky) and must
    # produce a blank page since there's nothing real to draw.
    image = Renderer().render_reconstructed_page(meta, native_words=[blank_word], zoom=2.0)

    assert image.convert("L").getextrema() == (255, 255)
