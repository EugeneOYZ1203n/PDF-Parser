"""v1VectorClass step 1: fill-glyph filter (copied from tests/raster_parser/test_reconstruct_box_filter.py)."""

from __future__ import annotations

from typing import Any

from raster_parser.commons.models import Vector
from raster_parser.P3_Vector_Parsing.v1VectorClass.s1_fill_filter import (
    filter_out_boxes,
    filter_text_vectors,
    is_box_or_rule,
    is_box_v1,
)


def _fill(
    *,
    items: list[Any],
    rect: tuple[float, float, float, float],
    fill: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
    color: tuple[float, float, float] | None = None,
    type_: str = "f",
    seqno: int = 1,
) -> Vector:
    return Vector(
        items=items,
        type=type_,
        rect=rect,
        fill=fill,
        color=color,
        width=0.0,
        even_odd=False,
        closePath=True,
        dashes="",
        fill_opacity=1.0,
        stroke_opacity=1.0,
        seqno=seqno,
        layer=None,
        lineCap=0,
        lineJoin=0.0,
        scissor=None,
        blendmode=None,
        isolated=False,
        knockout=False,
        opacity=None,
        page_index=0,
    )


def _four_lines(rect: tuple[float, float, float, float]) -> list:
    x0, y0, x1, y1 = rect
    return [
        ("l", (x0, y0), (x1, y0)),
        ("l", (x1, y0), (x1, y1)),
        ("l", (x1, y1), (x0, y1)),
        ("l", (x0, y1), (x0, y0)),
    ]


def test_is_box_v1_drops_every_filled_re() -> None:
    tiny = _fill(items=[("re", (0, 0, 2, 2))], rect=(0, 0, 2, 2))
    panel = _fill(
        items=[("re", (0, 0, 200, 150))],
        rect=(0, 0, 200, 150),
        fill=(0.7, 0.7, 0.9),
    )
    assert is_box_v1(tiny)
    assert is_box_v1(panel)


def test_is_box_v1_drops_every_filled_qu() -> None:
    qu = _fill(
        items=[("qu", ((0, 0), (10, 0), (10, 10), (0, 10)))],
        rect=(0, 0, 10, 10),
    )
    assert is_box_v1(qu)


def test_is_box_v1_keeps_stroke_re_without_fill() -> None:
    stroke_re = _fill(items=[("re", (0, 0, 50, 50))], rect=(0, 0, 50, 50), fill=None)
    assert not is_box_v1(stroke_re)


def test_is_box_v1_four_line_square_ish_black_is_box() -> None:
    rect = (0.0, 0.0, 40.0, 30.0)  # ratio ~1.33 < 3
    v = _fill(items=_four_lines(rect), rect=rect)
    assert is_box_or_rule(v)
    assert is_box_v1(v)


def test_is_box_v1_four_line_mid_ratio_black_not_box() -> None:
    rect = (0.0, 0.0, 8.0, 32.0)  # ratio 4 � not rule, not square-ish
    v = _fill(items=_four_lines(rect), rect=rect)
    assert not is_box_or_rule(v)
    assert not is_box_v1(v)


def test_is_box_v1_four_line_coloured_panel_is_box() -> None:
    rect = (0.0, 0.0, 8.0, 32.0)
    v = _fill(items=_four_lines(rect), rect=rect, fill=(0.8, 0.9, 0.5))
    assert is_box_or_rule(v)
    assert is_box_v1(v)


def test_is_box_v1_tiny_four_line_not_box() -> None:
    rect = (0.0, 0.0, 5.0, 5.0)  # both < 10
    v = _fill(items=_four_lines(rect), rect=rect)
    assert not is_box_or_rule(v)
    assert not is_box_v1(v)


def test_filter_text_vectors_removes_re_and_qu_keeps_glyph_like() -> None:
    re_v = _fill(items=[("re", (0, 0, 100, 80))], rect=(0, 0, 100, 80), seqno=1)
    qu_v = _fill(
        items=[("qu", ((0, 0), (20, 0), (20, 20), (0, 20)))],
        rect=(0, 0, 20, 20),
        seqno=2,
    )
    glyph = _fill(
        items=_four_lines((0.0, 0.0, 8.0, 32.0)),
        rect=(0.0, 0.0, 8.0, 32.0),
        seqno=3,
    )
    stroke = _fill(
        items=[("re", (0, 0, 10, 10))],
        rect=(0, 0, 10, 10),
        fill=None,
        type_="s",
        seqno=4,
    )
    white = _fill(
        items=_four_lines((0.0, 0.0, 8.0, 32.0)),
        rect=(0.0, 0.0, 8.0, 32.0),
        fill=(1.0, 1.0, 1.0),
        seqno=5,
    )

    kept = filter_text_vectors([re_v, qu_v, glyph, stroke, white])
    assert kept == [glyph]


def test_filter_out_boxes_only() -> None:
    re_v = _fill(items=[("re", (0, 0, 50, 50))], rect=(0, 0, 50, 50), seqno=1)
    glyph = _fill(
        items=_four_lines((0.0, 0.0, 8.0, 32.0)),
        rect=(0.0, 0.0, 8.0, 32.0),
        seqno=2,
    )
    assert filter_out_boxes([re_v, glyph]) == [glyph]


# Copied from tests/raster_parser/test_extract_pipeline.py


def _vector(seqno: int, type_: str, *, fill: tuple | None = None) -> Vector:
    return Vector(
        items=[],
        type=type_,
        rect=(float(seqno), 0.0, float(seqno + 10), 10.0),
        fill=fill,
        color=(0.0, 0.0, 0.0),
        width=1.0,
        even_odd=False,
        closePath=False,
        dashes="",
        fill_opacity=1.0,
        stroke_opacity=1.0,
        seqno=seqno,
        layer=None,
        lineCap=0,
        lineJoin=0.0,
        scissor=None,
        blendmode=None,
        isolated=False,
        knockout=False,
        opacity=None,
        page_index=0,
    )


def _split_strokes(all_vectors: list[Vector]) -> tuple[list[Vector], list[Vector]]:
    fills = filter_text_vectors(all_vectors)
    fill_ids = {v.seqno for v in fills}
    strokes = [v for v in all_vectors if v.seqno not in fill_ids]
    return strokes, fills


def test_filter_text_vectors_keeps_black_fills() -> None:
    stroke = _vector(1, "s")
    fill = _vector(2, "f", fill=(0.0, 0.0, 0.0))
    _, fills = _split_strokes([stroke, fill])
    assert fills == [fill]


def test_split_strokes_excludes_fill_glyphs() -> None:
    stroke = _vector(1, "s")
    fill = _vector(2, "f", fill=(0.0, 0.0, 0.0))
    strokes, fills = _split_strokes([stroke, fill])
    assert strokes == [stroke]
    assert fills == [fill]


def test_white_fill_stays_in_strokes_not_ocr_fills() -> None:
    stroke = _vector(1, "s")
    white = _vector(2, "f", fill=(1.0, 1.0, 1.0))
    strokes, fills = _split_strokes([stroke, white])
    assert fills == []
    assert strokes == [stroke, white]
