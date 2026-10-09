"""Unit tests for stroke vs fill isoperimetric classification."""

from __future__ import annotations

from pathlib import Path

from raster_parser.models import Vector
from raster_parser.archive.stroke_fill_parser import (
    is_stroke_like,
    isoperimetric_quotient,
    stroke_fill_parser,
)


def _vector(
    *,
    rect: tuple[float, float, float, float],
    items: list | None = None,
    fill: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
    seqno: int = 0,
) -> Vector:
    return Vector(
        items=items if items is not None else [],
        type="f",
        rect=rect,
        fill=fill,
        color=None,
        width=0.0,
        even_odd=False,
        close_path=True,
        dashes="",
        fill_opacity=1.0,
        stroke_opacity=1.0,
        seqno=seqno,
        layer=None,
    )


def test_square_has_high_q() -> None:
    # Axis-aligned square via rect item ? near circle-ish Q for square (~0.785).
    v = _vector(rect=(0, 0, 10, 10), items=[("re", (0, 0, 10, 10))])
    q = isoperimetric_quotient(v)
    assert 0.7 < q < 0.9


def test_thin_ribbon_is_stroke_like() -> None:
    # Long thin rectangle: low Q.
    v = _vector(rect=(0, 0, 100, 2), items=[("re", (0, 0, 100, 2))])
    assert isoperimetric_quotient(v) < 0.25
    assert is_stroke_like(v)


def test_white_fill_never_stroke_like() -> None:
    v = _vector(rect=(0, 0, 100, 2), fill=(1.0, 1.0, 1.0), items=[("re", (0, 0, 100, 2))])
    assert not is_stroke_like(v)


def test_stroke_fill_parser_buckets(tmp_path: Path) -> None:
    stroke = _vector(rect=(0, 0, 80, 2), items=[("re", (0, 0, 80, 2))], seqno=1)
    fill = _vector(rect=(0, 0, 20, 20), items=[("re", (0, 0, 20, 20))], seqno=2)
    result = stroke_fill_parser(
        [stroke, fill],
        page_width=100.0,
        page_height=100.0,
        out_dir=tmp_path,
        write_pdfs=False,
    )
    assert {v.seqno for v in result.stroke_vectors} == {1}
    assert {v.seqno for v in result.fill_vectors} == {2}
    assert result.stroke_pdf_path is None
    assert result.fill_pdf_path is None
