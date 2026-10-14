"""Unit tests for native-vs-OCR compare helpers and job path helpers."""

from __future__ import annotations

import pytest

pytest.skip("depends on raster_parser.models / serialisation / parsing, which were removed; this tool is kept unchanged until it is ported or removed", allow_module_level=True)


from pathlib import Path

from raster_parser.models import Text
from raster_parser.models import Vector
from raster_parser.evaluation.native_vs_ocr.native_vs_ocr import (
    NativeVsOcr,
    _cer,
    _character_match_rate,
    _edit_distance,
    _rect_iou,
    _text_for_compare,
    character_match_percentage,
    cer_percentage,
)


def _glyph(seqno: int, ch: str, rect: tuple[float, float, float, float]) -> tuple[str, Vector]:
    v = Vector(
        items=[],
        type="f",
        rect=rect,
        fill=(0.0, 0.0, 0.0),
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
    return ch, v


def test_edit_distance_and_cer() -> None:
    assert _edit_distance("abc", "abc") == 0
    assert _edit_distance("abc", "axc") == 1
    assert _edit_distance("", "ab") == 2
    assert _cer("hello", "hello") == 0.0
    assert _cer("ab", "ax") == 0.5
    assert _cer("", "x") == 1.0
    assert _cer("", "") == 0.0


def test_text_for_compare_uppercases_ambiguous() -> None:
    assert _text_for_compare("a z x") == "aZX"
    assert _text_for_compare("  hi  ") == "hi"


def test_rect_iou() -> None:
    assert _rect_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _rect_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    iou = _rect_iou((0, 0, 10, 10), (5, 0, 15, 10))
    assert 0.3 < iou < 0.4


def test_native_vs_ocr_compare_and_missed() -> None:
    native = {
        1: _glyph(1, "A", (0, 0, 10, 10)),
        2: _glyph(2, "B", (10, 0, 20, 10)),
        3: _glyph(3, "C", (40, 0, 50, 10)),
    }
    ocr_row = (
        Text(0, 0, 20, 10, "AB", 0, 0, 0),
        [1, 2],
    )
    checker = NativeVsOcr(native, [ocr_row])
    stats = checker.compare_words()
    assert len(stats) == 1
    assert stats[0].match is True
    assert stats[0].CER == 0.0
    assert stats[0].character_match_rate == 1.0
    assert checker.native_glyphs_not_ocred() == [(3, "C")]

    assert character_match_percentage(stats, checker.native_glyphs_not_ocred()) > 0
    assert cer_percentage(stats, checker.native_glyphs_not_ocred()) >= 0.0


def test_character_match_rate_partial() -> None:
    assert _character_match_rate("ABCD", "ABXX") == 0.5
    assert _character_match_rate("", "x") == 0.0
