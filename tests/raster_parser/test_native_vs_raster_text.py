"""Unit tests for Type-4 native-vs-raster text helpers."""

from __future__ import annotations

import pytest

pytest.skip("depends on raster_parser.models / serialisation / parsing, which were removed; this tool is kept unchanged until it is ported or removed", allow_module_level=True)


from raster_parser.models import Text
from raster_parser.evaluation.native_vs_raster.native_vs_raster_text import (
    dedupe_stacked_texts,
    format_report,
)
from raster_parser.evaluation.native_vs_raster.native_vs_raster_text import ComparisonResult


def _t(word: str, box: tuple[float, float, float, float], *, word_no: int = 0) -> Text:
    x0, y0, x1, y1 = box
    return Text(x0=x0, y0=y0, x1=x1, y1=y1, word=word, block_no=0, line_no=0, word_no=word_no)


def test_dedupe_stacked_texts_keeps_one_of_near_identical() -> None:
    a = _t("ROOM", (10, 10, 50, 20), word_no=0)
    b = _t("ROOM", (10.5, 10.2, 50.2, 20.1), word_no=1)  # high IoU overlay
    c = _t("DOOR", (80, 10, 120, 20), word_no=2)
    out = dedupe_stacked_texts([a, b, c], iou_thresh=0.9)
    assert len(out) == 2
    assert {t.word for t in out} == {"ROOM", "DOOR"}


def test_dedupe_keeps_distinct_boxes() -> None:
    a = _t("A", (0, 0, 10, 10))
    b = _t("A", (50, 50, 60, 60))
    out = dedupe_stacked_texts([a, b], iou_thresh=0.9)
    assert len(out) == 2


def test_format_report_smoke() -> None:
    result = ComparisonResult(missed=[_t("MISS", (0, 0, 5, 5))])
    text = format_report(result, page_label="p0")
    assert "[p0]" in text
    assert "MISS" in text
    assert "native_words:" in text
