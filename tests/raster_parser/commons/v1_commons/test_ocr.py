"""`commons.v1_commons.ocr`: crop normalisation and OCR word -> `Text`."""

from __future__ import annotations

import numpy as np
import pytest

from raster_parser.commons.v1_commons.ocr import normalise_crop_for_ocr, ocr_text


def test_normalise_crop_scales_to_recognition_height() -> None:
    out = normalise_crop_for_ocr(np.zeros((100, 300, 3), dtype=np.uint8))
    assert out.shape[0] == 48
    assert out.shape[1] <= 1024


def test_normalise_crop_leaves_empty_crops_alone() -> None:
    crop = np.zeros((0, 0, 3), dtype=np.uint8)
    assert normalise_crop_for_ocr(crop) is crop


@pytest.mark.parametrize("rotate, direction", [(0, (1.0, 0.0)), (90, (0.0, -1.0))])
def test_ocr_text_direction_from_rotate(rotate, direction) -> None:
    t = ocr_text("A1", (1, 2, 11, 12), rotate, page_index=4, block_no=7)
    assert (t.text, t.bbox, t.block_no, t.page_index, t.source) == (
        "A1",
        (1.0, 2.0, 11.0, 12.0),
        7,
        4,
        "ocr",
    )
    assert t.direction == pytest.approx(direction)
    assert (t.font, t.font_size, t.color, t.line_no, t.word_no) == ("", 0.0, None, 0, 0)


def test_ocr_text_records_confidence_and_that_the_orientation_came_from_ocr() -> None:
    t = ocr_text("AB", (0, 0, 1, 1), 0, page_index=0, confidence=0.87)
    assert (t.confidence, t.orientation_source) == (0.87, "ocr")
    assert ocr_text("AB", (0, 0, 1, 1), 0, page_index=0).confidence == 0.0  # not given: as before


def test_paddle_lines_walks_a_paddle_2x_result() -> None:
    from raster_parser.commons.v1_commons.ocr import paddle_lines

    box = [[1, 2], [11, 2], [11, 12], [1, 12]]
    raw = [[[box, ("AB", 0.9)], None, [box], [box, "bad"], [box, ("", 0.4)]], None]
    assert [(b, t, c) for b, t, c in paddle_lines(raw)] == [(box, "AB", 0.9), (box, "", 0.4)]
    assert list(paddle_lines(None)) == [] and list(paddle_lines([None])) == []
