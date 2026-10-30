"""v1_raster_to_vec step 4: box recognition, with PaddleOCR replaced by a fake."""

from __future__ import annotations

import numpy as np
import pytest

from raster_parser.models.source import Source
from raster_parser.phase_raster_vectorize.v1_raster_to_vec.s4_region_ocr import (
    recognize_boxes,
    recover_missed_glyphs,
)


class _Ocr:
    """Reads "AB" at confidence 0.9, both as a whole call and through the batch recognizer."""

    def ocr(self, img, **kwargs):
        return [[("AB", 0.9)]]

    def text_recognizer(self, crops):
        return [("AB", 0.95) for _ in crops], None


def test_recognize_boxes_keeps_the_recognition_confidence() -> None:
    gray = np.full((100, 100), 255, np.uint8)
    gray[20:40, 20:40] = 0
    result = recognize_boxes(gray, [(15, 15, 45, 45)], ocr=_Ocr())
    assert [(t.text, t.source, t.orientation_source) for t in result.texts] == [
        ("AB", Source.FROM_IMAGE, "ocr")
    ]
    assert result.texts[0].confidence == pytest.approx(0.9)


def test_recovered_glyphs_keep_the_recognition_confidence() -> None:
    gray = np.full((120, 120), 255, np.uint8)
    gray[30:60, 30:50] = 0  # a glyph-sized blob outside every detection box
    result = recover_missed_glyphs(gray, [], ocr=_Ocr())
    assert [t.text for t in result.texts] == ["AB"]
    assert result.texts[0].confidence == pytest.approx(0.95)
