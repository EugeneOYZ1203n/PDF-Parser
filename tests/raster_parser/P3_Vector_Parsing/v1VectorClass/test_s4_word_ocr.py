"""v1VectorClass step 4: word-group OCR, with PaddleOCR replaced by a fake."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf
import pytest

from raster_parser.commons.models import Vector
from raster_parser.P3_Vector_Parsing.v1VectorClass.s2_word_grouping import Orientation
from raster_parser.P3_Vector_Parsing.v1VectorClass import s4_word_ocr
from raster_parser.P3_Vector_Parsing.v1VectorClass.s1_fill_filter import filter_text_vectors
from raster_parser.P3_Vector_Parsing.v1VectorClass.s4_word_ocr import (
    normalise_crop_for_ocr,
    ocr_word_groups,
    rotate_crop_for_ocr,
)


class _FakeOcr:
    """Reads `word` from the scaled crop and `raw_word` (if set) from the unscaled one."""

    instances: list["_FakeOcr"] = []

    def __init__(self, use_angle_cls, *, lang="en", drop_score=0.0):
        self.calls: list[np.ndarray] = []
        _FakeOcr.instances.append(self)

    def ocr_image(self, image, *, min_confidence=0.5, cls=None):
        self.calls.append(image)
        scaled = image.shape[0] == 48
        if scaled:
            return [((0, 0, 1, 1), "word", 0.6), ((0, 0, 1, 1), "low", 0.1)]
        return [((0, 0, 1, 1), "raw_word", 0.9)]


@pytest.fixture
def fake_ocr(monkeypatch):
    _FakeOcr.instances = []
    monkeypatch.setattr(s4_word_ocr, "PaddleOcr", _FakeOcr)
    return _FakeOcr


def _fills(pdf: Path) -> list[Vector]:
    with pymupdf.open(pdf) as doc:
        return filter_text_vectors([Vector.from_pymupdf(d, page_index=0, seq=i) for i, d in enumerate(doc[0].get_drawings(extended=True))])


def test_no_groups_never_builds_the_engine(fake_ocr, synthetic_pdf: Path) -> None:
    with pymupdf.open(synthetic_pdf) as doc:
        assert ocr_word_groups(doc[0], [], [], use_angle_cls=False) == []
    assert fake_ocr.instances == []


def test_group_box_mismatch_raises(fake_ocr, synthetic_pdf: Path) -> None:
    with pymupdf.open(synthetic_pdf) as doc, pytest.raises(ValueError, match="length mismatch"):
        ocr_word_groups(doc[0], [[]], [], use_angle_cls=False)


def test_most_confident_word_with_group_box_and_index(fake_ocr, synthetic_pdf_with_fills: Path) -> None:
    fills = _fills(synthetic_pdf_with_fills)
    box = (50.0, 50.0, 74.0, 82.0)
    with pymupdf.open(synthetic_pdf_with_fills) as doc:
        pairs = ocr_word_groups(doc[0], [[], fills], [box, box], use_angle_cls=False)
    assert len(pairs) == 1  # the empty group is skipped
    text, seqnos = pairs[0]
    assert (text.word, text.block_no) == ("word", 1)
    assert (text.x0, text.y0, text.x1, text.y1) == box
    assert seqnos == [v.seqno for v in fills]
    assert len(fake_ocr.instances[0].calls) == 1  # small group: scaled crop only


def _glyph_row(path: Path, count: int) -> Path:
    """`count` black glyph stubs side by side: a crop much wider than tall."""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    shape = page.new_shape()
    for i in range(count):
        r = pymupdf.Rect(10 + 12 * i, 50, 18 + 12 * i, 82)
        shape.draw_polyline([r.tl, r.tr, r.br, r.bl, r.tl])
        shape.finish(fill=(0, 0, 0), color=None, width=0, closePath=True)
    shape.commit()
    doc.save(path)
    doc.close()
    return path


def test_large_group_takes_the_more_confident_reading(fake_ocr, tmp_path: Path) -> None:
    pdf = _glyph_row(tmp_path / "row.pdf", 20)
    fills = _fills(pdf)
    assert len(fills) == 20
    with pymupdf.open(pdf) as doc:
        pairs = ocr_word_groups(doc[0], [fills], [(10.0, 50.0, 246.0, 82.0)], use_angle_cls=False)
    assert pairs[0][0].word == "raw_word"
    assert len(fake_ocr.instances[0].calls) == 2


def test_normalise_crop_scales_to_recognition_height() -> None:
    crop = np.zeros((100, 300, 3), dtype=np.uint8)
    out = normalise_crop_for_ocr(crop)
    assert out.shape[0] == 48
    assert out.shape[1] <= 1024


def test_rotate_crop_only_turns_vertical_groups() -> None:
    crop = np.zeros((10, 30, 3), dtype=np.uint8)
    assert rotate_crop_for_ocr(crop, Orientation.HORIZONTAL) is crop
    assert rotate_crop_for_ocr(crop, Orientation.VERTICAL).shape[:2] == (30, 10)
