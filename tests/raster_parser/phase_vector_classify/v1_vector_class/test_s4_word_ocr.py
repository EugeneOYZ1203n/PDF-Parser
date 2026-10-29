"""v1_vector_class step 4: word-group OCR, with PaddleOCR replaced by a fake."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pymupdf
import pytest

from raster_parser.models.vector import Vector
from raster_parser.phase_vector_classify.v1_vector_class import s4_word_ocr
from raster_parser.phase_vector_classify.v1_vector_class.s1_fill_filter import filter_text_vectors
from raster_parser.phase_vector_classify.v1_vector_class.s4_word_ocr import ocr_word_groups


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
    s4_word_ocr.engine_for.cache_clear()
    monkeypatch.setattr(s4_word_ocr, "PaddleOcr", _FakeOcr)
    yield _FakeOcr
    s4_word_ocr.engine_for.cache_clear()


def _fills(pdf: Path) -> list[Vector]:
    with pymupdf.open(pdf) as doc:
        return filter_text_vectors(
            [
                Vector.from_pymupdf(d, page_index=0, seq=i)
                for i, d in enumerate(doc[0].get_drawings(extended=True))
            ]
        )


def test_no_groups_never_builds_the_engine(fake_ocr, synthetic_pdf: Path) -> None:
    with pymupdf.open(synthetic_pdf) as doc:
        assert ocr_word_groups(doc[0], [], [], use_angle_cls=False) == []
    assert fake_ocr.instances == []


def test_group_box_mismatch_raises(fake_ocr, synthetic_pdf: Path) -> None:
    with pymupdf.open(synthetic_pdf) as doc, pytest.raises(ValueError, match="length mismatch"):
        ocr_word_groups(doc[0], [[]], [], use_angle_cls=False)


def test_most_confident_word_with_group_index(fake_ocr, synthetic_pdf_with_fills: Path) -> None:
    fills = _fills(synthetic_pdf_with_fills)
    box = (50.0, 50.0, 74.0, 82.0)
    with pymupdf.open(synthetic_pdf_with_fills) as doc:
        words = ocr_word_groups(doc[0], [[], fills], [box, box], use_angle_cls=False)
    assert words == [
        (1, "word", 0.6)
    ]  # the empty group is skipped; the confidence comes with the word
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
    assert pairs == [(0, "raw_word", 0.9)]
    assert len(fake_ocr.instances[0].calls) == 2


def test_the_engine_is_built_once_per_settings_not_once_per_call(
    fake_ocr, synthetic_pdf_with_fills: Path
) -> None:
    fills = _fills(synthetic_pdf_with_fills)
    box = (50.0, 50.0, 74.0, 82.0)
    with pymupdf.open(synthetic_pdf_with_fills) as doc:
        for _ in range(3):
            ocr_word_groups(doc[0], [fills], [box], use_angle_cls=False)
        assert len(fake_ocr.instances) == 1
        ocr_word_groups(
            doc[0], [fills], [box], use_angle_cls=True
        )  # other settings: its own engine
    assert len(fake_ocr.instances) == 2
