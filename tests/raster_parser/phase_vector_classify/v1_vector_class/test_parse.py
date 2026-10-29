"""v1_vector_class `parse`: which vectors come back as text and which stay drawings (OCR faked)."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from raster_parser.models.settings import RunSettings
from raster_parser.phase_native_read.native_read import read_and_extract
from raster_parser.phase_vector_classify.v1_vector_class import s4_word_ocr
from raster_parser.phase_vector_classify.v1_vector_class.parse import parse


@pytest.fixture
def two_words_pdf(tmp_path: Path) -> Path:
    """A stroke rect, plus two groups of three glyph-like fills far apart (so two word groups)."""
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=400)
    page.draw_rect(pymupdf.Rect(60, 200, 200, 280), color=(1, 0, 0), width=1.5)
    shape = page.new_shape()
    for x_start in (50, 300):
        for x0 in (x_start, x_start + 8, x_start + 16):
            r = pymupdf.Rect(x0, 50, x0 + 8, 82)
            shape.draw_line(r.tl, r.tr)
            shape.draw_line(r.tr, r.br)
            shape.draw_line(r.br, r.bl)
            shape.draw_line(r.bl, r.tl)
            shape.finish(fill=(0, 0, 0), color=None, width=0, closePath=True)
    shape.commit()
    out = tmp_path / "two_words.pdf"
    doc.save(out)
    doc.close()
    return out


def _run(pdf: Path, monkeypatch: pytest.MonkeyPatch, read: list[tuple[int, str, float]]):
    monkeypatch.setattr(s4_word_ocr, "ocr_word_groups", lambda *a, **k: read)
    native = read_and_extract(str(pdf), 0)
    drawings, texts = parse(native.vectors, native.page, RunSettings())
    return native, drawings, texts


def test_unread_glyph_vectors_return_as_drawings(two_words_pdf: Path, monkeypatch) -> None:
    native, drawings, texts = _run(two_words_pdf, monkeypatch, read=[])
    assert texts == []
    assert [v.seqno for v in drawings] == [
        v.seqno for v in native.vectors
    ]  # nothing lost, same order


def test_read_groups_become_text_and_leave_drawings(two_words_pdf: Path, monkeypatch) -> None:
    native, drawings, texts = _run(two_words_pdf, monkeypatch, read=[(0, "ABC", 0.9)])
    assert [t.text for t in texts] == ["ABC"]
    fills = sorted(v.seqno for v in native.vectors if v.type == "f")
    first_group, second_group = fills[:3], fills[3:]
    kept = {v.seqno for v in drawings}
    assert not kept & set(first_group)  # the group that was read is now text, not drawing
    assert set(second_group) <= kept  # the group OCR could not read stays as drawing
    assert any(v.type == "s" for v in drawings)


def test_text_carries_the_ocr_confidence(two_words_pdf: Path, monkeypatch) -> None:
    _, _, texts = _run(two_words_pdf, monkeypatch, read=[(0, "ABC", 0.87)])
    assert [(t.text, t.confidence, t.source, t.orientation_source) for t in texts] == [
        ("ABC", 0.87, "ocr", "ocr")
    ]
