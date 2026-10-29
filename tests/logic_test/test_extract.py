"""Smoke test for `PdfRasterParser.extract()` and `extract_svg()`.

We build a tiny synthetic PDF on the fly (text + a vector path +
an embedded raster image) so the test is self-contained and doesn't
need a real PDF asset checked into the repo.
"""

from pathlib import Path

import pymupdf as fitz
import pytest

from raster_parser.parsing.parser import PdfRasterParser
from raster_parser.models import Text
from raster_parser.models import Vector


@pytest.fixture
def synthetic_pdf_with_image(tmp_path: Path) -> Path:
    """Like ``synthetic_pdf`` but includes an embedded raster for SVG image tests."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)

    page.insert_text((50, 50), "hello world", fontsize=14)

    page.draw_rect(
        fitz.Rect(60, 100, 200, 180),
        color=(1, 0, 0),
        width=1.5,
    )

    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20))
    pix.clear_with(128)
    page.insert_image(fitz.Rect(250, 250, 290, 290), pixmap=pix)

    out = tmp_path / "synthetic.pdf"
    doc.save(out)
    doc.close()
    return out


def test_extract_text_only(synthetic_pdf_with_image: Path) -> None:
    parser = PdfRasterParser(str(synthetic_pdf_with_image), page=0)
    words = parser.extract_text()

    assert parser.text_extracted is True
    assert all(isinstance(t, Text) for t in words)

    word_set = {t.word for t in words}
    assert "hello" in word_set
    assert "world" in word_set


def test_extract_vectors_only(synthetic_pdf_with_image: Path) -> None:
    parser = PdfRasterParser(str(synthetic_pdf_with_image), page=0)
    vectors = parser.extract_vectors()

    assert parser.vectors_extracted is True
    assert len(vectors) >= 1
    assert all(isinstance(v, Vector) for v in vectors)
    for v in vectors:
        assert len(v.rect) == 4
        x0, y0, x1, y1 = v.rect
        assert x0 <= x1 and y0 <= y1


def test_extract_full_pipeline(synthetic_pdf_with_image: Path) -> None:
    parser = PdfRasterParser(str(synthetic_pdf_with_image), page=0)
    result = parser.extract()

    assert result["page_index"] == 0
    assert result["width"] == pytest.approx(400.0)
    assert result["height"] == pytest.approx(400.0)

    words = {t["word"] for t in result["text"]}
    assert {"hello", "world"} <= words

    assert len(result["vectors"]) >= 1

    assert parser.svg_extracted is True
    assert parser.svg is not None
    assert parser.svg.startswith("<svg")
    assert "<image" in parser.svg


def test_extract_svg_auto_runs_prior_steps(synthetic_pdf_with_image: Path) -> None:
    """extract_svg() alone should run text + vector extraction, then export SVG."""
    parser = PdfRasterParser(str(synthetic_pdf_with_image), page=0)

    assert parser.text_extracted is False
    assert parser.vectors_extracted is False

    svg = parser.extract_svg()

    assert parser.text_extracted is True
    assert parser.vectors_extracted is True
    assert parser.svg_extracted is True
    assert svg is parser.svg


def test_extract_svg_matches_fresh_page_export(synthetic_pdf_with_image: Path) -> None:
    """SVG from the parser should match ``get_svg_image()`` on the same page."""
    raw_doc = fitz.open(synthetic_pdf_with_image)
    raw_svg = raw_doc[0].get_svg_image(text_as_path=False)
    raw_doc.close()

    parser = PdfRasterParser(str(synthetic_pdf_with_image), page=0)
    exported = parser.extract_svg()

    assert exported == raw_svg
    assert exported.startswith("<svg")
