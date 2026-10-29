"""Integration tests for ``extract()``."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from raster_parser import extract
from raster_parser.models import Text
from raster_parser.models import NativePDFElements, VectorDTO

pytest.importorskip("paddleocr")


def test_extract_raises_when_pdf_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="PDF not found"):
        extract(tmp_path / "missing.pdf")


def test_extract_stroke_pdf_returns_native_text(synthetic_pdf: Path) -> None:
    result = extract(
        synthetic_pdf,
        page_index=0,
        ocr_dpi=150,
        use_angle_cls=False,
        enable_raster_pass=False,
    )
    assert isinstance(result, NativePDFElements)
    assert len(result.vectors) == 1
    assert result.vectors[0].type == "s"
    assert isinstance(result.vectors[0], VectorDTO)
    assert {t.word for t in result.words} == {"hello", "world"}


@patch("raster_parser.parsing.parser.PdfRasterParser.extract_text_from_paddle_word_groups")
def test_extract_merges_mock_ocr_with_strokes(mock_ocr, synthetic_pdf_with_fills: Path) -> None:
    mock_ocr.return_value = [(Text(0, 0, 20, 20, "glyph", 0, 0, 0), [1])]
    result = extract(synthetic_pdf_with_fills, use_angle_cls=False, enable_raster_pass=False)
    assert len(result.vectors) == 1
    assert result.vectors[0].type == "s"
    assert any(t.word == "glyph" for t in result.words)
    mock_ocr.assert_called_once()


@patch(
    "raster_parser.parsing.parser.PdfRasterParser.extract_text_from_paddle_word_groups",
    return_value=[],
)
def test_extract_without_ocr_hits_returns_strokes_only(
    mock_ocr, synthetic_pdf_with_fills: Path
) -> None:
    result = extract(synthetic_pdf_with_fills, use_angle_cls=False, enable_raster_pass=False)
    assert len(result.vectors) == 1
    assert result.words == []
    mock_ocr.assert_called_once()


@pytest.mark.slow
def test_extract_runs_paddle_on_fill_groups(synthetic_pdf_with_fills: Path) -> None:
    result = extract(
        synthetic_pdf_with_fills,
        page_index=0,
        ocr_dpi=150,
        use_angle_cls=False,
        enable_raster_pass=False,
    )
    assert len(result.vectors) == 1
    assert all(isinstance(v, VectorDTO) for v in result.vectors)
    json.loads(result.model_dump_json())


@patch(
    "raster_parser.parsing.parser.PdfRasterParser.extract_text_from_paddle_word_groups",
    return_value=[],
)
def test_extract_merges_type4_when_raster_pass(
    mock_ocr, synthetic_pdf: Path, tmp_path: Path
) -> None:
    """Type-2 strokes + mocked Type-4 words/vectors merge into one result."""
    from raster_parser.models import Vector

    work = tmp_path / "lo_work"
    work.mkdir()
    cleared = work / "page_cleared.pdf"
    doc = __import__("pymupdf").open()
    doc.new_page(width=100, height=100)
    doc.save(cleared)
    doc.close()

    t4_word = Text(1, 1, 10, 10, "raster", 0, 0, 0)
    t4_vec = Vector(
        items=[],
        type="s",
        rect=(0.0, 0.0, 5.0, 5.0),
        fill=None,
        color=(0.0, 0.0, 0.0),
        width=1.0,
        even_odd=False,
        close_path=False,
        dashes="",
        fill_opacity=1.0,
        stroke_opacity=1.0,
        seqno=0,
        layer=None,
    )

    with (
        patch(
            "raster_parser.delete_natives.delete_native.clear_page",
            return_value=(work, cleared),
        ),
        patch("raster_parser.delete_natives.delete_native.page_is_nonempty", return_value=True),
        patch(
            "raster_parser.parsing.extract_raster_page.extract_raster_page",
            return_value=([t4_word], [t4_vec]),
        ),
    ):
        result = extract(
            synthetic_pdf,
            page_index=0,
            ocr_dpi=150,
            use_angle_cls=False,
            enable_raster_pass=True,
            dump_dir=tmp_path / "dumps",
        )

    assert any(t.word == "hello" for t in result.words)
    assert any(t.word == "raster" for t in result.words)
    assert len(result.vectors) >= 2


@patch(
    "raster_parser.parsing.parser.PdfRasterParser.extract_text_from_paddle_word_groups",
    return_value=[],
)
def test_extract_cleans_dump_dir_when_debug_false(
    mock_ocr, synthetic_pdf: Path, tmp_path: Path
) -> None:
    dump_dir = tmp_path / "dumps"
    work = dump_dir / "extract_test_p0_work"
    work.mkdir(parents=True)
    cleared = work / "page_cleared.pdf"
    doc = __import__("pymupdf").open()
    doc.new_page(width=100, height=100)
    doc.save(cleared)
    doc.close()

    with (
        patch(
            "raster_parser.delete_natives.delete_native.clear_page",
            return_value=(work, cleared),
        ),
        patch("raster_parser.delete_natives.delete_native.page_is_nonempty", return_value=False),
    ):
        extract(
            synthetic_pdf,
            page_index=0,
            enable_raster_pass=True,
            dump_dir=dump_dir,
            debug=False,
        )

    assert not dump_dir.exists()
    mock_ocr.assert_called_once()


@patch(
    "raster_parser.parsing.parser.PdfRasterParser.extract_text_from_paddle_word_groups",
    return_value=[],
)
def test_extract_keeps_dump_dir_when_debug_true(
    mock_ocr, synthetic_pdf: Path, tmp_path: Path
) -> None:
    dump_dir = tmp_path / "dumps"
    work = dump_dir / "extract_test_p0_work"
    work.mkdir(parents=True)
    cleared = work / "page_cleared.pdf"
    doc = __import__("pymupdf").open()
    doc.new_page(width=100, height=100)
    doc.save(cleared)
    doc.close()

    with (
        patch(
            "raster_parser.delete_natives.delete_native.clear_page",
            return_value=(work, cleared),
        ),
        patch("raster_parser.delete_natives.delete_native.page_is_nonempty", return_value=False),
    ):
        extract(
            synthetic_pdf,
            page_index=0,
            enable_raster_pass=True,
            dump_dir=dump_dir,
            debug=True,
        )

    assert dump_dir.is_dir()
    mock_ocr.assert_called_once()
