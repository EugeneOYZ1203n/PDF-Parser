"""Integration tests for ``extract()``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from raster_parser import extract
from raster_parser.P2_Raster_To_Vec.v1RasterToVec.s4_region_ocr import OcrWord as RasterWord
from raster_parser.P2_Raster_To_Vec.v1RasterToVec.s5_colour_autotrace import TraceVector
from raster_parser.P3_Vector_Parsing.v1VectorClass.s4_word_ocr import OcrWord
from raster_parser.commons.models import ExtractResult, Vector

pytest.importorskip("paddleocr")


def test_extract_raises_when_pdf_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="PDF not found"):
        extract(tmp_path / "missing.pdf")


def test_extract_stroke_pdf_returns_native_text(synthetic_pdf: Path) -> None:
    result = extract(
        synthetic_pdf,
        p3="v1VectorClass",
        page_index=0,
        ocr_dpi=150,
        use_angle_cls=False,
        enable_raster_pass=False,
    )
    assert isinstance(result, ExtractResult)
    assert len(result.vectors) == 1
    assert result.vectors[0].type == "s"
    assert isinstance(result.vectors[0], Vector)
    assert {t.text for t in result.texts} == {"hello", "world"}


@patch("raster_parser.P3_Vector_Parsing.v1VectorClass.s4_word_ocr.ocr_word_groups")
def test_extract_merges_mock_ocr_with_strokes(
    mock_ocr, synthetic_pdf_with_fills: Path
) -> None:
    mock_ocr.return_value = [(OcrWord(0, 0, 20, 20, "glyph", 0, 0, 0), [1])]
    result = extract(
        synthetic_pdf_with_fills, use_angle_cls=False, enable_raster_pass=False, p3="v1VectorClass"
    )
    assert len(result.vectors) == 1
    assert result.vectors[0].type == "s"
    assert any(t.text == "glyph" for t in result.texts)
    mock_ocr.assert_called_once()


@patch("raster_parser.P3_Vector_Parsing.v1VectorClass.s4_word_ocr.ocr_word_groups", return_value=[])
def test_extract_without_ocr_hits_returns_strokes_only(
    mock_ocr, synthetic_pdf_with_fills: Path
) -> None:
    result = extract(
        synthetic_pdf_with_fills, use_angle_cls=False, enable_raster_pass=False, p3="v1VectorClass"
    )
    assert len(result.vectors) == 1
    assert result.texts == []
    mock_ocr.assert_called_once()


@pytest.mark.slow
def test_extract_runs_paddle_on_fill_groups(synthetic_pdf_with_fills: Path) -> None:
    result = extract(
        synthetic_pdf_with_fills,
        p3="v1VectorClass",
        page_index=0,
        ocr_dpi=150,
        use_angle_cls=False,
        enable_raster_pass=False,
    )
    assert len(result.vectors) == 1
    assert all(isinstance(v, Vector) for v in result.vectors)
    assert ExtractResult.from_json(result.to_json()) == result


@patch("raster_parser.P3_Vector_Parsing.v1VectorClass.s4_word_ocr.ocr_word_groups", return_value=[])
def test_extract_merges_type4_when_raster_pass(
    mock_ocr, synthetic_pdf: Path, tmp_path: Path
) -> None:
    """Type-2 strokes + mocked Type-4 words/vectors merge into one result."""
    work = tmp_path / "lo_work"
    work.mkdir()
    cleared = work / "page_cleared.pdf"
    doc = __import__("pymupdf").open()
    doc.new_page(width=100, height=100)
    doc.save(cleared)
    doc.close()

    t4_word = RasterWord(1, 1, 10, 10, "raster", 0, 0, 0)
    t4_vec = TraceVector(
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
            "raster_parser.P2_Raster_To_Vec.v1RasterToVec.s1_clear_page.clear_page",
            return_value=(work, cleared),
        ),
        patch("raster_parser.P2_Raster_To_Vec.v1RasterToVec.s1_clear_page.page_is_nonempty", return_value=True),
        patch(
            "raster_parser.P2_Raster_To_Vec.v1RasterToVec.raster_page.extract_raster_page",
            return_value=([t4_word], [t4_vec]),
        ),
    ):
        result = extract(
            synthetic_pdf,
            p3="v1VectorClass",
            page_index=0,
            ocr_dpi=150,
            use_angle_cls=False,
            enable_raster_pass=True,
            dump_dir=tmp_path / "dumps",
        )

    assert any(t.text == "hello" for t in result.texts)
    assert any(t.text == "raster" for t in result.texts)
    assert len(result.vectors) >= 2


@patch("raster_parser.P3_Vector_Parsing.v1VectorClass.s4_word_ocr.ocr_word_groups", return_value=[])
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
            "raster_parser.P2_Raster_To_Vec.v1RasterToVec.s1_clear_page.clear_page",
            return_value=(work, cleared),
        ),
        patch("raster_parser.P2_Raster_To_Vec.v1RasterToVec.s1_clear_page.page_is_nonempty", return_value=False),
    ):
        extract(
            synthetic_pdf,
            p3="v1VectorClass",
            page_index=0,
            enable_raster_pass=True,
            dump_dir=dump_dir,
            debug=False,
        )

    assert not dump_dir.exists()
    mock_ocr.assert_called_once()


@patch("raster_parser.P3_Vector_Parsing.v1VectorClass.s4_word_ocr.ocr_word_groups", return_value=[])
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
            "raster_parser.P2_Raster_To_Vec.v1RasterToVec.s1_clear_page.clear_page",
            return_value=(work, cleared),
        ),
        patch("raster_parser.P2_Raster_To_Vec.v1RasterToVec.s1_clear_page.page_is_nonempty", return_value=False),
    ):
        extract(
            synthetic_pdf,
            p3="v1VectorClass",
            page_index=0,
            enable_raster_pass=True,
            dump_dir=dump_dir,
            debug=True,
        )

    assert dump_dir.is_dir()
    mock_ocr.assert_called_once()


@pytest.mark.parametrize("backend", [{"p2": "Nope"}, {"p3": "Nope"}])
def test_extract_rejects_unknown_backend(backend: dict, synthetic_pdf: Path) -> None:
    """A bad ``p2`` fails even with the raster pass off."""
    with pytest.raises(ValueError, match="unknown p[23] backend 'Nope'"):
        extract(synthetic_pdf, enable_raster_pass=False, **backend)


@patch("raster_parser.P3_Vector_Parsing.v1VectorClass.s4_word_ocr.ocr_word_groups")
def test_extract_p3_stub_keeps_every_vector_without_ocr(
    mock_ocr, synthetic_pdf_with_fills: Path
) -> None:
    result = extract(synthetic_pdf_with_fills, enable_raster_pass=False, p3="Stub")
    assert len(result.vectors) == 4  # the stroke and all three fills, none read as glyphs
    assert result.texts == []
    mock_ocr.assert_not_called()


@patch("raster_parser.P3_Vector_Parsing.v1VectorClass.s4_word_ocr.ocr_word_groups", return_value=[])
def test_extract_without_raster_pass_never_clears_the_page(
    mock_ocr, synthetic_pdf: Path
) -> None:
    with patch("raster_parser.P2_Raster_To_Vec.v1RasterToVec.s1_clear_page.clear_page") as clear_page:
        extract(synthetic_pdf, enable_raster_pass=False, p3="v1VectorClass")
    clear_page.assert_not_called()
