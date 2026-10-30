"""Integration tests for ``extract()``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from raster_parser import extract
from raster_parser.models.source import Source
from raster_parser.models import ExtractResult, Vector
from raster_parser.commons.v1_commons.ocr import ocr_text

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
    assert isinstance(result, ExtractResult)
    assert len(result.vectors) == 1
    assert result.vectors[0].type == "s"
    assert isinstance(result.vectors[0], Vector)
    assert {t.text for t in result.texts} == {"hello", "world"}


@patch("raster_parser.phase_vector_classify.v1_vector_class.s4_word_ocr.ocr_word_groups")
def test_extract_merges_mock_ocr_with_strokes(mock_ocr, synthetic_pdf_with_fills: Path) -> None:
    mock_ocr.return_value = [(0, "glyph", 0.9)]
    result = extract(synthetic_pdf_with_fills, use_angle_cls=False, enable_raster_pass=False)
    assert len(result.vectors) == 1
    assert result.vectors[0].type == "s"
    assert any(t.text == "glyph" for t in result.texts)
    mock_ocr.assert_called_once()


@patch(
    "raster_parser.phase_vector_classify.v1_vector_class.s4_word_ocr.ocr_word_groups",
    return_value=[],
)
def test_extract_without_ocr_hits_keeps_unread_glyphs_as_drawings(
    mock_ocr, synthetic_pdf_with_fills: Path
) -> None:
    result = extract(synthetic_pdf_with_fills, use_angle_cls=False, enable_raster_pass=False)
    assert len(result.vectors) == 4  # the stroke and the three glyphs OCR could not read
    assert result.texts == []
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
    # The three glyph stubs are one word group: read by OCR they become text, otherwise (the stubs
    # are plain rectangles, so a real reader may well not) they stay as drawings next to the stroke.
    read = [t for t in result.texts if t.source != Source.NATIVE_TEXT]
    assert len(result.vectors) == (1 if read else 4)
    assert all(isinstance(v, Vector) for v in result.vectors)
    assert ExtractResult.from_json(result.to_json()) == result


@patch(
    "raster_parser.phase_vector_classify.v1_vector_class.s4_word_ocr.ocr_word_groups",
    return_value=[],
)
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

    t4_word = ocr_text("raster", (1, 1, 10, 10), 0, page_index=0, source=Source.FROM_IMAGE)
    t4_vec = Vector(
        type="s",
        items=[],
        color=(0.0, 0.0, 0.0),
        fill=None,
        width=1.0,
        dashes=None,
        closePath=False,
        lineCap=0,
        lineJoin=0.0,
        even_odd=False,
        stroke_opacity=1.0,
        fill_opacity=None,
        layer=None,
        rect=(0.0, 0.0, 5.0, 5.0),
        scissor=None,
        seqno=0,
        blendmode=None,
        isolated=False,
        knockout=False,
        opacity=None,
        page_index=0,
    )

    with (
        patch(
            "raster_parser.phase_raster_vectorize.v1_raster_to_vec.s1_clear_page.clear_page",
            return_value=(work, cleared),
        ),
        patch(
            "raster_parser.phase_raster_vectorize.v1_raster_to_vec.s1_clear_page.page_is_nonempty",
            return_value=True,
        ),
        patch(
            "raster_parser.phase_raster_vectorize.v1_raster_to_vec.raster_page.extract_raster_page",
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

    assert any(t.text == "hello" for t in result.texts)
    assert any(t.text == "raster" for t in result.texts)
    assert len(result.vectors) >= 2


@patch(
    "raster_parser.phase_vector_classify.v1_vector_class.s4_word_ocr.ocr_word_groups",
    return_value=[],
)
def test_extract_removes_only_its_own_run_dir_when_debug_false(
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
            "raster_parser.phase_raster_vectorize.v1_raster_to_vec.s1_clear_page.clear_page",
            return_value=(work, cleared),
        ),
        patch(
            "raster_parser.phase_raster_vectorize.v1_raster_to_vec.s1_clear_page.page_is_nonempty",
            return_value=False,
        ),
    ):
        extract(
            synthetic_pdf,
            page_index=0,
            enable_raster_pass=True,
            dump_dir=dump_dir,
            debug=False,
        )

    # the folder the caller passed (and what was already in it) is left alone; the run's own sub-folder is gone
    assert sorted(p.name for p in dump_dir.iterdir()) == ["extract_test_p0_work"]
    mock_ocr.assert_called_once()


@patch(
    "raster_parser.phase_vector_classify.v1_vector_class.s4_word_ocr.ocr_word_groups",
    return_value=[],
)
def test_extract_keeps_its_run_dir_when_debug_true(
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
            "raster_parser.phase_raster_vectorize.v1_raster_to_vec.s1_clear_page.clear_page",
            return_value=(work, cleared),
        ),
        patch(
            "raster_parser.phase_raster_vectorize.v1_raster_to_vec.s1_clear_page.page_is_nonempty",
            return_value=False,
        ),
    ):
        extract(
            synthetic_pdf,
            page_index=0,
            enable_raster_pass=True,
            dump_dir=dump_dir,
            debug=True,
        )

    kept = sorted(p.name for p in dump_dir.iterdir())
    assert (
        len(kept) == 2 and "extract_test_p0_work" in kept
    )  # the pre-existing folder plus this run's own
    mock_ocr.assert_called_once()


@pytest.mark.parametrize("backend", [{"p2": "Nope"}, {"p3": "Nope"}])
def test_extract_rejects_unknown_backend(backend: dict, synthetic_pdf: Path) -> None:
    """A bad ``p2`` fails even with the raster pass off."""
    with pytest.raises(ValueError, match="unknown p[23] backend 'Nope'"):
        extract(synthetic_pdf, enable_raster_pass=False, **backend)


@patch("raster_parser.phase_vector_classify.v1_vector_class.s4_word_ocr.ocr_word_groups")
def test_extract_p3_stub_keeps_every_vector_without_ocr(
    mock_ocr, synthetic_pdf_with_fills: Path
) -> None:
    result = extract(synthetic_pdf_with_fills, enable_raster_pass=False, p3="stub")
    assert len(result.vectors) == 4  # the stroke and all three fills, none read as glyphs
    assert result.texts == []
    mock_ocr.assert_not_called()


@patch(
    "raster_parser.phase_vector_classify.v1_vector_class.s4_word_ocr.ocr_word_groups",
    return_value=[],
)
def test_extract_without_raster_pass_never_clears_the_page(mock_ocr, synthetic_pdf: Path) -> None:
    with patch(
        "raster_parser.phase_raster_vectorize.v1_raster_to_vec.s1_clear_page.clear_page"
    ) as clear_page:
        extract(synthetic_pdf, enable_raster_pass=False)
    clear_page.assert_not_called()


def test_extract_leaves_files_already_in_dump_dir(synthetic_pdf: Path, tmp_path: Path) -> None:
    dump = tmp_path / "dump"
    dump.mkdir()
    (dump / "keep.txt").write_text("mine")
    extract(synthetic_pdf, p2="stub", p3="stub", dump_dir=dump)
    assert (dump / "keep.txt").read_text() == "mine"


def test_extract_default_does_not_touch_cwd_tmp(
    synthetic_pdf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cwd = tmp_path / "cwd"
    (cwd / "tmp").mkdir(parents=True)
    (cwd / "tmp" / "keep.txt").write_text("mine")
    monkeypatch.chdir(cwd)
    extract(synthetic_pdf, p2="stub", p3="stub")
    assert (cwd / "tmp" / "keep.txt").read_text() == "mine"
    assert [p.name for p in cwd.iterdir()] == ["tmp"]  # and nothing new in the cwd


def test_extract_calls_do_not_share_a_run_dir(synthetic_pdf: Path, tmp_path: Path) -> None:
    seen = []

    def spy(images, page, settings):
        seen.append(settings.dump_dir)
        return [], []

    with patch.dict("raster_parser.core.registry.VECTORIZE_BACKENDS", {"spy": spy}):
        extract(synthetic_pdf, p2="spy", p3="stub", dump_dir=tmp_path / "d", debug=True)
        extract(synthetic_pdf, p2="spy", p3="stub", dump_dir=tmp_path / "d", debug=True)
    assert len(set(seen)) == 2 and all(d.parent == tmp_path / "d" for d in seen)
