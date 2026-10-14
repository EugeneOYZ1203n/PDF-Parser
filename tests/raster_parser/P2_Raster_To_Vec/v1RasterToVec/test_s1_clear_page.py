"""Tests for LibreOffice clear path helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pymupdf

from raster_parser.P2_Raster_To_Vec.v1RasterToVec.s1_clear_page import (
    DEFAULT_OUTPUT_DUMP_DIR,
    clear_page,
    cleared_pdf_path,
    delete_drawings_from_pdf,
)


def test_cleared_pdf_path_beside_input() -> None:
    page = Path("/work/extract_foo/page.pdf")
    assert cleared_pdf_path(page) == Path("/work/extract_foo/page_cleared.pdf")


def test_default_output_dump_dir_is_tmp() -> None:
    assert DEFAULT_OUTPUT_DUMP_DIR == Path("./tmp")


def test_clear_page_uses_dump_dir(tmp_path: Path) -> None:
    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    doc.new_page(width=100, height=100)
    doc.save(src)
    doc.close()

    dump = tmp_path / "dumps"

    def _fake_delete(fp: Path | str) -> None:
        fp = Path(fp)
        cleared = cleared_pdf_path(fp)
        out = pymupdf.open()
        out.new_page(width=100, height=100)
        out.save(cleared)
        out.close()

    with patch("raster_parser.P2_Raster_To_Vec.v1RasterToVec.s1_clear_page.delete_drawing_page", side_effect=_fake_delete):
        work_dir, cleared = clear_page(src, 0, dump_dir=dump)

    assert work_dir.parent == dump
    assert cleared == work_dir / "page_cleared.pdf"
    assert cleared.is_file()


def test_delete_drawings_from_pdf_writes_under_dump_dir(tmp_path: Path) -> None:
    src = tmp_path / "drawing.pdf"
    doc = pymupdf.open()
    doc.new_page(width=100, height=100)
    doc.new_page(width=100, height=100)
    doc.save(src)
    doc.close()

    dump = tmp_path / "dumps"

    def _fake_delete(fp: Path | str) -> None:
        fp = Path(fp)
        cleared = cleared_pdf_path(fp)
        out = pymupdf.open()
        out.new_page(width=100, height=100)
        out.save(cleared)
        out.close()

    with patch("raster_parser.P2_Raster_To_Vec.v1RasterToVec.s1_clear_page.delete_drawing_page", side_effect=_fake_delete):
        cleared = delete_drawings_from_pdf(src, num_parallel=1, dump_dir=dump)

    assert cleared == dump / "drawing_cleared.pdf"
    assert cleared.is_file()
    assert not (src.parent / "drawing_0.pdf").exists()
    # per-page intermediates cleaned up
    assert not (dump / "drawing_0.pdf").exists()
    assert not (dump / "drawing_0_cleared.pdf").exists()
