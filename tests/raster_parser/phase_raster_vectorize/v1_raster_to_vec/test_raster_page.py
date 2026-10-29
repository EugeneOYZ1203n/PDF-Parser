"""`raster_page.extract_raster_page` end to end on a small page, PaddleOCR faked (it finds no text)."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from raster_parser.phase_raster_vectorize.v1_raster_to_vec import raster_page


class _NoTextOcr:
    """A PaddleOCR stand-in that detects and recognises nothing."""

    use_angle_cls = True

    def ocr(self, img, **kwargs):
        return [None]


@pytest.fixture
def drawing_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(raster_page, "make_ocr", lambda *a, **k: _NoTextOcr())
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=100)
    page.draw_rect(pymupdf.Rect(30, 20, 150, 80), color=(0, 0, 0), width=2)
    out = tmp_path / "drawing.pdf"
    doc.save(out)
    doc.close()
    return out


def test_without_keep_intermediates_nothing_is_left_in_the_dump_dir(
    drawing_pdf: Path, tmp_path: Path
) -> None:
    dump = tmp_path / "dump"
    texts, vectors = raster_page.extract_raster_page(
        drawing_pdf,
        0,
        dpi=100,
        output_dump_dir=dump,
        keep_intermediates=False,
    )
    assert texts == [] and vectors
    assert list(dump.iterdir()) == []


def test_by_default_the_debug_images_and_dumps_are_still_written(
    drawing_pdf: Path, tmp_path: Path
) -> None:
    dump = tmp_path / "dump"
    raster_page.extract_raster_page(drawing_pdf, 0, dpi=100, output_dump_dir=dump)
    (run,) = dump.iterdir()
    assert {
        "initial_det.png",
        "det_boxes.png",
        "text_final.png",
        "drawing_final.png",
        "ocr.txt",
        "vectors.txt",
    } <= {p.name for p in run.iterdir()}
