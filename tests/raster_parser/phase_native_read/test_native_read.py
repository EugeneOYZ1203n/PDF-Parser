"""`phase1.read_and_extract`: everything native extraction yields for one page."""

from __future__ import annotations

import pymupdf
from raster_parser.phase_native_read.native_read import read_and_extract


def test_read_and_extract_returns_text_vectors_and_images(tmp_pdf_path):
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)
    page.insert_text((20, 30), "word", fontsize=12)
    page.draw_line((10, 50), (100, 50))
    page.insert_image(
        pymupdf.Rect(100, 100, 180, 180),
        pixmap=pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 8, 8), 0),
    )
    result = read_and_extract(tmp_pdf_path(doc), 0)
    assert [t.text for t in result.texts] == ["word"]
    assert len(result.vectors) == 1 and len(result.images) == 1
    assert result.page.meta.index == 0
