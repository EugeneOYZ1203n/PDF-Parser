"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz
import pytest


@pytest.fixture
def synthetic_pdf(tmp_path: Path) -> Path:
    """1-page PDF with native text and a vector stroke."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    page.insert_text((50, 50), "hello world", fontsize=14)
    page.draw_rect(
        fitz.Rect(60, 100, 200, 180),
        color=(1, 0, 0),
        width=1.5,
    )
    out = tmp_path / "synthetic.pdf"
    doc.save(out)
    doc.close()
    return out


@pytest.fixture
def synthetic_pdf_with_fills(tmp_path: Path) -> Path:
    """PDF with one stroke rect and three adjacent black fill glyph stubs.

    Glyph stubs are closed 4-line paths (not MuPDF ``re``), mid aspect ratio so
    ``is_box_v1`` does not treat them as layout boxes.
    """
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    page.draw_rect(fitz.Rect(60, 100, 200, 180), color=(1, 0, 0), width=1.5)
    shape = page.new_shape()
    for x0 in (50, 58, 66):
        r = fitz.Rect(x0, 50, x0 + 8, 82)
        shape.draw_line(r.tl, r.tr)
        shape.draw_line(r.tr, r.br)
        shape.draw_line(r.br, r.bl)
        shape.draw_line(r.bl, r.tl)
        shape.finish(fill=(0, 0, 0), color=None, width=0, closePath=True)
    shape.commit()
    out = tmp_path / "synthetic_fills.pdf"
    doc.save(out)
    doc.close()
    return out
