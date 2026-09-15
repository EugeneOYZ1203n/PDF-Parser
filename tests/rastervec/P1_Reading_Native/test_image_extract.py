from __future__ import annotations

import pymupdf as fitz
import pytest

from rastervec.P1_Reading_Native import image_extract
from rastervec.P1_Reading_Native.reader import Reader

# Distinct marker colors near two opposite unrotated-MediaBox corners --
# lets a test assert *where* content lands, not just the pixmap's size,
# since a 180-degree misrotation preserves dimensions but not placement.
_RED = (1, 0, 0)
_BLUE = (0, 0, 1)


def _build_marked_page(tmp_pdf_path, *, rotation: int) -> str:
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)  # unrotated: wide, short
    top_left = page.new_shape()
    top_left.draw_rect(fitz.Rect(0, 0, 20, 20))
    top_left.finish(color=None, fill=_RED)
    top_left.commit()
    bottom_right = page.new_shape()
    bottom_right.draw_rect(fitz.Rect(180, 80, 200, 100))
    bottom_right.finish(color=None, fill=_BLUE)
    bottom_right.commit()
    if rotation:
        page.set_rotation(rotation)
    return tmp_pdf_path(doc)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_render_whole_page_stays_in_unrotated_mediabox_space(tmp_pdf_path, rotation):
    path = _build_marked_page(tmp_pdf_path, rotation=rotation)
    with Reader(path) as reader:
        page = reader.get_page(0)
        img = image_extract._render_whole_page(page, dpi=72.0)

    # bbox is always the unrotated page dims -- the array must match, at
    # every /Rotate value, not just 0.
    assert img.bbox == (0.0, 0.0, 200.0, 100.0)
    assert img.array.shape[:2] == (100, 200)  # (height, width)

    # Marker colors must land at the *unrotated* corners regardless of
    # /Rotate -- this is the actual regression check (matching pixmap
    # dimensions alone would still pass a 180-degree-off render).
    top_left_px = tuple(int(c) for c in img.array[5, 5][:3])
    bottom_right_px = tuple(int(c) for c in img.array[-6, -6][:3])
    assert top_left_px == (255, 0, 0)
    assert bottom_right_px == (0, 0, 255)


def test_render_whole_page_respects_dpi_scaling(tmp_pdf_path):
    path = _build_marked_page(tmp_pdf_path, rotation=90)
    with Reader(path) as reader:
        page = reader.get_page(0)
        img = image_extract._render_whole_page(page, dpi=144.0)  # zoom = 2.0

    assert img.bbox == (0.0, 0.0, 200.0, 100.0)
    assert img.array.shape[:2] == (200, 400)  # (height, width) at 2x zoom
    assert img.dpi == 144.0


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_extract_images_whole_page_entry_matches_meta_dims(tmp_pdf_path, rotation):
    path = _build_marked_page(tmp_pdf_path, rotation=rotation)
    with Reader(path) as reader:
        page = reader.get_page(0)
        images = image_extract.extract_images(page, dpi=72.0)

    whole_page = images[0]
    assert whole_page.source == "page"
    assert whole_page.bbox == (0.0, 0.0, page.meta.width, page.meta.height)
    assert whole_page.array.shape[:2] == (page.meta.height, page.meta.width)
