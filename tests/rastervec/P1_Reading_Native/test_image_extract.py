from __future__ import annotations

import io

import numpy as np
import pymupdf as fitz
import pytest
from PIL import Image as PILImage

from rastervec.P1_Reading_Native import image_extract
from rastervec.P1_Reading_Native.reader import Reader

# A 40x20 px image: a red block in its top-left pixel corner, a blue block in
# its bottom-right -- lets a test assert where the placement transform lands
# each *pixel* corner on the page, not just the placement's bbox.
_W, _H = 40, 20


def _marker_png() -> bytes:
    arr = np.full((_H, _W, 3), 255, dtype=np.uint8)
    arr[:4, :4] = (255, 0, 0)
    arr[-4:, -4:] = (0, 0, 255)
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _apply(transform, fx: float, fy: float) -> tuple[float, float]:
    a, b, c, d, e, f = transform
    return (fx * a + fy * c + e, fx * b + fy * d + f)


def _build(tmp_pdf_path, *, rect, rotate: int = 0, page_rotation: int = 0) -> str:
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    page.insert_image(fitz.Rect(*rect), stream=_marker_png(), rotate=rotate, keep_proportion=False)
    if page_rotation:
        page.set_rotation(page_rotation)
    return tmp_pdf_path(doc)


def test_vector_only_page_yields_no_images(tmp_pdf_path):
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    page.draw_line((10, 10), (190, 90))
    path = tmp_pdf_path(doc)
    with Reader(path) as reader:
        assert image_extract.extract_images(reader.get_page(0)) == []


def test_embedded_image_native_pixels_and_bbox(tmp_pdf_path):
    path = _build(tmp_pdf_path, rect=(20, 10, 100, 50))
    with Reader(path) as reader:
        images = image_extract.extract_images(reader.get_page(0))

    assert len(images) == 1
    img = images[0]
    assert img.source == "embedded"
    assert img.array.shape[:2] == (_H, _W)  # native resolution, not a page render
    assert img.bbox == pytest.approx((20, 10, 100, 50))
    assert img.dpi == pytest.approx(_W / (80 / 72.0))
    assert tuple(int(c) for c in img.array[1, 1]) == (255, 0, 0)


@pytest.mark.parametrize("rotate", [0, 90, 180, 270])
@pytest.mark.parametrize("page_rotation", [0, 90])
def test_transform_maps_pixel_corners_to_placement(tmp_pdf_path, rotate, page_rotation):
    rect = (20, 10, 100, 90)
    path = _build(tmp_pdf_path, rect=rect, rotate=rotate, page_rotation=page_rotation)
    with Reader(path) as reader:
        page = reader.get_page(0)
        img = image_extract.extract_images(page)[0]
        # Ground truth: where does the red (pixel top-left) marker really
        # land on the page? Render the unrotated page and find it.
        fp = page.fitz_page
        pix = fp.get_pixmap(matrix=fp.derotation_matrix, alpha=False)
        rendered = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

    assert img.transform is not None
    red_ys, red_xs = np.nonzero(
        (rendered[..., 0] > 200) & (rendered[..., 1] < 80) & (rendered[..., 2] < 80)
    )
    assert red_xs.size
    red_page = (red_xs.mean(), red_ys.mean())
    # centre of the 4x4 red block, in unit-square image coordinates
    mapped = _apply(img.transform, 2.0 / _W, 2.0 / _H)
    assert mapped[0] == pytest.approx(red_page[0], abs=3.0)
    assert mapped[1] == pytest.approx(red_page[1], abs=3.0)
