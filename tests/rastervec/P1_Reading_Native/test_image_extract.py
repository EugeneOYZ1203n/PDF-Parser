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


def _rgba_marker_png() -> bytes:
    """Opaque green everywhere, except a 4x4 corner block that's fully
    transparent but has red hidden underneath -- exercises real alpha
    compositing (should read back near-white), not a naive alpha drop
    (which would leak the hidden red)."""
    arr = np.zeros((_H, _W, 4), dtype=np.uint8)
    arr[..., :3] = (0, 255, 0)
    arr[..., 3] = 255
    arr[:4, :4, :3] = (255, 0, 0)
    arr[:4, :4, 3] = 0
    buf = io.BytesIO()
    PILImage.fromarray(arr, mode="RGBA").save(buf, format="PNG")
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


def test_masked_image_composites_over_background_not_raw_color(tmp_pdf_path):
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    page.insert_image(fitz.Rect(20, 10, 100, 50), stream=_rgba_marker_png())
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        images = image_extract.extract_images(reader.get_page(0))

    # The base image's own /SMask xref must not surface as a second,
    # independent placement.
    assert len(images) == 1
    img = images[0]
    # The fully-transparent corner's hidden red never leaks through --
    # composited toward white instead of kept raw.
    corner = tuple(int(c) for c in img.array[1, 1])
    assert corner[0] > 200 and corner[1] > 200 and corner[2] > 200
    # The opaque region keeps its real color.
    center = tuple(int(c) for c in img.array[_H // 2, _W // 2])
    assert center == (0, 255, 0)


def _build_with_standalone_stencil_mask(tmp_pdf_path) -> tuple[str, int]:
    """A real `/ImageMask true` XObject, drawn directly via `Do` and owned
    by no other image -- `fitz.Pixmap(doc, xref)` for a stencil mask like
    this comes back with `colorspace=None, alpha=1`, which is exactly what
    made the original bug report's `fitz.Pixmap(pix, 0)` raise "cannot drop
    alpha for 'NULL' colorspace" (verified directly against this object).
    `insert_image` has no public way to embed a raw `/ImageMask` XObject,
    so this is built at the low xref level, alongside a normal image so the
    page has real content too. Returns `(pdf_path, mask_xref)`."""
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    page.insert_image(fitz.Rect(20, 10, 100, 50), stream=_marker_png())

    w, h = 8, 8
    stream_data = bytes([0xFF]) * h  # 1 bpc, 1 byte/row, every bit set
    mask_xref = doc.get_new_xref()
    doc.update_object(mask_xref, f"""<<
/Type /XObject
/Subtype /Image
/Width {w}
/Height {h}
/BitsPerComponent 1
/ImageMask true
/Length {len(stream_data)}
>>""")
    doc.update_stream(mask_xref, stream_data)

    _, res_xref = doc.xref_get_key(page.xref, "Resources")
    res_xref = int(res_xref.split()[0])
    doc.xref_set_key(res_xref, "XObject/MaskImg", f"{mask_xref} 0 R")
    extra = b"q 40 0 0 40 120 10 cm /MaskImg Do Q"
    doc.update_stream(page.xref, page.read_contents() + b"\n" + extra)

    return tmp_pdf_path(doc), mask_xref


def test_standalone_stencil_mask_pixels_are_skipped_not_crashed(tmp_pdf_path):
    """A colorless mask/stencil pixmap (`colorspace is None`) must be
    skipped, not raise -- this is the shape of the original bug report
    (`fitz.Pixmap(pix, 0)` raising 'cannot drop alpha for NULL colorspace').
    Calls `_extract_xref_pixels` directly on the real mask xref: on this
    PyMuPDF version `get_image_info(xrefs=True)` doesn't itself enumerate a
    raw content-stream-drawn `/ImageMask` as a placement, so exercising the
    guard through `extract_images` alone wouldn't actually reach it."""
    path, mask_xref = _build_with_standalone_stencil_mask(tmp_pdf_path)
    with Reader(path) as reader:
        page = reader.get_page(0)
        assert image_extract._extract_xref_pixels(page, mask_xref) is None


def test_page_with_standalone_stencil_mask_extracts_the_real_image(tmp_pdf_path):
    path, _mask_xref = _build_with_standalone_stencil_mask(tmp_pdf_path)
    with Reader(path) as reader:
        images = image_extract.extract_images(reader.get_page(0))

    assert len(images) == 1
    assert images[0].array.shape[:2] == (_H, _W)
