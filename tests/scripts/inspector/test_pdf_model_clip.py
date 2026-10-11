import pymupdf as fitz
import pytest

from scripts.inspector import pdf_model
from scripts.inspector.layers import build_layers, filter_items


def _page_with_content(stream: bytes) -> tuple[fitz.Document, fitz.Page]:
    doc = fitz.open()
    page = doc.new_page(width=300, height=300)
    xref = doc.get_new_xref()
    doc.update_object(xref, "<<>>")
    doc.update_stream(xref, stream)
    doc.xref_set_key(page.xref, "Contents", f"{xref} 0 R")
    return doc, doc[0]


def _png(gray: bool) -> bytes:
    cs = fitz.csGRAY if gray else fitz.csRGB
    pix = fitz.Pixmap(cs, fitz.IRect(0, 0, 8, 8), False)
    pix.set_rect(pix.irect, (128,) if gray else (200, 50, 50))
    return pix.tobytes("png")


@pytest.fixture
def clipped_page():
    # Clip to a 100x50 rect (PDF y-up at y=10 -> page-space y 240..290),
    # then stroke a diagonal line through it.
    doc, page = _page_with_content(
        b"q 10 10 100 50 re W n 0 0 1 RG 0 0 m 200 200 l S Q"
    )
    yield page
    doc.close()


def test_clip_path_extracted(clipped_page):
    items = pdf_model.extract_clip_mask_items(clipped_page)

    assert len(items) == 1
    item = items[0]
    assert item.attrs["mask_source"] == "clip_path"
    assert item.attrs["kind"] == "re"
    assert tuple(item.bbox) == pytest.approx((10, 240, 110, 290))
    assert item.metadata["scissor"] == pytest.approx((10, 240, 110, 290))


def test_clip_path_not_in_drawings_layer(clipped_page):
    drawings = pdf_model.extract_drawing_items(clipped_page)

    assert [item.attrs["kind"] for item in drawings] == ["l"]


def test_soft_mask_image_extracted():
    doc = fitz.open()
    page = doc.new_page(width=300, height=300)
    rect = fitz.Rect(20, 30, 120, 130)
    page.insert_image(rect, stream=_png(gray=False), mask=_png(gray=True))

    items = pdf_model.extract_clip_mask_items(page)

    assert [item.attrs["mask_source"] for item in items] == ["soft_mask"]
    item = items[0]
    assert tuple(item.bbox) == pytest.approx(tuple(rect))
    assert item.attrs["xref"] != item.attrs["parent_xref"]
    assert item.metadata["mask_width_px"] == 8
    doc.close()


def test_plain_page_has_no_clip_masks():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "hello")
    page.draw_line((0, 0), (100, 100))

    assert pdf_model.extract_clip_mask_items(page) == []
    doc.close()


def test_mask_source_filter():
    doc = fitz.open()
    page = doc.new_page(width=300, height=300)
    page.insert_image(
        fitz.Rect(20, 30, 120, 130), stream=_png(gray=False), mask=_png(gray=True)
    )
    items = pdf_model.extract_clip_mask_items(page)

    assert len(filter_items(items, {"mask_source": {"soft_mask"}})) == 1
    assert filter_items(items, {"mask_source": {"clip_path"}}) == []
    doc.close()


def test_clip_masks_layer_registered():
    layers = {layer.key: layer for layer in build_layers(pdf_model)}

    assert layers["clip_masks"].extractor is pdf_model.extract_clip_mask_items
    assert not layers["clip_masks"].enabled_default
