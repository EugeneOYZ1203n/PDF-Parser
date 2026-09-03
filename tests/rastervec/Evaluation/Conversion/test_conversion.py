from __future__ import annotations

import pymupdf as fitz

from rastervec.Evaluation.Conversion.conversion import (
    convert_page_drawings_only,
    convert_page_text_only,
    convert_page_to_vector_text,
)


def _page_with_text_and_vector(synthetic_pdf_factory):
    doc = synthetic_pdf_factory(
        [{"width": 300, "height": 200,
          "texts": [{"point": (20, 100), "text": "LABEL ME", "fontsize": 16}]}]
    )
    page = doc[0]
    shape = page.new_shape()
    shape.draw_rect(fitz.Rect(40, 40, 120, 80))
    shape.draw_line(fitz.Point(200, 20), fitz.Point(260, 160))
    shape.finish(color=(0, 0, 1), width=1.5)
    shape.commit()
    return doc


def test_convert_page_to_vector_text_moves_text_into_drawings(
    synthetic_pdf_factory, tmp_pdf_path,
):
    doc = synthetic_pdf_factory(
        [{"width": 200, "height": 100, "texts": [{"point": (10, 50), "text": "Hello World", "fontsize": 20}]}]
    )
    path = tmp_pdf_path(doc)

    result_bytes = convert_page_to_vector_text(path, 0)
    converted = fitz.open("pdf", result_bytes)
    page = converted[0]

    assert page.get_text().strip() == ""
    drawings = page.get_drawings()
    assert len(drawings) > 0
    assert page.get_images() == []

    all_items_bbox = fitz.Rect()
    for drawing in drawings:
        all_items_bbox |= drawing["rect"]
    # Roughly the original text's bbox region (page 200x100, text near y=50).
    assert 0 <= all_items_bbox.x0 <= 200
    assert 0 <= all_items_bbox.y1 <= 100


def test_convert_page_to_vector_text_preserves_page_size(
    synthetic_pdf_factory, tmp_pdf_path,
):
    doc = synthetic_pdf_factory(
        [{"width": 300, "height": 150, "texts": [{"point": (5, 20), "text": "Size"}]}]
    )
    path = tmp_pdf_path(doc)

    result_bytes = convert_page_to_vector_text(path, 0)
    converted = fitz.open("pdf", result_bytes)

    assert converted[0].rect.width == 300
    assert converted[0].rect.height == 150


def test_convert_page_to_vector_text_preserves_existing_vectors(
    synthetic_pdf_factory, tmp_pdf_path,
):
    doc = synthetic_pdf_factory(
        [{"width": 300, "height": 200, "texts": [{"point": (20, 100), "text": "LABEL ME", "fontsize": 16}]}]
    )
    page = doc[0]
    shape = page.new_shape()
    shape.draw_rect(fitz.Rect(40, 40, 120, 80))
    shape.draw_line(fitz.Point(200, 20), fitz.Point(260, 160))
    shape.finish(color=(0, 0, 1), width=1.5)
    shape.commit()
    original_rects = sorted(tuple(round(c, 3) for c in d["rect"]) for d in page.get_drawings())
    path = tmp_pdf_path(doc)

    converted = fitz.open("pdf", convert_page_to_vector_text(path, 0))
    cpage = converted[0]
    converted_rects = sorted(tuple(round(c, 3) for c in d["rect"]) for d in cpage.get_drawings())

    # every original drawing rect is still present, byte-identical geometry
    for r in original_rects:
        assert r in converted_rects
    # and new glyph-path drawings have appeared over the text region
    assert len(converted_rects) > len(original_rects)
    assert cpage.get_text().strip() == ""


def test_convert_page_to_vector_text_preserves_rotation(
    synthetic_pdf_factory, tmp_pdf_path,
):
    doc = synthetic_pdf_factory(
        [{"width": 300, "height": 200, "rotation": 90,
          "texts": [{"point": (20, 100), "text": "Rot", "fontsize": 16}]}]
    )
    page = doc[0]
    shape = page.new_shape()
    shape.draw_rect(fitz.Rect(40, 40, 120, 80))
    shape.finish(color=(0, 0, 1))
    shape.commit()
    path = tmp_pdf_path(doc)

    converted = fitz.open("pdf", convert_page_to_vector_text(path, 0))
    cpage = converted[0]

    assert cpage.rotation == 90
    rects = [tuple(round(c, 3) for c in d["rect"]) for d in cpage.get_drawings()]
    # the original rect stays in unrotated MediaBox space, unchanged
    assert (40.0, 40.0, 120.0, 80.0) in rects


def test_convert_page_text_only_drops_drawings(synthetic_pdf_factory, tmp_pdf_path):
    doc = _page_with_text_and_vector(synthetic_pdf_factory)
    original_rects = {tuple(round(c, 3) for c in d["rect"]) for d in doc[0].get_drawings()}
    path = tmp_pdf_path(doc)

    cpage = fitz.open("pdf", convert_page_text_only(path, 0))[0]
    rects = {tuple(round(c, 3) for c in d["rect"]) for d in cpage.get_drawings()}

    assert cpage.get_text().strip() == ""
    assert len(rects) > 0  # glyph paths present
    assert rects.isdisjoint(original_rects)  # the rectangle / line are gone
    assert cpage.get_images() == []


def test_convert_page_drawings_only_keeps_drawings_drops_text(
    synthetic_pdf_factory, tmp_pdf_path,
):
    doc = _page_with_text_and_vector(synthetic_pdf_factory)
    original_rects = sorted(tuple(round(c, 3) for c in d["rect"]) for d in doc[0].get_drawings())
    path = tmp_pdf_path(doc)

    cpage = fitz.open("pdf", convert_page_drawings_only(path, 0))[0]
    converted_rects = sorted(tuple(round(c, 3) for c in d["rect"]) for d in cpage.get_drawings())

    assert cpage.get_text().strip() == ""
    assert converted_rects == original_rects  # byte-for-byte geometry, nothing added


def test_convert_page_drawings_only_preserves_rotation(
    synthetic_pdf_factory, tmp_pdf_path,
):
    doc = synthetic_pdf_factory(
        [{"width": 300, "height": 200, "rotation": 90,
          "texts": [{"point": (20, 100), "text": "Rot", "fontsize": 16}]}]
    )
    shape = doc[0].new_shape()
    shape.draw_rect(fitz.Rect(40, 40, 120, 80))
    shape.finish(color=(0, 0, 1))
    shape.commit()
    path = tmp_pdf_path(doc)

    cpage = fitz.open("pdf", convert_page_drawings_only(path, 0))[0]
    assert cpage.rotation == 90
    rects = [tuple(round(c, 3) for c in d["rect"]) for d in cpage.get_drawings()]
    assert (40.0, 40.0, 120.0, 80.0) in rects


def test_convert_page_to_vector_text_writes_output_path(
    synthetic_pdf_factory, tmp_pdf_path, tmp_path,
):
    doc = synthetic_pdf_factory(
        [{"texts": [{"point": (10, 50), "text": "Save Me"}]}]
    )
    path = tmp_pdf_path(doc)
    out_path = str(tmp_path / "converted.pdf")

    convert_page_to_vector_text(path, 0, output_path=out_path)

    converted = fitz.open(out_path)
    assert converted[0].get_text().strip() == ""
    assert len(converted[0].get_drawings()) > 0
