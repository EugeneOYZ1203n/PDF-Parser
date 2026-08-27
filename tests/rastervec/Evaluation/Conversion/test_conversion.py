from __future__ import annotations

import pymupdf as fitz

from rastervec.Evaluation.Conversion.conversion import convert_page_to_vector_text


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
