from __future__ import annotations

from rastervec.Evaluation.Labelling.label_schema import LabelEntry, LabelSet
from rastervec.Evaluation.Labelling.raster_label import (
    embedded_images_for_page,
    raster_geometry_for_page,
    sync_text_from_vector_labels,
)


def test_raster_geometry_for_page_covers_every_drawing(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory(
        [
            {
                "drawings": [
                    {"rects": [(10, 10, 30, 20)], "color": (0, 0, 0), "width": 1.0},
                    {"lines": [((5, 5), (50, 50))], "color": (1, 0, 0), "width": 2.0},
                ],
            }
        ]
    )
    path = tmp_pdf_path(doc)

    annotations = raster_geometry_for_page(path, 0)

    assert all(a.source == "auto" for a in annotations)
    # the rect becomes 4 lines, the line stays 1 -- at least 5 total
    assert len(annotations) >= 5
    kinds = {a.kind for a in annotations}
    assert kinds == {"l"}


def test_raster_geometry_for_page_empty_page_returns_nothing(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory([{}])
    path = tmp_pdf_path(doc)

    assert raster_geometry_for_page(path, 0) == []


def _vector_entry(page_index: int, label_id: str, text: str) -> LabelEntry:
    return LabelEntry(
        page_index=page_index, cluster_bbox=(0, 0, 10, 5), cluster_signature="s",
        label_id=label_id, text=text, source="vector", expected_rotation=90,
        vector_signatures=["sig1"],
    )


def test_sync_text_from_vector_labels_adds_prefixed_entries():
    raster_labels = LabelSet(pdf_path="x.pdf")
    vector_labels = LabelSet(
        pdf_path="x.pdf",
        entries=[_vector_entry(0, "v1", "Hello"), _vector_entry(1, "v2", "World")],
    )

    sync_text_from_vector_labels(raster_labels, vector_labels, page_indices=[0, 1])

    assert len(raster_labels.entries) == 2
    ids = {e.label_id for e in raster_labels.entries}
    assert ids == {"vecsync:v1", "vecsync:v2"}
    assert all(e.source == "raster" for e in raster_labels.entries)
    synced = next(e for e in raster_labels.entries if e.label_id == "vecsync:v1")
    assert synced.text == "Hello"
    assert synced.expected_rotation == 90
    assert synced.vector_signatures == ["sig1"]


def test_sync_text_from_vector_labels_replaces_only_its_own_entries():
    manual_entry = LabelEntry(
        page_index=0, cluster_bbox=(0, 0, 1, 1), cluster_signature="raster:0:0",
        label_id="manual-uuid", text="hand labelled", source="raster",
    )
    raster_labels = LabelSet(pdf_path="x.pdf", entries=[manual_entry])
    vector_labels = LabelSet(pdf_path="x.pdf", entries=[_vector_entry(0, "v1", "Hello")])

    sync_text_from_vector_labels(raster_labels, vector_labels, page_indices=[0])
    assert len(raster_labels.entries) == 2
    assert any(e.label_id == "manual-uuid" for e in raster_labels.entries)
    assert any(e.label_id == "vecsync:v1" for e in raster_labels.entries)

    # re-running with an updated vector label text replaces only the vecsync entry
    vector_labels.entries[0].text = "Updated"
    sync_text_from_vector_labels(raster_labels, vector_labels, page_indices=[0])
    assert len(raster_labels.entries) == 2
    synced = next(e for e in raster_labels.entries if e.label_id == "vecsync:v1")
    assert synced.text == "Updated"
    assert any(e.label_id == "manual-uuid" and e.text == "hand labelled" for e in raster_labels.entries)


def test_embedded_images_for_page_finds_inserted_image(synthetic_pdf_factory, tmp_pdf_path, tmp_path):
    import pymupdf as fitz

    doc = synthetic_pdf_factory([{}])
    page = doc[0]
    # 1x1 red pixel PNG
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
        b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    page.insert_image(fitz.Rect(10, 10, 40, 40), stream=png_bytes)
    path = tmp_pdf_path(doc)

    regions = embedded_images_for_page(path, 0)

    assert len(regions) == 1
    assert regions[0].page_index == 0
    x0, y0, x1, y1 = regions[0].bbox
    assert (x0, y0) == (10.0, 10.0)
    assert (x1, y1) == (40.0, 40.0)
