from __future__ import annotations

from pathlib import Path

import pymupdf as fitz
import pytest

from rastervec.Reader.reader import Reader
from rastervec.Vector import vector

REFERENCES_DIR = Path(__file__).resolve().parents[2] / "references"
REFERENCE_PDFS = sorted(REFERENCES_DIR.glob("test_pdfs_*.pdf"))


def _build_test_page(tmp_pdf_path) -> "Reader":
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)

    bg = page.new_shape()
    bg.draw_rect(fitz.Rect(0, 0, 200, 200))
    bg.finish(color=None, fill=(1, 1, 1))
    bg.commit()

    for i in range(4):
        shape = page.new_shape()
        shape.draw_rect(fitz.Rect(10 + i * 5, 10, 10 + i * 5 + 3, 16))
        shape.finish(color=None, fill=(0, 0, 0))
        shape.commit()

    line = page.new_shape()
    line.draw_line((100, 100), (180, 180))
    line.finish(color=(0, 0, 0), width=2)
    line.commit()

    path = tmp_pdf_path(doc)
    return Reader(path)


def test_extract_vectors_count_matches_get_drawings(tmp_pdf_path):
    with _build_test_page(tmp_pdf_path) as reader:
        page = reader.get_page(0)
        vectors = vector.extract_vectors(page)
        n_drawings = len(page.fitz_page.get_drawings())

    assert len(vectors) == n_drawings
    assert all(v.page_index == 0 for v in vectors)
    kinds = {item[0] for v in vectors for item in v.items}
    assert kinds <= {"l", "re", "qu", "c"}


def test_extract_vectors_round_trip_to_pymupdf_matches_source(tmp_pdf_path):
    with _build_test_page(tmp_pdf_path) as reader:
        page = reader.get_page(0)
        vectors = vector.extract_vectors(page)
        drawings = page.fitz_page.get_drawings()

    for v, drawing in zip(vectors, drawings):
        back = v.to_pymupdf()
        assert back["rect"] == drawing["rect"]
        assert back["type"] == drawing["type"]
        assert [item[0] for item in back["items"]] == [item[0] for item in drawing["items"]]


def test_extract_vectors_rect_bbox_matches(tmp_pdf_path):
    with _build_test_page(tmp_pdf_path) as reader:
        page = reader.get_page(0)
        vectors = vector.extract_vectors(page)

    small_rects = [
        v for v in vectors
        if any(item[0] == "re" for item in v.items) and (v.bbox[2] - v.bbox[0]) < 10
    ]
    assert len(small_rects) == 4
    first = sorted(small_rects, key=lambda v: v.bbox[0])[0]
    assert first.bbox == pytest.approx((10, 10, 13, 16))
    assert first.fill == (0.0, 0.0, 0.0)


@pytest.mark.skipif(not REFERENCE_PDFS, reason="tests/references/test_pdfs_*.pdf not generated")
@pytest.mark.parametrize("pdf_path", REFERENCE_PDFS, ids=lambda p: p.stem)
def test_extract_vectors_matches_reference_pdf_drawings(pdf_path):
    with Reader(str(pdf_path)) as reader:
        page = reader.get_page(0)
        vectors = vector.extract_vectors(page)
        n_drawings = len(page.fitz_page.get_drawings())

    assert len(vectors) == n_drawings
    assert len(vectors) > 0


def test_separate_by_layer_groups_by_layer_field(vector):
    from rastervec.Vector.vector import separate_by_layer

    vectors = [
        vector(layer="A"),
        vector(layer="A"),
        vector(layer="B"),
        vector(layer=None),
    ]

    groups = separate_by_layer(vectors)

    assert set(groups.keys()) == {"A", "B", ""}
    assert len(groups["A"]) == 2
    assert len(groups["B"]) == 1
    assert len(groups[""]) == 1
    for key, members in groups.items():
        assert all((v.layer or "") == key for v in members)


def test_separate_by_color_groups_by_color_fill_and_opacity(vector):
    from rastervec.Vector.vector import separate_by_color

    v0 = vector(color=(1, 0, 0), fill=(0, 1, 0))
    v1 = vector(color=None, fill=(0, 1, 0))
    v2 = vector(color=None, fill=None)
    # same color/fill as v0 but different opacity -> its own group.
    v3 = vector(color=(1, 0, 0), fill=(0, 1, 0), stroke_opacity=0.5)
    vectors = [v0, v1, v2, v3]

    groups = separate_by_color(vectors)

    assert groups[(1, 0, 0), (0, 1, 0), None, None] == [v0]
    assert groups[None, (0, 1, 0), None, None] == [v1]
    assert groups[None, None, None, None] == [v2]
    assert groups[(1, 0, 0), (0, 1, 0), 0.5, None] == [v3]
    for key, members in groups.items():
        for v in members:
            assert (v.color, v.fill, v.stroke_opacity, v.fill_opacity) == key


@pytest.mark.skipif(not REFERENCE_PDFS, reason="tests/references/test_pdfs_*.pdf not generated")
def test_separation_invariants_hold_for_reference_pdf_vectors():
    from rastervec.Vector.vector import separate_by_color, separate_by_layer

    with Reader(str(REFERENCE_PDFS[0])) as reader:
        page = reader.get_page(0)
        vectors = vector.extract_vectors(page)

    by_layer = separate_by_layer(vectors)
    for layer, members in by_layer.items():
        assert all((v.layer or "") == layer for v in members)

    by_color = separate_by_color(vectors)
    for key, members in by_color.items():
        for v in members:
            assert (v.color, v.fill, v.stroke_opacity, v.fill_opacity) == key
