from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from raster_parser.phase_native_read.reader import Reader
from raster_parser.phase_native_read import vector_extract as vector

REFERENCES_DIR = Path(__file__).resolve().parents[2] / "references"
REFERENCE_PDFS = sorted(REFERENCES_DIR.glob("test_pdfs_*.pdf"))


def _build_test_page(tmp_pdf_path) -> "Reader":
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)

    bg = page.new_shape()
    bg.draw_rect(pymupdf.Rect(0, 0, 200, 200))
    bg.finish(color=None, fill=(1, 1, 1))
    bg.commit()

    for i in range(4):
        shape = page.new_shape()
        shape.draw_rect(pymupdf.Rect(10 + i * 5, 10, 10 + i * 5 + 3, 16))
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
        n_drawings = len(page.pymupdf_page.get_drawings())

    assert len(vectors) == n_drawings
    assert all(v.page_index == 0 for v in vectors)
    kinds = {item[0] for v in vectors for item in v.items}
    assert kinds <= {"l", "re", "qu", "c"}


def test_extract_vectors_round_trip_to_pymupdf_matches_source(tmp_pdf_path):
    with _build_test_page(tmp_pdf_path) as reader:
        page = reader.get_page(0)
        vectors = vector.extract_vectors(page)
        drawings = page.pymupdf_page.get_drawings()

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
        v
        for v in vectors
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
        n_drawings = len(page.pymupdf_page.get_drawings())

    assert len(vectors) == n_drawings
    assert len(vectors) > 0


# A page with a Multiply-blended red stroke over a solid-cyan rect, plus a
# plain (no blend) black stroke as a control. `/BM` never reaches the path
# dict from plain get_drawings(), so extract_vectors must read it off the
# enclosing group via get_drawings(extended=True).
_BLEND_PDF = b"""%PDF-1.7
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R\
/Resources<</ExtGState<</GSm<</BM/Multiply>>>>>>>>endobj
4 0 obj<</Length 110>>stream
0 1 1 rg 0 70 200 60 re f
q /GSm gs 1 0 0 RG 30 w 0 100 m 200 100 l S Q
0 0 0 RG 4 w 0 30 m 200 30 l S
endstream endobj
trailer<</Root 1 0 R>>
"""


def _blend_pdf_reader(tmp_path):
    path = tmp_path / "blend.pdf"
    path.write_bytes(_BLEND_PDF)
    return Reader(str(path))


def test_extract_vectors_folds_group_blend_mode_onto_vector(tmp_path):
    with _blend_pdf_reader(tmp_path) as reader:
        page = reader.get_page(0)
        vectors = vector.extract_vectors(page)
        n_paths = len(
            [
                d
                for d in page.pymupdf_page.get_drawings(extended=True)
                if d.get("type") in ("s", "f", "fs")
            ]
        )

    # group/clip wrapper entries never become Vectors
    assert len(vectors) == n_paths
    assert [v.seqno for v in vectors] == list(range(len(vectors)))

    multiply = [v for v in vectors if v.blendmode == "Multiply"]
    assert len(multiply) == 1
    assert multiply[0].color == (1.0, 0.0, 0.0)

    # the plain black stroke stays un-blended
    plain = [v for v in vectors if v.blendmode is None]
    assert len(plain) == len(vectors) - 1
    assert all(v.opacity is None for v in plain)


def _stream_page(tmp_pdf_path, stream: bytes) -> "Reader":
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)
    shape = page.new_shape()
    shape.draw_line((0, 0), (1, 1))
    shape.finish()
    shape.commit()
    doc.update_stream(page.get_contents()[0], stream)
    return Reader(tmp_pdf_path(doc))


@pytest.mark.parametrize(
    "clip",
    [
        b"50 50 100 100 re",  # a single `re`
        b"50 50 m 150 50 l 150 150 l 50 150 l h",  # four axis-aligned `l`s
    ],
)
def test_rectangular_clip_is_scissor_only(tmp_pdf_path, clip):
    with _stream_page(tmp_pdf_path, b"q " + clip + b" W n 0 0 1 rg 20 20 160 160 re f Q") as reader:
        (v,) = vector.extract_vectors(reader.get_page(0))
    assert v.scissor == pytest.approx((50.0, 50.0, 150.0, 150.0))
    assert v.clips == ()


def test_nested_rect_and_curved_clip_keeps_only_the_curved_shape(tmp_pdf_path):
    k = 0.5523 * 40
    curved = (
        f"100 60 m {100 + k:g} 60 140 {100 - k:g} 140 100 c "
        f"140 {100 + k:g} {100 + k:g} 140 100 140 c 60 140 l 60 60 l h"
    ).encode()
    stream = b"q 10 10 180 180 re W n q " + curved + b" W n 0 0 1 rg 0 0 200 200 re f Q Q"
    with _stream_page(tmp_pdf_path, stream) as reader:
        (v,) = vector.extract_vectors(reader.get_page(0))
    assert v.scissor == pytest.approx((60.0, 60.0, 140.0, 140.0))
    assert len(v.clips) == 1
    items, even_odd = v.clips[0]
    assert [it[0] for it in items][:2] == ["c", "c"] and even_odd is False


def test_unclipped_path_has_no_clips(tmp_pdf_path):
    with _stream_page(tmp_pdf_path, b"0 0 1 rg 20 20 50 50 re f") as reader:
        (v,) = vector.extract_vectors(reader.get_page(0))
    assert v.scissor is None and v.clips == ()
