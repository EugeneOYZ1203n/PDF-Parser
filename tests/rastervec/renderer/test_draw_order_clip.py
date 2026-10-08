"""Paint order + clipping of the final reconstructed page: vectors must be
painted in source (seqno) order, and a path clipped in the source must stay
clipped in the reconstruction."""
from __future__ import annotations

import numpy as np
import pymupdf as fitz
import pytest

from rastervec.P1_Reading_Native.reader import Reader
from rastervec.P1_Reading_Native.vector_extract import extract_vectors
from rastervec.P4_Output_Organization.organize import organize_outputs
from rastervec.P4_Output_Organization.render import render_output_page


def _pdf_with_stream(tmp_pdf_path, stream: bytes) -> str:
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    shape = page.new_shape()
    shape.draw_line((0, 0), (1, 1))
    shape.finish()
    shape.commit()
    doc.update_stream(page.get_contents()[0], stream)
    return tmp_pdf_path(doc)


def _extract(path):
    reader = Reader(path)
    try:
        page = reader.get_page(0)
        return page, extract_vectors(page)
    finally:
        reader.close()


def _rgb(image, x, y):
    return tuple(int(c) for c in np.asarray(image.convert("RGB"))[y, x])


def test_clipped_path_stays_clipped_in_reconstruction(tmp_pdf_path):
    # Blue line spans x 10..190 but is clipped to x 50..100 (PDF y-up: the
    # line sits at page-space y 125, the clip at y 100..150).
    path = _pdf_with_stream(
        tmp_pdf_path, b"q 50 50 50 50 re W n 0 0 1 RG 4 w 10 75 m 190 75 l S Q",
    )
    page, vectors = _extract(path)
    assert len(vectors) == 1
    assert vectors[0].scissor == (50.0, 100.0, 100.0, 150.0)

    image = render_output_page(page.meta, [], vectors)

    inside = _rgb(image, 75, 125)
    assert inside[2] > 200 and inside[0] < 80          # blue inside the clip
    assert _rgb(image, 150, 125) == (255, 255, 255)   # nothing past it
    assert _rgb(image, 20, 125) == (255, 255, 255)


def test_shuffled_p3_output_is_painted_in_source_order(tmp_pdf_path):
    # Black line, then a white filled rect over its middle (a mask).
    path = _pdf_with_stream(
        tmp_pdf_path, b"0 0 0 RG 4 w 10 100 m 190 100 l S 1 1 1 rg 80 90 40 20 re f",
    )
    page, vectors = _extract(path)
    assert len(vectors) == 2

    # A P3 backend handing them back mask-first must not bury the mask.
    _texts, ordered = organize_outputs([], [], [], list(reversed(vectors)), page)
    image = render_output_page(page.meta, [], ordered)

    assert _rgb(image, 100, 100) == (255, 255, 255)   # masked
    assert _rgb(image, 30, 100)[0] < 80                # line elsewhere


# A "D" outline of radius 50 about PDF (100, 100): right half two cubic
# arcs, left half straight lines -- a closed `c`+`l` clip path. Page space is
# y-down on this 200x200 page, so the shape is symmetric about y 100 either way.
_K = 0.5523 * 50
_D_CLIP = (
    f"100 50 m {100 + _K:g} 50 150 {100 - _K:g} 150 100 c "
    f"150 {100 + _K:g} {100 + _K:g} 150 100 150 c 50 150 l 50 50 l h"
)


def test_rect_clipped_by_curved_path_keeps_the_curve(tmp_pdf_path):
    path = _pdf_with_stream(
        tmp_pdf_path, f"q {_D_CLIP} W n 0 0 1 rg 50 50 100 100 re f Q".encode(),
    )
    page, vectors = _extract(path)
    assert len(vectors) == 1
    assert vectors[0].scissor == pytest.approx((50.0, 50.0, 150.0, 150.0))
    assert len(vectors[0].clips) == 1
    items, even_odd = vectors[0].clips[0]
    assert {it[0] for it in items} == {"c", "l"} and even_odd is False

    image = render_output_page(page.meta, [], vectors)

    assert _rgb(image, 100, 100)[2] > 200            # inside the curve
    assert _rgb(image, 60, 60)[2] > 200              # straight (left) half
    assert _rgb(image, 145, 55) == (255, 255, 255)   # bbox corner past the arc
    assert _rgb(image, 145, 145) == (255, 255, 255)


def test_even_odd_clip_leaves_its_hole_unpainted(tmp_pdf_path):
    # Two nested squares as one even-odd clip -> a square ring.
    path = _pdf_with_stream(
        tmp_pdf_path,
        b"q 40 40 m 160 40 l 160 160 l 40 160 l h "
        b"80 80 m 120 80 l 120 120 l 80 120 l h W* n "
        b"1 0 0 rg 0 0 200 200 re f Q",
    )
    page, vectors = _extract(path)
    assert len(vectors) == 1
    assert len(vectors[0].clips) == 1 and vectors[0].clips[0][1] is True

    image = render_output_page(page.meta, [], vectors)

    assert _rgb(image, 50, 50)[0] > 200 and _rgb(image, 50, 50)[2] < 80   # ring
    assert _rgb(image, 100, 100) == (255, 255, 255)                        # hole
    assert _rgb(image, 20, 20) == (255, 255, 255)                          # outside
