"""Paint order + clipping of the final reconstructed page: vectors must be
painted in source (seqno) order, and a path clipped in the source must stay
clipped in the reconstruction."""
from __future__ import annotations

import numpy as np
import pymupdf as fitz

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
