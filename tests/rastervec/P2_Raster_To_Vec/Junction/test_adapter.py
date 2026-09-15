from __future__ import annotations

import numpy as np

from rastervec.commons.models import Image, Page
from rastervec.P2_Raster_To_Vec.Junction import adapter


def _line_image(width: int = 60, height: int = 40) -> Image:
    arr = np.full((height, width), 255, dtype=np.uint8)
    arr[height // 2, 5:width - 5] = 0  # one horizontal black line
    return Image(array=arr, bbox=(0.0, 0.0, float(width), float(height)), dpi=72.0, source="page")


def _page(page_meta) -> Page:
    return Page(doc_path="synthetic.pdf", meta=page_meta(width=60.0, height=40.0), fitz_page=None)


def test_extract_streams_layers_and_matches_batch_render_debug(page_meta):
    page = _page(page_meta)
    images = [_line_image()]

    seen: list[tuple] = []
    vectors, texts = adapter.extract(
        images, page, on_debug_layer=lambda *layer: seen.append(layer),
    )
    assert texts == []
    stages_streamed = [layer[0] for layer in seen]

    debug_out: dict = {}
    vectors2, _texts2 = adapter.extract(images, page, debug_out=debug_out)
    batch_layers = adapter.render_debug(debug_out, page.meta)
    stages_batch = [layer[0] for layer in batch_layers]

    assert stages_streamed == stages_batch
    assert len(vectors) == len(vectors2)  # same input -> same vector count
    for stage, label, hexcolor, pdf_bytes in seen:
        assert isinstance(pdf_bytes, (bytes, bytearray))
        assert pdf_bytes[:4] == b"%PDF"
        assert hexcolor.startswith("#")


def test_extract_no_debug_args_is_unaffected(page_meta):
    page = _page(page_meta)
    vectors, texts = adapter.extract([_line_image()], page)
    assert texts == []
    assert isinstance(vectors, list)
