from __future__ import annotations

from rastervec.commons.models import Page
from rastervec.P3_Vector_Parsing.LegacyRecreation import parse as legacyrecreation


def _page(page_meta) -> Page:
    return Page(doc_path="synthetic.pdf", meta=page_meta(width=200.0, height=100.0), fitz_page=None)


def test_parse_empty_input_returns_empty_output(page_meta):
    drawing, texts = legacyrecreation.parse([], [], _page(page_meta))
    assert drawing == []
    assert texts == []


def test_parse_streaming_matches_batch_render_debug(page_meta):
    page = _page(page_meta)

    streamed: list[tuple] = []
    legacyrecreation.parse([], [], page, on_debug_layer=lambda *layer: streamed.append(layer))

    debug_out: dict = {}
    legacyrecreation.parse([], [], page, debug_out=debug_out)
    batch_layers = legacyrecreation.render_debug(debug_out, page.meta)

    assert [layer[0] for layer in streamed] == [layer[0] for layer in batch_layers]
    assert [layer[0] for layer in streamed] == ["filter_fill", "group_words", "ocr", "drawing"]
    for stage, label, hexcolor, pdf_bytes in streamed:
        assert isinstance(pdf_bytes, (bytes, bytearray))
        assert pdf_bytes[:4] == b"%PDF"
        assert hexcolor.startswith("#")
