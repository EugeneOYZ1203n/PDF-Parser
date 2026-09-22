from __future__ import annotations

from rastervec.commons.models import Page
from rastervec.P3_Vector_Parsing.VectorClassification import parse as vectorclassification


def _page(page_meta) -> Page:
    return Page(doc_path="synthetic.pdf", meta=page_meta(width=200.0, height=100.0), fitz_page=None)


def test_parse_empty_input_returns_empty_output(page_meta):
    drawing, texts = vectorclassification.parse([], [], _page(page_meta), verbose=True)
    assert drawing == []
    assert texts == []


def test_parse_debug_out_has_fast_result_and_word_group_keys(page_meta):
    debug_out: dict = {}
    vectorclassification.parse([], [], _page(page_meta), verbose=True, debug_out=debug_out)
    assert debug_out["fast_result"] is not None
    assert debug_out["word_groups"] == []
    assert debug_out["texts"] == []
    assert debug_out["ocr_crops"] == []
    assert debug_out["cluster_detections"] == []


def test_parse_streaming_matches_batch_render_debug(page_meta):
    page = _page(page_meta)

    streamed: list[tuple] = []
    vectorclassification.parse(
        [], [], page, verbose=True, on_debug_layer=lambda *layer: streamed.append(layer),
    )

    debug_out: dict = {}
    vectorclassification.parse([], [], page, verbose=True, debug_out=debug_out)
    batch_layers = vectorclassification.render_debug(debug_out, page.meta)

    assert [layer[0] for layer in streamed] == [layer[0] for layer in batch_layers]
    for stage, label, hexcolor, pdf_bytes in streamed:
        assert isinstance(pdf_bytes, (bytes, bytearray))
        assert pdf_bytes[:4] == b"%PDF"
        assert hexcolor.startswith("#")
