from __future__ import annotations

from rastervec.commons.models import Page
from rastervec.P3_Vector_Parsing.FastIntoPaddle import parse as fastintopaddle


def _page(page_meta) -> Page:
    return Page(doc_path="synthetic.pdf", meta=page_meta(width=200.0, height=100.0), fitz_page=None)


def test_parse_empty_input_returns_empty_output(page_meta):
    drawing, texts = fastintopaddle.parse([], [], _page(page_meta))
    assert drawing == []
    assert texts == []


def test_parse_streams_one_layer_call_per_stage(page_meta):
    page = _page(page_meta)
    seen: list[tuple] = []
    drawing, texts = fastintopaddle.parse(
        [], [], page, on_debug_layer=lambda *layer: seen.append(layer),
    )

    stages_seen = [layer[0] for layer in seen]
    # every named stage this backend renders, in the order parse() runs
    # its own chain -- confirms layers stream out during the run rather
    # than only being available after the whole chain finishes.
    assert stages_seen == [
        "similarity", "fast", "fast", "reclassify", "clusters",
        "paddle_detect", "paddle_detect", "assignment", "assignment",
        "rotate", "ocr", "drawing",
    ]
    for stage, label, hexcolor, pdf_bytes in seen:
        assert isinstance(pdf_bytes, (bytes, bytearray))
        assert pdf_bytes[:4] == b"%PDF"
        assert hexcolor.startswith("#")


def test_parse_debug_out_batch_mode_still_populates_and_render_debug_matches(page_meta):
    page = _page(page_meta)
    debug_out: dict = {}
    drawing, texts = fastintopaddle.parse([], [], page, debug_out=debug_out)

    assert set(debug_out) == {
        "similarity_groups", "fast", "reclassify", "clusters", "cluster_detections",
        "reassignment", "segments", "texts", "drawing",
    }

    batch_layers = fastintopaddle.render_debug(debug_out, page.meta)
    stages_batch = [layer[0] for layer in batch_layers]

    streamed: list[tuple] = []
    fastintopaddle.parse([], [], page, on_debug_layer=lambda *layer: streamed.append(layer))
    stages_streamed = [layer[0] for layer in streamed]

    assert stages_batch == stages_streamed


def test_parse_on_debug_layer_and_debug_out_are_independent(page_meta):
    page = _page(page_meta)
    debug_out: dict = {}
    seen: list[tuple] = []
    fastintopaddle.parse(
        [], [], page, debug_out=debug_out, on_debug_layer=lambda *layer: seen.append(layer),
    )
    assert debug_out  # batch stash still happened
    assert seen  # streaming callback still happened
