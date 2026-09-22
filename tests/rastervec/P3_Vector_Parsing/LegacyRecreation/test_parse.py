from __future__ import annotations

import numpy as np

from rastervec.commons.models import Page
from rastervec.P3_Vector_Parsing.LegacyRecreation import parse as legacyrecreation
from rastervec.P3_Vector_Parsing.LegacyRecreation.config import RENDER_PADDING_EXTRA_PT
from rastervec.P3_Vector_Parsing.LegacyRecreation.paddle_engine import OcrBox, PaddleDetectBackend, PaddleRecBackend


def _page(page_meta) -> Page:
    return Page(doc_path="synthetic.pdf", meta=page_meta(width=200.0, height=100.0), fitz_page=None)


def test_cluster_render_padding_zero_width_vector(vector):
    v = vector(bbox=(0.0, 0.0, 10.0, 10.0), width=None)
    assert legacyrecreation._cluster_render_padding([v]) == RENDER_PADDING_EXTRA_PT


def test_cluster_render_padding_scales_with_max_stroke_width(vector):
    v1 = vector(bbox=(0.0, 0.0, 10.0, 10.0), width=2.0)
    v2 = vector(bbox=(0.0, 0.0, 10.0, 10.0), width=6.0)
    assert legacyrecreation._cluster_render_padding([v1, v2]) == 6.0 / 2.0 + RENDER_PADDING_EXTRA_PT


def test_parse_empty_input_returns_empty_output(page_meta):
    drawing, texts = legacyrecreation.parse([], [], _page(page_meta))
    assert drawing == []
    assert texts == []


def test_parse_debug_out_has_ocr_crops_key(page_meta):
    debug_out: dict = {}
    legacyrecreation.parse([], [], _page(page_meta), debug_out=debug_out)
    assert debug_out["ocr_crops"] == []


def test_parse_keeps_ocr_crop_for_blank_recognition(page_meta, vector, monkeypatch):
    """A detected quad whose PaddleOCR recognition comes back blank must
    still be recorded in `ocr_crops` (for debug-image dumping) -- it just
    shouldn't become a real output `Text`. Regression test for the bug where
    `ocr_crops.append` sat after the `if not box.text: continue` guard, so
    blank recognitions were silently dropped from the debug-crop list."""
    glyph = vector(kind="c", type="f", fill=(0.0, 0.0, 0.0), bbox=(10.0, 10.0, 20.0, 20.0), seqno=1)

    quad = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    monkeypatch.setattr(PaddleDetectBackend, "detect", lambda self, bgr: [quad])
    monkeypatch.setattr(
        PaddleRecBackend, "recognize_crops",
        lambda self, crops: [OcrBox(text="", confidence=0.0, flip_deg=0) for _ in crops],
    )

    debug_out: dict = {}
    drawing, texts = legacyrecreation.parse([glyph], [], _page(page_meta), debug_out=debug_out)

    assert texts == []  # a blank recognition never becomes a real Text
    assert len(debug_out["ocr_crops"]) == 1  # but its crop is still captured for debugging
    assert debug_out["ocr_crops"][0][1] == ""


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
