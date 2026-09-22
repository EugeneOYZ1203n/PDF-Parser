from __future__ import annotations

import numpy as np

from rastervec.commons.models import Page
from rastervec.P3_Vector_Parsing.VectorClassification import parse as vectorclassification
from rastervec.P3_Vector_Parsing.VectorClassification.paddle_engine import OcrBox, PaddleDetectBackend, PaddleRecBackend


def _page(page_meta) -> Page:
    return Page(doc_path="synthetic.pdf", meta=page_meta(width=200.0, height=100.0), fitz_page=None)


def test_parse_empty_input_returns_empty_output(page_meta):
    drawing, texts = vectorclassification.parse([], [], _page(page_meta), verbose=True)
    assert drawing == []
    assert texts == []


def test_parse_debug_out_has_fast_result_key(page_meta):
    debug_out: dict = {}
    vectorclassification.parse([], [], _page(page_meta), verbose=True, debug_out=debug_out)
    assert debug_out["fast_result"] is not None
    assert debug_out["fast_passed"] == []
    assert debug_out["texts"] == []
    assert debug_out["ocr_crops"] == []
    assert debug_out["cluster_detections"] == []


def test_parse_keeps_ocr_crop_for_blank_recognition(page_meta, vector, monkeypatch):
    """A detected quad whose PaddleOCR recognition comes back blank must
    still be recorded in `ocr_crops` (for debug-image dumping) -- it just
    shouldn't become a real output `Text`. Regression test for the bug where
    `ocr_crops.append` sat after the `if not box.text: continue` guard, so
    blank recognitions were silently dropped from the debug-crop list."""
    v = vector(kind="l", bbox=(10.0, 10.0, 20.0, 20.0), color=(0.0, 0.0, 0.0), seqno=1)

    quad = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    monkeypatch.setattr(PaddleDetectBackend, "detect", lambda self, bgr: [quad])
    monkeypatch.setattr(
        PaddleRecBackend, "recognize_crops",
        lambda self, crops: [OcrBox(text="", confidence=0.0, flip_deg=0) for _ in crops],
    )

    debug_out: dict = {}
    drawing, texts = vectorclassification.parse(
        [v], [], _page(page_meta), enable_fast=False, debug_out=debug_out,
    )

    assert texts == []  # a blank recognition never becomes a real Text
    assert len(debug_out["ocr_crops"]) == 1  # but its crop is still captured for debugging
    assert debug_out["ocr_crops"][0][1] == ""
    assert len(debug_out["ocr_blank_boxes"]) == 1  # and its page-space bbox is recorded for debug rendering


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
