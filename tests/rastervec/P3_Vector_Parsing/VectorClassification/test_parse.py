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
    assert debug_out["classifier_crops"] == []
    assert debug_out["cluster_detections"] == []
    assert debug_out["rotation"] == []


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


def test_parse_ocr_crop_reflects_post_flip_rotation(page_meta, vector, monkeypatch):
    """`ocr_crops` must store the crop PaddleOCR actually recognised from --
    when the angle classifier decides a crop is upside-down (`flip_deg=180`),
    recognition itself runs on the rotated pixels, so the stashed debug crop
    should be rotated the same way, not the raw pre-flip crop."""
    from rastervec.P3_Vector_Parsing.VectorClassification.paddle_engine import RotationDebug

    v = vector(kind="l", bbox=(10.0, 10.0, 20.0, 20.0), color=(0.0, 0.0, 0.0), seqno=1)

    quad = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    monkeypatch.setattr(PaddleDetectBackend, "detect", lambda self, bgr: [quad])

    raw_crop = np.zeros((4, 4, 3), dtype=np.uint8)
    raw_crop[0, 0] = [255, 0, 0]  # marker pixel in one corner, to detect rotation
    rotation_debug = RotationDebug(quad_angle_deg=0.0, hough_angle_deg=None, combined_angle_deg=0.0)
    monkeypatch.setattr(
        vectorclassification, "hough_deskew", lambda bgr, quad: (raw_crop, rotation_debug),
    )
    monkeypatch.setattr(
        PaddleRecBackend, "recognize_crops",
        lambda self, crops: [OcrBox(text="X", confidence=1.0, flip_deg=180) for _ in crops],
    )

    debug_out: dict = {}
    vectorclassification.parse(
        [v], [], _page(page_meta), enable_fast=False, debug_out=debug_out,
    )

    stored_crop = debug_out["ocr_crops"][0][0]
    pre_flip_crop = raw_crop[:, :, ::-1]  # parse.py's own BGR->RGB reversal
    assert np.array_equal(stored_crop, np.rot90(pre_flip_crop, 2))
    assert not np.array_equal(stored_crop, pre_flip_crop)  # sanity: rotation actually happened

    before_crop, after_crop = debug_out["classifier_crops"][0]
    assert np.array_equal(before_crop, pre_flip_crop)
    assert np.array_equal(after_crop, stored_crop)

    entry = debug_out["rotation"][0]
    assert entry["quad_angle_deg"] == 0.0
    assert entry["hough_angle_deg"] is None
    assert entry["combined_angle_deg"] == 0.0


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


def test_classification_layers_bbox_only_and_skip_unchanged_steps(page_meta):
    from types import SimpleNamespace as NS

    def step(label, bboxes):
        groups = [[NS(bbox=b)] for b in bboxes]
        return NS(label=label, categories={
            "kept": NS(role="kept", groups=groups),
            "dropped": NS(role="dropped", groups=[]),
        })

    a, b = (10.0, 10.0, 20.0, 20.0), (30.0, 30.0, 40.0, 40.0)
    cls = NS(clustering={"bucket": NS(steps=[
        step("first", [a, b]),
        step("same", [b, a]),  # same boxes, different order -> no layer
        step("drop b", [a]),
    ])})
    layers = vectorclassification._render_classification_layers(page_meta(), cls)
    assert [(stage, label) for stage, label, _hex, _pdf in layers] == [
        ("classify_01_first", "kept bbox"),
        ("classify_03_drop_b", "kept bbox"),
    ]


def test_parse_renders_debug_layers_only_with_a_callback(page_meta, monkeypatch):
    calls: list = []
    real = vectorclassification._render_drawing_layers
    monkeypatch.setattr(
        vectorclassification, "_render_drawing_layers",
        lambda *a, **k: calls.append(1) or real(*a, **k),
    )
    page = _page(page_meta)

    steps: dict = {}
    vectorclassification.parse([], [], page, step_durations=steps)
    assert calls == [] and "debug_render" not in steps  # nobody listening -> nothing rendered

    steps = {}
    vectorclassification.parse([], [], page, step_durations=steps, on_debug_layer=lambda *layer: None)
    assert calls == [1] and "debug_render" in steps
