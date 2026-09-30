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


def test_parse_debug_out_has_no_fast_keys(page_meta):
    """FAST was removed entirely from this backend -- `debug_out` should
    never contain `fast_result`/`fast_passed`/`fast_dropped` (a regression
    guard against accidentally re-introducing them)."""
    debug_out: dict = {}
    vectorclassification.parse([], [], _page(page_meta), verbose=True, debug_out=debug_out)
    assert "fast_result" not in debug_out
    assert "fast_passed" not in debug_out
    assert "fast_dropped" not in debug_out
    assert debug_out["texts"] == []
    assert debug_out["ocr_crops"] == []
    assert debug_out["classifier_crops"] == []
    assert debug_out["cluster_detections"] == []
    assert debug_out["rotation"] == []
    assert debug_out["retry_stats"] == {"0": 0, "1": 0, "2": 0, "3": 0, "failed": 0}


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
    # Every blank-retry pass (+90/180/270) stays blank too, so this
    # detection is permanently unrecoverable -- exercises the "exhausted
    # every retry pass" path without hitting a real PaddleOCR engine.
    monkeypatch.setattr(
        PaddleRecBackend, "recognize_crops_raw",
        lambda self, crops: [OcrBox(text="", confidence=0.0, flip_deg=0) for _ in crops],
    )

    debug_out: dict = {}
    drawing, texts = vectorclassification.parse(
        [v], [], _page(page_meta), debug_out=debug_out,
    )

    assert texts == []  # a blank recognition never becomes a real Text
    assert len(debug_out["ocr_crops"]) == 1  # but its crop is still captured for debugging
    assert debug_out["ocr_crops"][0][1] == ""
    assert len(debug_out["ocr_blank_boxes"]) == 1  # and its page-space bbox is recorded for debug rendering
    assert debug_out["rotation"][0]["retry_count"] is None  # never recovered
    assert debug_out["retry_stats"] == {"0": 0, "1": 0, "2": 0, "3": 0, "failed": 1}


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
    rotation_debug = RotationDebug(hough_angle_deg=None, minarea_angle_deg=0.0, combined_angle_deg=0.0)
    monkeypatch.setattr(
        vectorclassification, "hough_deskew",
        lambda bgr, quad, **kwargs: (raw_crop, rotation_debug),
    )
    monkeypatch.setattr(
        PaddleRecBackend, "recognize_crops",
        lambda self, crops: [OcrBox(text="X", confidence=1.0, flip_deg=180) for _ in crops],
    )

    debug_out: dict = {}
    vectorclassification.parse(
        [v], [], _page(page_meta), debug_out=debug_out,
    )

    stored_crop = debug_out["ocr_crops"][0][0]
    pre_flip_crop = raw_crop[:, :, ::-1]  # parse.py's own BGR->RGB reversal
    assert np.array_equal(stored_crop, np.rot90(pre_flip_crop, 2))
    assert not np.array_equal(stored_crop, pre_flip_crop)  # sanity: rotation actually happened

    before_crop, after_crop = debug_out["classifier_crops"][0]
    assert np.array_equal(before_crop, pre_flip_crop)
    assert np.array_equal(after_crop, stored_crop)

    entry = debug_out["rotation"][0]
    assert entry["minarea_angle_deg"] == 0.0
    assert entry["hough_angle_deg"] is None
    assert entry["combined_angle_deg"] == 0.0
    assert entry["flip_angle_deg"] == 180.0
    assert entry["retry_count"] == 0  # pass-1 (classifier + recognizer) already succeeded
    # combined (0) + pass-1's own flip (180) = 180, normalized (mod 180, into
    # [-90, 90)) to 0.0.
    assert entry["best_angle_deg"] == 0.0
    assert debug_out["retry_stats"] == {"0": 1, "1": 0, "2": 0, "3": 0, "failed": 0}


def test_parse_blank_recognition_recovers_via_retry_sweep(page_meta, vector, monkeypatch):
    """A blank pass-1 recognition should be retried at +90 -> 180 -> 270
    (raw recognizer calls, no classifier), stopping at the first pass that
    recovers non-blank text -- here the +90 (k=1) pass."""
    from rastervec.P3_Vector_Parsing.VectorClassification.paddle_engine import RotationDebug

    v = vector(kind="l", bbox=(10.0, 10.0, 20.0, 20.0), color=(0.0, 0.0, 0.0), seqno=1)
    quad = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    monkeypatch.setattr(PaddleDetectBackend, "detect", lambda self, bgr: [quad])

    raw_crop = np.zeros((4, 4, 3), dtype=np.uint8)
    rotation_debug = RotationDebug(hough_angle_deg=None, minarea_angle_deg=0.0, combined_angle_deg=0.0)
    monkeypatch.setattr(
        vectorclassification, "hough_deskew",
        lambda bgr, quad, **kwargs: (raw_crop, rotation_debug),
    )
    monkeypatch.setattr(
        PaddleRecBackend, "recognize_crops",
        lambda self, crops: [OcrBox(text="", confidence=0.0, flip_deg=0) for _ in crops],
    )
    raw_calls: list[int] = []

    def _recognize_crops_raw(self, crops):
        raw_calls.append(len(crops))
        if len(raw_calls) == 1:  # the +90 pass -- recovers text
            return [OcrBox(text="Y", confidence=1.0, flip_deg=0) for _ in crops]
        return [OcrBox(text="", confidence=0.0, flip_deg=0) for _ in crops]  # never reached

    monkeypatch.setattr(PaddleRecBackend, "recognize_crops_raw", _recognize_crops_raw)

    debug_out: dict = {}
    _drawing, texts = vectorclassification.parse(
        [v], [], _page(page_meta), debug_out=debug_out,
    )

    assert raw_calls == [1]  # only the +90 pass ran -- success stopped the sweep
    assert len(texts) == 1 and texts[0].text == "Y"
    entry = debug_out["rotation"][0]
    assert entry["retry_count"] == 1
    # combined (0) + the +90 retry rotation = 90, normalized (mod 180, into
    # [-90, 90)) to -90.0.
    assert entry["best_angle_deg"] == -90.0
    assert debug_out["retry_stats"] == {"0": 0, "1": 1, "2": 0, "3": 0, "failed": 0}
    # classifier_crops logs pass 1's own before/after plus the +90 retry's
    # before/after -- 2 entries total for this one detection.
    assert len(debug_out["classifier_crops"]) == 2


class _FakeComputePool:
    """Minimal stand-in for a `multiprocessing.Manager().Pool()` proxy --
    calls the given job function directly in this process instead of
    dispatching to a real worker, but exercises the exact same
    `compute.starmap`/`compute.apply` call shape `parse.py` uses, so this
    test fails if that dispatch wiring is ever removed or malformed
    (wrong arg shape, wrong job function, etc)."""

    def __init__(self):
        self.starmap_calls: list[tuple] = []
        self.apply_calls: list[tuple] = []

    def starmap(self, fn, args_list):
        self.starmap_calls.append((fn, list(args_list)))
        return [fn(*args) for args in args_list]

    def apply(self, fn, args):
        self.apply_calls.append((fn, args))
        return fn(*args)


def test_parse_dispatches_detect_and_recognize_through_compute(page_meta, vector, monkeypatch):
    """When `compute` is given, detect is dispatched per-cluster via
    `compute.starmap(_detect_job, ...)` and recognize via
    `compute.apply(_recognize_crops_job, ...)` -- not called in-process
    directly on the backend instances. Regression test for the Pool-2
    OCR-batching optimization (this codepath had no coverage before)."""
    v = vector(kind="l", bbox=(10.0, 10.0, 20.0, 20.0), color=(0.0, 0.0, 0.0), seqno=1)
    quad = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])

    monkeypatch.setattr(PaddleDetectBackend, "detect", lambda self, bgr: [quad])
    monkeypatch.setattr(
        PaddleRecBackend, "recognize_crops",
        lambda self, crops: [OcrBox(text="X", confidence=1.0, flip_deg=0) for _ in crops],
    )

    compute = _FakeComputePool()
    drawing, texts = vectorclassification.parse(
        [v], [], _page(page_meta), compute=compute,
    )

    assert len(texts) == 1 and texts[0].text == "X"
    assert len(compute.starmap_calls) == 1
    fn, args_list = compute.starmap_calls[0]
    assert fn is vectorclassification._detect_job
    assert len(args_list) == 1  # one cluster
    assert len(compute.apply_calls) == 1
    fn, args = compute.apply_calls[0]
    assert fn is vectorclassification._recognize_crops_job
    assert len(args[0]) == 1  # one crop batched


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


def test_bbox_aspect_ratio_is_orientation_agnostic(vector):
    wide = (0.0, 0.0, 30.0, 10.0)
    tall = (0.0, 0.0, 10.0, 30.0)
    assert vectorclassification._bbox_aspect_ratio(wide) == 3.0
    assert vectorclassification._bbox_aspect_ratio(tall) == 3.0


def test_bbox_aspect_ratio_zero_for_degenerate_bbox():
    assert vectorclassification._bbox_aspect_ratio((0.0, 0.0, 0.0, 10.0)) == 0.0
    assert vectorclassification._bbox_aspect_ratio((0.0, 0.0, 10.0, 0.0)) == 0.0


def test_quad_allows_rotation_false_below_aspect_ratio_threshold(vector):
    # A near-square bbox, even with several disjoint vectors under it.
    bbox = (0.0, 0.0, 10.0, 9.0)
    vectors = [
        vector(kind="l", bbox=(0.0, 0.0, 2.0, 2.0), color=(0.0, 0.0, 0.0), seqno=1),
        vector(kind="l", bbox=(8.0, 8.0, 10.0, 9.0), color=(0.0, 0.0, 0.0), seqno=2),
    ]
    assert vectorclassification._quad_allows_rotation(bbox, vectors) is False


def test_quad_allows_rotation_false_for_single_connected_component(vector):
    # Elongated bbox, but the two vectors overlap each other -> one component.
    bbox = (0.0, 0.0, 30.0, 5.0)
    vectors = [
        vector(kind="l", bbox=(0.0, 0.0, 5.0, 5.0), color=(0.0, 0.0, 0.0), seqno=1),
        vector(kind="l", bbox=(3.0, 0.0, 8.0, 5.0), color=(0.0, 0.0, 0.0), seqno=2),
    ]
    assert vectorclassification._quad_allows_rotation(bbox, vectors) is False


def test_quad_allows_rotation_true_for_elongated_multi_component_quad(vector):
    # Elongated bbox, two disjoint (non-overlapping) vectors under it.
    bbox = (0.0, 0.0, 30.0, 5.0)
    vectors = [
        vector(kind="l", bbox=(0.0, 0.0, 5.0, 5.0), color=(0.0, 0.0, 0.0), seqno=1),
        vector(kind="l", bbox=(20.0, 0.0, 25.0, 5.0), color=(0.0, 0.0, 0.0), seqno=2),
    ]
    assert vectorclassification._quad_allows_rotation(bbox, vectors) is True


def test_drawing_extra_vectors_empty_inputs_return_empty():
    assert vectorclassification._drawing_extra_vectors([], []) == []
    assert vectorclassification._drawing_extra_vectors([], [(0.0, 0.0, 1.0, 1.0)]) == []


def test_drawing_extra_vectors_excludes_vector_overlapping_accepted_box(vector):
    v = vector(kind="l", bbox=(0.0, 0.0, 5.0, 5.0), color=(0.0, 0.0, 0.0), seqno=1)
    accepted = [(0.0, 0.0, 5.0, 5.0)]
    assert vectorclassification._drawing_extra_vectors([v], accepted) == []


def test_drawing_extra_vectors_includes_vector_overlapping_only_rejected_box(vector):
    # No accepted boxes at all -- a rejected/blank detection never appears in
    # `accepted_boxes` to begin with, so this is indistinguishable from "no
    # accepted box covers it".
    v = vector(kind="l", bbox=(0.0, 0.0, 5.0, 5.0), color=(0.0, 0.0, 0.0), seqno=1)
    assert vectorclassification._drawing_extra_vectors([v], []) == [v]


def test_drawing_extra_vectors_includes_vector_not_overlapping_any_box(vector):
    v = vector(kind="l", bbox=(0.0, 0.0, 5.0, 5.0), color=(0.0, 0.0, 0.0), seqno=1)
    accepted = [(50.0, 50.0, 55.0, 55.0)]  # far away, no overlap
    assert vectorclassification._drawing_extra_vectors([v], accepted) == [v]


def test_drawing_extra_vectors_excludes_transitively_connected_chain(vector):
    """`a` doesn't directly touch the accepted box, but `b` does, and `a`
    overlaps `b` -- both should be excluded from drawing (transitivity),
    not just `b`."""
    a = vector(kind="l", bbox=(0.0, 0.0, 5.0, 5.0), color=(0.0, 0.0, 0.0), seqno=1)
    b = vector(kind="l", bbox=(4.0, 0.0, 9.0, 5.0), color=(0.0, 0.0, 0.0), seqno=2)
    accepted = [(8.0, 0.0, 12.0, 5.0)]  # touches b's bbox, not a's
    assert vectorclassification._drawing_extra_vectors([a, b], accepted) == []


def test_parse_ocr_rejected_cluster_ends_up_in_drawing(page_meta, vector, monkeypatch):
    """With FAST removed, every classification cluster reaches OCR -- a
    cluster whose detection comes back blank (or is never detected) must
    now surface as `drawing` output instead of silently vanishing, since
    that's FAST's former role, now decided downstream of OCR itself."""
    v = vector(kind="l", bbox=(10.0, 10.0, 20.0, 20.0), color=(0.0, 0.0, 0.0), seqno=1)
    monkeypatch.setattr(PaddleDetectBackend, "detect", lambda self, bgr: [])  # never detected

    drawing, texts = vectorclassification.parse([v], [], _page(page_meta))

    assert texts == []
    assert drawing == [v]
