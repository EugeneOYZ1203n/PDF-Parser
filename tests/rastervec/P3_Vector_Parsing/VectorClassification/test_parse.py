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


def test_parse_debug_out_has_no_old_fast_keys(page_meta):
    """The old pre-OCR FAST filtering stage's `debug_out` keys
    (`fast_result`/`fast_passed`/`fast_dropped`) must not come back -- FAST
    now only runs after recognition, under `debug_out["fast"]`."""
    debug_out: dict = {}
    vectorclassification.parse([], [], _page(page_meta), verbose=True, debug_out=debug_out)
    assert "fast_result" not in debug_out
    assert "fast_passed" not in debug_out
    assert "fast_dropped" not in debug_out
    assert debug_out["texts"] == []
    assert debug_out["paddle_classifier_crops"] == []
    assert debug_out["recog_bucket_crops"] == {"0": [], "1": [], "2": [], "3": [], "failed": []}
    assert debug_out["cluster_detections"] == []
    assert debug_out["rotation"] == []
    assert debug_out["retry_stats"] == {"0": 0, "1": 0, "2": 0, "3": 0, "failed": 0}
    assert debug_out["fast"] == {"groups": [], "kept": [], "dropped": []}
    assert debug_out["fast_images"] == []


def test_parse_keeps_ocr_crop_for_blank_recognition(page_meta, vector, monkeypatch):
    """A detected quad whose PaddleOCR recognition comes back blank must
    still be recorded in `recog_bucket_crops["failed"]` (for debug-image
    dumping) -- it just shouldn't become a real output `Text`. Regression
    test for the bug where the crop-capture sat after the `if not box.text:
    continue` guard, so blank recognitions were silently dropped from the
    debug-crop list."""
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
    # but its crop is still captured for debugging
    assert len(debug_out["recog_bucket_crops"]["failed"]) == 1
    assert debug_out["recog_bucket_crops"]["failed"][0][1] == ""
    assert len(debug_out["ocr_blank_boxes"]) == 1  # and its page-space bbox is recorded for debug rendering
    assert debug_out["rotation"][0]["retry_count"] is None  # never recovered
    assert debug_out["retry_stats"] == {"0": 0, "1": 0, "2": 0, "3": 0, "failed": 1}


def test_parse_ocr_crop_reflects_post_flip_rotation(page_meta, vector, monkeypatch):
    """`recog_bucket_crops` must store the crop PaddleOCR actually recognised
    from -- when the angle classifier decides a crop is upside-down
    (`flip_deg=180`), recognition itself runs on the rotated pixels, so the
    stashed debug crop should be rotated the same way, not the raw pre-flip
    crop. `paddle_classifier_crops` stores the pre-flip crop, exactly what
    the classifier itself saw."""
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

    stored_crop = debug_out["recog_bucket_crops"]["0"][0][0]
    pre_flip_crop = raw_crop[:, :, ::-1]  # parse.py's own BGR->RGB reversal
    assert np.array_equal(stored_crop, np.rot90(pre_flip_crop, 2))
    assert not np.array_equal(stored_crop, pre_flip_crop)  # sanity: rotation actually happened

    classifier_crop = debug_out["paddle_classifier_crops"][0]
    assert np.array_equal(classifier_crop, pre_flip_crop)

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
    # paddle_classifier_crops logs pass 1's own classifier input once per
    # detection, regardless of which later retry pass actually succeeded.
    assert len(debug_out["paddle_classifier_crops"]) == 1
    # the successful +90 retry's own crop lands in the "1 retry" bucket.
    assert len(debug_out["recog_bucket_crops"]["1"]) == 1
    assert debug_out["recog_bucket_crops"]["1"][0][1] == "Y"


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
    `compute.starmap(_detect_job, ...)`, recognize via
    `compute.apply(_recognize_crops_job, ...)` and FAST per accepted detect
    group via `compute.starmap(_fast_job, ...)` -- not called in-process
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
    assert len(compute.starmap_calls) == 2
    fn, args_list = compute.starmap_calls[0]
    assert fn is vectorclassification._detect_job
    assert len(args_list) == 1  # one cluster
    fn, args_list = compute.starmap_calls[1]
    assert fn is vectorclassification._fast_job
    assert len(args_list) == 1  # one accepted detect group
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


def test_quad_allows_rotation_false_for_single_or_no_vectors(vector):
    # A single vector under the bbox -> no baseline to measure, regardless
    # of the bbox's own shape.
    bbox = (0.0, 0.0, 30.0, 5.0)
    vectors = [vector(kind="l", bbox=(0.0, 0.0, 5.0, 5.0), color=(0.0, 0.0, 0.0), seqno=1)]
    assert vectorclassification._quad_allows_rotation(bbox, vectors) is False
    assert vectorclassification._quad_allows_rotation(bbox, []) is False


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


def test_quad_allows_rotation_true_regardless_of_aspect_ratio(vector):
    # A near-square bbox with two disjoint vectors under it -> still True;
    # the aspect-ratio gate was removed, only connectivity matters now.
    bbox = (0.0, 0.0, 10.0, 9.0)
    vectors = [
        vector(kind="l", bbox=(0.0, 0.0, 2.0, 2.0), color=(0.0, 0.0, 0.0), seqno=1),
        vector(kind="l", bbox=(8.0, 8.0, 10.0, 9.0), color=(0.0, 0.0, 0.0), seqno=2),
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


def test_parse_keeps_the_detectors_rotated_quad(page_meta, vector, monkeypatch):
    """PaddleOCR's detector returns rotated quads -- the output `Text`
    carries that quad in page space (`quad_points`, whose envelope is its
    `bbox`), and the debug layers draw it rather than the envelope."""
    v = vector(kind="l", bbox=(10.0, 10.0, 40.0, 30.0), color=(0.0, 0.0, 0.0), seqno=1)

    def _detect(self, bgr):
        h, w = bgr.shape[:2]
        return [np.array([[w * 0.5, h * 0.1], [w * 0.9, h * 0.5], [w * 0.5, h * 0.9], [w * 0.1, h * 0.5]])]

    monkeypatch.setattr(PaddleDetectBackend, "detect", _detect)
    monkeypatch.setattr(
        PaddleRecBackend, "recognize_crops",
        lambda self, crops: [OcrBox(text="AB", confidence=0.9, flip_deg=0) for _ in crops],
    )
    debug_out: dict = {}
    _drawing, texts = vectorclassification.parse([v], [], _page(page_meta), debug_out=debug_out)

    [t] = texts
    assert t.quad_points is not None and len(t.quad_points) == 4
    xs = [p[0] for p in t.quad_points]
    ys = [p[1] for p in t.quad_points]
    assert t.bbox == (min(xs), min(ys), max(xs), max(ys))
    assert len({round(x, 6) for x in xs}) == 3  # a diamond, not an axis-aligned box
    assert debug_out["ocr_detect_quads"] == [t.quad_points]
    assert debug_out["ocr_blank_quads"] == []


def _label_with_leader(vector, monkeypatch):
    """A small filled "glyph" plus a long leader line touching it, forced
    into one classification cluster, with a detect quad around the glyph
    only and a non-blank recognition. Returns `(glyph, line)`."""
    from types import SimpleNamespace as NS

    from rastervec.commons.renderer import ocr_prep, page_points_to_pixel
    from rastervec.P3_Vector_Parsing.VectorClassification.config import (
        MAX_RENDER_DPI, MIN_RENDER_SIDE_PX, OCR_DPI,
    )

    glyph = vector(kind="re", type="f", fill=(0.0, 0.0, 0.0), color=None, bbox=(20.0, 20.0, 26.0, 28.0), seqno=1)
    line = vector(kind="l", bbox=(26.0, 24.0, 150.0, 24.0), width=1.0, seqno=2)
    cluster = [glyph, line]
    monkeypatch.setattr(
        vectorclassification, "classify_vectors",
        lambda vectors, page, verbose=False: NS(text_clusters=[[cluster]], drawing_vectors=[], clustering={}),
    )
    padding = vectorclassification._cluster_render_padding(cluster)
    dpi = ocr_prep.dpi_for_cluster(cluster, OCR_DPI, padding, MIN_RENDER_SIDE_PX, MAX_RENDER_DPI)
    corners = [(19.0, 19.0), (27.0, 19.0), (27.0, 29.0), (19.0, 29.0)]
    quad = np.array(page_points_to_pixel(cluster, dpi, corners, padding))
    monkeypatch.setattr(PaddleDetectBackend, "detect", lambda self, bgr: [quad])
    monkeypatch.setattr(
        PaddleRecBackend, "recognize_crops",
        lambda self, crops: [OcrBox(text="A", confidence=1.0, flip_deg=0) for _ in crops],
    )
    return glyph, line


def test_parse_fast_sends_leader_line_touching_a_label_to_drawing(page_meta, vector, monkeypatch):
    """FAST highlights the crop around the label (stub: everything in the
    crop is hot) -- the glyph's ink is all inside it, but most of the
    leader line's ink lies outside the crop and counts as cold, so the line
    goes to drawing even though it touches the recognized box."""
    glyph, line = _label_with_leader(vector, monkeypatch)
    debug_out: dict = {}
    drawing, texts = vectorclassification.parse([glyph, line], [], _page(page_meta), debug_out=debug_out)

    assert [t.text for t in texts] == ["A"]
    assert drawing == [line]
    [group] = debug_out["fast"]["groups"]
    assert group["accepted"] and group["heat"] is not None
    assert debug_out["fast"]["kept"] == [glyph] and debug_out["fast"]["dropped"] == [line]
    assert len(debug_out["fast_images"]) == 1


def test_parse_fast_cold_heatmap_sends_everything_to_drawing(page_meta, vector, monkeypatch):
    from rastervec.P3_Vector_Parsing.VectorClassification.fast_detect import FastDetector

    glyph, line = _label_with_leader(vector, monkeypatch)
    monkeypatch.setattr(
        FastDetector, "detect", lambda self, image: np.zeros((image.height, image.width), np.float32),
    )
    drawing, texts = vectorclassification.parse([glyph, line], [], _page(page_meta))
    assert [t.text for t in texts] == ["A"]  # recognition itself is unaffected by FAST
    assert drawing == [glyph, line]


def test_parse_fast_disabled_keeps_old_connectivity_rule(page_meta, vector, monkeypatch):
    from rastervec.P3_Vector_Parsing.VectorClassification.fast_detect import FastDetector

    glyph, line = _label_with_leader(vector, monkeypatch)
    monkeypatch.setattr(vectorclassification, "FAST_FILTER_ENABLED", False)
    monkeypatch.setattr(
        FastDetector, "detect", lambda self, image: (_ for _ in ()).throw(AssertionError("FAST ran")),
    )
    debug_out: dict = {}
    drawing, texts = vectorclassification.parse([glyph, line], [], _page(page_meta), debug_out=debug_out)
    assert [t.text for t in texts] == ["A"]
    assert drawing == []  # the line touches the accepted box -> text, as before
    assert debug_out["fast"] is None


def test_parse_emits_fast_and_geometry_debug_layers(page_meta, vector, monkeypatch):
    glyph, line = _label_with_leader(vector, monkeypatch)
    streamed: list[tuple] = []
    vectorclassification.parse(
        [glyph, line], [], _page(page_meta), on_debug_layer=lambda *layer: streamed.append(layer),
    )
    names = [(stage, label) for stage, label, _hex, _pdf in streamed]
    for expected in [
        ("geometry", "collinear groups"), ("geometry", "collinear singletons"),
        ("geometry", "parallel groups"), ("geometry", "parallel singletons"),
        ("fast", "group crop bbox"), ("fast", "group member quads"), ("fast", "heatmap"),
        ("fast", "kept as text"), ("fast", "dropped to drawing"),
    ]:
        assert expected in names
    assert all(pdf[:4] == b"%PDF" for *_rest, pdf in streamed)


def test_geometry_layers_color_each_group(page_meta, vector):
    import pymupdf as fitz

    a = vector(kind="l", items=[("l", (0.0, 0.0), (10.0, 0.0))], bbox=(0.0, 0.0, 10.0, 0.0), seqno=1)
    b = vector(kind="l", items=[("l", (20.0, 0.0), (30.0, 0.0))], bbox=(20.0, 0.0, 30.0, 0.0), seqno=2)
    c = vector(kind="l", items=[("l", (0.0, 9.0), (0.0, 19.0))], bbox=(0.0, 9.0, 0.0, 19.0), seqno=3)
    layers = {
        (stage, label): pdf
        for stage, label, _hex, pdf in vectorclassification._render_geometry_layers(page_meta(), [[a, b, c]])
    }
    collinear = fitz.open("pdf", layers[("geometry", "collinear groups")])[0].get_drawings()
    singles = fitz.open("pdf", layers[("geometry", "collinear singletons")])[0].get_drawings()
    assert len(collinear) == 2  # a + b, one shared colour
    assert len({d["color"] for d in collinear}) == 1
    assert len(singles) == 1  # c
