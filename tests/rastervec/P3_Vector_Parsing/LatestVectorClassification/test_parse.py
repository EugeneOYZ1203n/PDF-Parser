from __future__ import annotations

import math

import numpy as np
import pytest

from rastervec.commons.models import Page
from rastervec.P3_Vector_Parsing.LatestVectorClassification import parse as lvc
from rastervec.P3_Vector_Parsing.LatestVectorClassification.paddle_engine import (
    OcrBox, PaddleDetectBackend, PaddleRecBackend,
)


def _page(page_meta) -> Page:
    return Page(doc_path="synthetic.pdf", meta=page_meta(width=200.0, height=100.0), fitz_page=None)


def _whole_image_quad(bgr):
    h, w = bgr.shape[:2]
    return [np.array([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]])]


def _rotated_quad(angle_deg):
    """A detect returning one quad rotated by `angle_deg` (y-down) about
    the render's centre, half the render's width long."""
    def _detect(self, bgr):
        h, w = bgr.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        hw, hh = w / 4.0, h / 8.0
        c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
        pts = [(x * c - y * s + cx, x * s + y * c + cy) for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
        return [np.array(pts)]
    return _detect


def _patch_rec(monkeypatch, text="X", conf=1.0, flip=0):
    monkeypatch.setattr(
        PaddleRecBackend, "recognize_crops",
        lambda self, crops: [OcrBox(text=text, confidence=conf, flip_deg=flip) for _ in crops],
    )
    monkeypatch.setattr(
        PaddleRecBackend, "recognize_crops_raw",
        lambda self, crops: [OcrBox(text=text, confidence=conf) for _ in crops],
    )


def _strokes(vector):
    """Three vertical strokes, 2 pt apart -- one cluster."""
    return [
        vector(bbox=(x, 10.0, x, 20.0), items=[("l", (x, 10.0), (x, 20.0))], width=0.5, seqno=i)
        for i, x in enumerate((10.0, 12.0, 14.0))
    ]


def test_parse_empty_input_returns_empty_output(page_meta):
    debug_out: dict = {}
    drawing, texts = lvc.parse([], [], _page(page_meta), verbose=True, debug_out=debug_out)
    assert drawing == [] and texts == []
    assert debug_out["rotation"] == []
    assert debug_out["retry_stats"] == {"0": 0, "1": 0, "2": 0, "3": 0, "failed": 0}
    for gone in ("global_angles", "cluster_rotation", "fast", "fast_images"):
        assert gone not in debug_out


def test_parse_detects_on_the_unrotated_render(page_meta, vector, monkeypatch):
    seen_shapes: list = []

    def _detect(self, bgr):
        seen_shapes.append(bgr.shape)
        return _whole_image_quad(bgr)

    monkeypatch.setattr(PaddleDetectBackend, "detect", _detect)
    _patch_rec(monkeypatch, "III")
    strokes = _strokes(vector)
    debug_out: dict = {}
    drawing, texts = lvc.parse(strokes, [], _page(page_meta), debug_out=debug_out)
    # The render is taller than wide (4 pt x 10 pt + padding) -- not rotated.
    h, w = seen_shapes[0][:2]
    assert h > w
    (entry,) = debug_out["rotation"]
    # Whole-image quad: long edge vertical -> -90 (normalised), no flip.
    assert entry["quad_angle_deg"] == -90.0
    assert entry["final_angle_deg"] == pytest.approx(270.0)
    x0, y0, x1, y1 = entry["bbox"]
    pad = lvc._cluster_render_padding(strokes)
    assert x0 == pytest.approx(10.0 - pad, abs=0.5) and x1 == pytest.approx(14.0 + pad, abs=0.5)
    assert y0 == pytest.approx(10.0 - pad, abs=0.5) and y1 == pytest.approx(20.0 + pad, abs=0.5)
    assert [t.text for t in texts] == ["III"]
    assert drawing == []  # every stroke lies inside the quad


def test_parse_quad_angle_drives_the_text_direction(page_meta, vector, monkeypatch):
    monkeypatch.setattr(PaddleDetectBackend, "detect", _rotated_quad(30.0))
    _patch_rec(monkeypatch, "AB")
    debug_out: dict = {}
    _drawing, texts = lvc.parse(
        [vector(bbox=(10.0, 10.0, 60.0, 40.0), width=0.5, seqno=1)], [], _page(page_meta), debug_out=debug_out,
    )
    (entry,) = debug_out["rotation"]
    assert entry["quad_angle_deg"] == pytest.approx(30.0, abs=0.5)
    (t,) = texts
    assert t.angle() == pytest.approx(30.0, abs=0.5)
    # Reading-ordered quad: p0 -> p1 runs along the text direction.
    (x0, y0), (x1, y1) = t.quad_points[0], t.quad_points[1]
    assert math.degrees(math.atan2(y1 - y0, x1 - x0)) == pytest.approx(30.0, abs=0.5)


def test_parse_classifier_flip_turns_the_text_180(page_meta, vector, monkeypatch):
    monkeypatch.setattr(PaddleDetectBackend, "detect", _rotated_quad(30.0))
    _patch_rec(monkeypatch, "AB", flip=180)
    debug_out: dict = {}
    _drawing, texts = lvc.parse(
        [vector(bbox=(10.0, 10.0, 60.0, 40.0), width=0.5, seqno=1)], [], _page(page_meta), debug_out=debug_out,
    )
    (entry,) = debug_out["rotation"]
    assert entry["cls_flip_deg"] == 180
    assert entry["final_angle_deg"] == pytest.approx(210.0, abs=0.5)
    assert texts[0].angle() == pytest.approx(-150.0, abs=0.5)
    layers = lvc._render_rotation_layers(page_meta(), debug_out["rotation"])
    assert ("rotation", "cls flipped (1)") in [(s, l) for s, l, _h, _p in layers]


def _scripted_rec(monkeypatch, by_pass):
    """Pass 1 (`recognize_crops`, with the classifier) returns `by_pass[0]`;
    the n-th retry (`recognize_crops_raw`) returns `by_pass[n]`. Records
    each call's batch size."""
    calls: list[int] = []

    def _cls(self, crops):
        text, conf = by_pass[0]
        calls.append(len(crops))
        return [OcrBox(text=text, confidence=conf) for _ in crops]

    def _raw(self, crops):
        text, conf = by_pass[len(calls)]
        calls.append(len(crops))
        return [OcrBox(text=text, confidence=conf) for _ in crops]

    monkeypatch.setattr(PaddleRecBackend, "recognize_crops", _cls)
    monkeypatch.setattr(PaddleRecBackend, "recognize_crops_raw", _raw)
    return calls


def _one_vector_run(page_meta, vector, monkeypatch):
    v = vector(bbox=(10.0, 10.0, 30.0, 15.0), width=0.5, seqno=1)  # wider than tall -> quad angle 0
    monkeypatch.setattr(PaddleDetectBackend, "detect", lambda self, bgr: _whole_image_quad(bgr))
    debug_out: dict = {}
    _drawing, texts = lvc.parse([v], [], _page(page_meta), debug_out=debug_out)
    return texts, debug_out


def test_parse_low_confidence_retries_all_rotations_and_keeps_best(page_meta, vector, monkeypatch):
    calls = _scripted_rec(monkeypatch, [("AB", 0.5), ("CD", 0.7), ("EF", 0.95), ("GH", 0.9)])
    texts, debug_out = _one_vector_run(page_meta, vector, monkeypatch)
    assert calls == [1, 1, 1, 1]  # pass 1 + all three retries, even after a good one
    assert [t.text for t in texts] == ["EF"]
    entry = debug_out["rotation"][0]
    assert entry["retry_count"] == 2 and entry["retried"] and entry["score"] == 0.95
    assert entry["final_angle_deg"] == 180.0  # quad 0 + 2 * 90
    assert entry["cls_flip_deg"] is None  # a retry won -- the classifier flip does not apply
    assert debug_out["retry_stats"] == {"0": 0, "1": 0, "2": 1, "3": 0, "failed": 0}
    assert len(debug_out["paddle_classifier_crops"]) == 1
    assert len(debug_out["quad_rotation_regions"]) == 1


def test_parse_single_character_is_penalised(page_meta, vector, monkeypatch):
    _scripted_rec(monkeypatch, [("I", 0.95), ("AB", 0.6), ("", 0.0), ("", 0.0)])
    texts, debug_out = _one_vector_run(page_meta, vector, monkeypatch)
    assert [t.text for t in texts] == ["AB"]  # 0.6 beats 0.95 * 0.5
    assert texts[0].confidence == 0.6  # raw confidence, not the score
    assert debug_out["rotation"][0]["final_angle_deg"] == 90.0  # quad 0 + 90


def test_parse_confident_pass_one_is_not_retried(page_meta, vector, monkeypatch):
    calls = _scripted_rec(monkeypatch, [("AB", 0.8)])
    texts, debug_out = _one_vector_run(page_meta, vector, monkeypatch)
    assert calls == [1] and [t.text for t in texts] == ["AB"]
    assert debug_out["rotation"][0]["retried"] is False


def test_parse_retry_keeps_pass_one_when_nothing_beats_it(page_meta, vector, monkeypatch):
    _scripted_rec(monkeypatch, [("AB", 0.5), ("", 0.0), ("X", 0.9), ("CD", 0.5)])
    texts, debug_out = _one_vector_run(page_meta, vector, monkeypatch)
    assert [t.text for t in texts] == ["AB"]  # X scores 0.45; CD ties and loses to pass 1
    entry = debug_out["rotation"][0]
    assert entry["retry_count"] == 0 and entry["retried"]


def test_parse_english_words_boost_the_retry_winner(page_meta, vector, monkeypatch):
    _scripted_rec(monkeypatch, [("XQZV", 0.7), ("DOOR FRAME", 0.6), ("", 0.0), ("", 0.0)])
    texts, debug_out = _one_vector_run(page_meta, vector, monkeypatch)
    assert [t.text for t in texts] == ["DOOR FRAME"]  # 0.6 * 1.2**2 = 0.864 beats 0.7
    assert texts[0].confidence == 0.6
    entry = debug_out["rotation"][0]
    assert entry["retry_count"] == 1
    assert entry["score"] == pytest.approx(0.864)
    assert entry["raw_score"] == 0.6 and entry["english_words"] == 2


def test_parse_english_words_do_not_skip_the_retry(page_meta, vector, monkeypatch):
    # Boosted 0.75 * 1.2 = 0.9 would clear the threshold -- the trigger uses the raw score.
    calls = _scripted_rec(monkeypatch, [("DOOR", 0.75), ("XQZV", 0.85), ("", 0.0), ("", 0.0)])
    texts, debug_out = _one_vector_run(page_meta, vector, monkeypatch)
    assert calls == [1, 1, 1, 1]
    assert [t.text for t in texts] == ["DOOR"]  # 0.9 beats 0.85
    assert debug_out["rotation"][0]["retry_count"] == 0


def test_parse_all_blank_is_failed_and_its_vectors_are_drawing(page_meta, vector, monkeypatch):
    _scripted_rec(monkeypatch, [("", 0.0)] * 4)
    v = vector(bbox=(10.0, 10.0, 30.0, 15.0), width=0.5, seqno=1)
    monkeypatch.setattr(PaddleDetectBackend, "detect", lambda self, bgr: _whole_image_quad(bgr))
    debug_out: dict = {}
    drawing, texts = lvc.parse([v], [], _page(page_meta), debug_out=debug_out)
    assert texts == [] and drawing == [v]  # a blank quad owns nothing
    assert debug_out["rotation"][0]["retry_count"] is None
    assert debug_out["retry_stats"]["failed"] == 1
    assert len(debug_out["ocr_blank_quads"]) == 1 and len(debug_out["ocr_blank_quads"][0]) == 4
    assert len(debug_out["recog_bucket_crops"]["failed"]) == 1


class _FakeComputePool:
    def __init__(self):
        self.starmap_calls: list = []
        self.apply_calls: list = []

    def starmap(self, fn, args_list):
        self.starmap_calls.append((fn, list(args_list)))
        return [fn(*args) for args in args_list]

    def apply(self, fn, args):
        self.apply_calls.append((fn, args))
        return fn(*args)


def test_parse_dispatches_detect_and_recognize_through_compute(page_meta, vector, monkeypatch):
    monkeypatch.setattr(PaddleDetectBackend, "detect", lambda self, bgr: _whole_image_quad(bgr))
    _patch_rec(monkeypatch, "X")
    compute = _FakeComputePool()
    _drawing, texts = lvc.parse(
        [vector(bbox=(10.0, 10.0, 20.0, 20.0), width=0.5, seqno=1)], [], _page(page_meta), compute=compute,
    )
    assert [t.text for t in texts] == ["X"]
    assert compute.starmap_calls[0][0] is lvc._detect_job
    assert compute.apply_calls[0][0] is lvc._recognize_crops_job


def test_parse_streaming_matches_batch_render_debug(page_meta, vector, monkeypatch):
    monkeypatch.setattr(PaddleDetectBackend, "detect", lambda self, bgr: _whole_image_quad(bgr))
    _patch_rec(monkeypatch, "X")
    page = _page(page_meta)
    streamed: list[tuple] = []
    lvc.parse(_strokes(vector), [], page, verbose=True, on_debug_layer=lambda *layer: streamed.append(layer))
    debug_out: dict = {}
    lvc.parse(_strokes(vector), [], page, verbose=True, debug_out=debug_out)
    batch = lvc.render_debug(debug_out, page.meta)

    assert [(s, l) for s, l, _h, _p in streamed] == [(s, l) for s, l, _h, _p in batch]
    names = {(s, l) for s, l, _h, _p in streamed}
    for expected in [
        ("geometry", "collinear groups"), ("geometry", "parallel singletons"),
        ("rotation", "quad angle"), ("rotation", "final angle"),
        ("ownership", "text vectors"), ("ownership", "drawing vectors"),
        ("ocr", "detect bbox"), ("drawing", "drawing vectors"),
    ]:
        assert expected in names
    stages = {s for s, *_ in streamed}
    assert not stages & {"fast", "rotation_rule_cluster", "rotation_rule_quad"}
    for _stage, _label, hexcolor, pdf_bytes in streamed:
        assert pdf_bytes[:4] == b"%PDF" and hexcolor.startswith("#")


def test_classification_layers_render_dropped_vectors(page_meta, vector):
    from types import SimpleNamespace as NS

    dropped = vector(bbox=(0.0, 0.0, 5.0, 0.0))
    kept = vector(bbox=(10.0, 10.0, 20.0, 20.0))

    def step(label, kept_groups, dropped_groups):
        return NS(label=label, categories={
            "kept": NS(role="kept", groups=kept_groups),
            "drawing": NS(role="dropped", groups=dropped_groups),
        })

    cls = NS(clustering={"b": NS(steps=[
        step("Collinear drawing", [[kept]], [[dropped]]),
        step("Seq overlap merge", [[kept]], []),
    ])})
    layers = lvc._render_classification_layers(page_meta(), cls)
    assert [(s, l) for s, l, _h, _p in layers] == [
        ("classify_01_collinear_drawing", "dropped drawing"),
        ("classify_02_seq_overlap_merge", "kept bbox"),
    ]


def test_classification_layers_intersection_stage(page_meta, vector):
    from types import SimpleNamespace as NS

    crossed = vector(bbox=(0.0, 0.0, 5.0, 0.0))
    off_grid = vector(bbox=(30.0, 0.0, 35.0, 3.0))
    kept = vector(bbox=(10.0, 10.0, 20.0, 20.0))
    cls = NS(clustering={"b": NS(steps=[
        NS(label="Collinear drawing", categories={"kept": NS(role="kept", groups=[[kept]])}),
        NS(label="Crossings", categories={
            "kept": NS(role="kept", groups=[[[kept, off_grid]]]),
            "crossed": NS(role="dropped", groups=[[crossed]]),
            "crossed_off_grid": NS(role="info", groups=[[off_grid]]),
        }),
    ])})
    layers = lvc._render_classification_layers(page_meta(), cls)
    labels = [(s, l) for s, l, _h, _p in layers]
    assert ("intersection", "dropped to drawing (1)") in labels
    assert ("intersection", "flagged, kept (off-grid) (1)") in labels
    assert not any(l == "dropped crossed" for _s, l in labels)


def test_geometry_layers_color_each_group(page_meta, vector):
    import pymupdf as fitz

    a = vector(kind="l", items=[("l", (0.0, 0.0), (10.0, 0.0))], bbox=(0.0, 0.0, 10.0, 0.0), seqno=1)
    b = vector(kind="l", items=[("l", (20.0, 0.0), (30.0, 0.0))], bbox=(20.0, 0.0, 30.0, 0.0), seqno=2)
    c = vector(kind="l", items=[("l", (0.0, 9.0), (0.0, 19.0))], bbox=(0.0, 9.0, 0.0, 19.0), seqno=3)
    layers = {
        (stage, label): pdf
        for stage, label, _hex, pdf in lvc._render_geometry_layers(page_meta(), [[a, b, c]])
    }
    collinear = fitz.open("pdf", layers[("geometry", "collinear groups")])[0].get_drawings()
    singles = fitz.open("pdf", layers[("geometry", "collinear singletons")])[0].get_drawings()
    assert len(collinear) == 2  # a + b, one shared colour
    assert len({d["color"] for d in collinear}) == 1
    assert len(singles) == 1  # c


def _line(vector, x0, y0, x1, y1):
    return vector(bbox=(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), items=[("l", (x0, y0), (x1, y1))])


def test_text_vectors_by_quad_needs_most_of_the_ink_inside(vector):
    quad = ((0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0))
    glyph = _line(vector, 2, 2, 2, 8)  # fully inside
    leader = _line(vector, 14, 5, 34, 5)  # 30% inside
    text, drawing = lvc._text_vectors_by_quad([[glyph, leader]], {0: [quad]})
    assert text == [glyph] and drawing == [leader]


def test_text_vectors_by_quad_uses_the_rotated_quad(vector):
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    quad = tuple((x * c - y * s + 50, x * s + y * c + 50) for x, y in ((-20, -2), (20, -2), (20, 2), (-20, 2)))
    line = _line(vector, 35, 50, 65, 50)  # inside the envelope, mostly outside the quad
    text, drawing = lvc._text_vectors_by_quad([[line]], {0: [quad]})
    assert text == [] and drawing == [line]


def test_text_vectors_by_quad_only_uses_its_own_clusters_quads(vector):
    quad = ((0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0))
    in_a = _line(vector, 2, 2, 2, 8)
    in_b = _line(vector, 4, 2, 4, 8)  # inside cluster 0's quad, but in cluster 1
    text, drawing = lvc._text_vectors_by_quad([[in_a], [in_b]], {0: [quad]})
    assert text == [in_a] and drawing == [in_b]


def test_parse_text_carries_its_page_space_detect_quad(page_meta, vector, monkeypatch):
    _scripted_rec(monkeypatch, [("AB", 0.95)])
    texts, debug_out = _one_vector_run(page_meta, vector, monkeypatch)
    [t] = texts
    xs = [p[0] for p in t.quad_points]
    ys = [p[1] for p in t.quad_points]
    assert t.bbox == pytest.approx((min(xs), min(ys), max(xs), max(ys)))
    assert set(debug_out["ocr_detect_quads"][0]) == set(t.quad_points)


_SQUARE = ((0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0))


def test_quad_owns_bbox_fully_inside(vector):
    glyph = vector(bbox=(2, 2, 8, 8), items=[("re", (2, 2, 8, 8))])
    assert lvc._quad_owns(glyph, _SQUARE, (0, 0, 20, 10), 200.0)


def test_quad_owns_rejects_bbox_larger_than_the_quad(vector):
    # Centred on the quad, but its bbox (30x30) is bigger than the quad (20x10).
    frame = vector(bbox=(-5, -10, 25, 20), items=[("re", (-5, -10, 25, 20))])
    assert not lvc._quad_owns(frame, _SQUARE, (0, 0, 20, 10), 200.0)


def test_quad_owns_rejects_when_few_pieces_touch(vector):
    # 1 of 3 pieces touches the quad (< 50%), small bbox.
    v = vector(bbox=(15, 5, 26, 9), items=[
        ("l", (15, 5), (19, 5)), ("l", (22, 5), (26, 5)), ("l", (22, 9), (26, 9)),
    ])
    assert not lvc._quad_owns(v, _SQUARE, (0, 0, 20, 10), 200.0)


def test_quad_owns_falls_back_to_ink_fraction(vector):
    # Both pieces touch (100%); ink 8 of 10 inside -> owned.
    mostly = vector(bbox=(12, 2, 22, 8), items=[("l", (12, 2), (22, 2)), ("l", (12, 8), (22, 8))])
    assert lvc._quad_owns(mostly, _SQUARE, (0, 0, 20, 10), 200.0)
    # Both pieces touch; ink 3 of 10 inside -> not owned.
    barely = vector(bbox=(17, 2, 27, 8), items=[("l", (17, 2), (27, 2)), ("l", (17, 8), (27, 8))])
    assert not lvc._quad_owns(barely, _SQUARE, (0, 0, 20, 10), 200.0)


_TIMED_STEPS = {
    "classify_separate", "classify_collinear", "classify_seqno", "classify_spatial",
    "classify_outliers", "classify_crossings", "classify_collect",
    "ocr_render", "ocr_detect", "ocr_crop", "ocr_recognize", "ocr_assemble",
    "quad_ownership", "drawing",
}


def test_parse_times_every_step_without_nesting(page_meta, vector, monkeypatch):
    from rastervec.Evaluation.Evaluate.timing import flatten_page_timing

    monkeypatch.setattr(PaddleDetectBackend, "detect", lambda self, bgr: _whole_image_quad(bgr))
    _patch_rec(monkeypatch, "III")
    page = _page(page_meta)

    steps: dict = {}
    lvc.parse(_strokes(vector), [], page, step_durations=steps)
    assert set(steps) == _TIMED_STEPS  # no outer `classify`, no `debug_render`
    assert all(secs >= 0.0 for secs in steps.values())

    steps = {}
    lvc.parse(_strokes(vector), [], page, step_durations=steps, on_debug_layer=lambda *layer: None)
    assert set(steps) == _TIMED_STEPS | {"debug_render"}

    # The sub-steps partition phase3: none is nested inside another, so a
    # phase3 just above their sum leaves only that margin in `other`.
    row = flatten_page_timing({"phase3": sum(steps.values()) + 0.01}, steps)
    assert row["phase3.other"] == pytest.approx(0.01)
