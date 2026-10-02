from __future__ import annotations

import numpy as np
import pytest

from rastervec.commons.models import Page
from rastervec.P3_Vector_Parsing.CollinearVectorClass import parse as cvc
from rastervec.P3_Vector_Parsing.CollinearVectorClass.paddle_engine import (
    OcrBox, PaddleDetectBackend, PaddleRecBackend,
)


def _page(page_meta) -> Page:
    return Page(doc_path="synthetic.pdf", meta=page_meta(width=200.0, height=100.0), fitz_page=None)


def _whole_image_quad(bgr):
    h, w = bgr.shape[:2]
    return [np.array([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]])]


def _const_rec(text, conf=1.0):
    return lambda self, crops: [OcrBox(text=text, confidence=conf) for _ in crops]


def _strokes(vector):
    """Three vertical strokes, 2 pt apart -- one parallel group at 90 deg,
    three connected components."""
    return [
        vector(bbox=(x, 10.0, x, 20.0), items=[("l", (x, 10.0), (x, 20.0))], width=0.5, seqno=i)
        for i, x in enumerate((10.0, 12.0, 14.0))
    ]


def test_parse_empty_input_returns_empty_output(page_meta):
    debug_out: dict = {}
    drawing, texts = cvc.parse([], [], _page(page_meta), verbose=True, debug_out=debug_out)
    assert drawing == [] and texts == []
    assert debug_out["global_angles"] == []
    assert debug_out["cluster_rotation"] == []
    assert debug_out["rotation"] == []
    assert debug_out["retry_stats"] == {"0": 0, "1": 0, "2": 0, "3": 0, "failed": 0}


def test_parse_rotates_cluster_before_detect_and_maps_quads_back(page_meta, vector, monkeypatch):
    seen_shapes: list = []

    def _detect(self, bgr):
        seen_shapes.append(bgr.shape)
        return _whole_image_quad(bgr)

    monkeypatch.setattr(PaddleDetectBackend, "detect", _detect)
    monkeypatch.setattr(
        PaddleRecBackend, "recognize_crops_raw",
        lambda self, crops: [OcrBox(text="III", confidence=1.0, flip_deg=0) for _ in crops],
    )
    strokes = _strokes(vector)
    debug_out: dict = {}
    drawing, texts = cvc.parse(strokes, [], _page(page_meta), debug_out=debug_out)

    (cluster,) = debug_out["cluster_rotation"]
    assert cluster["angle_deg"] == -90.0  # 90 deg strokes, normalised to [-90, 90)
    assert cluster["rule"] == "cluster 1 parallel group"
    # The render (taller than wide: 4 pt x 10 pt + padding on both) reaches
    # detect rotated a quarter turn.
    h, w = seen_shapes[0][:2]
    assert w > h

    (entry,) = debug_out["rotation"]
    # Three components -> real Hough on the crop, snapped to the global angle
    # (the three vertical lines are three collinear groups at 90 deg).
    assert debug_out["global_angles"] == pytest.approx([90.0])
    assert entry["rule"] == "multi-cc hough snapped to global angle"
    assert entry["hough_angle_deg"] == pytest.approx(90.0, abs=3.0)
    assert entry["final_angle_deg"] == -90.0
    # The quad covering the whole rotated render maps back onto the cluster
    # bbox grown by the render padding.
    x0, y0, x1, y1 = entry["bbox"]
    pad = cvc._cluster_render_padding(strokes)
    assert x0 == pytest.approx(10.0 - pad, abs=0.5) and x1 == pytest.approx(14.0 + pad, abs=0.5)
    assert y0 == pytest.approx(10.0 - pad, abs=0.5) and y1 == pytest.approx(20.0 + pad, abs=0.5)
    assert len(texts) == 1 and texts[0].text == "III"
    assert drawing == []


def test_parse_hough_branch_snaps_to_global_angle(page_meta, vector, monkeypatch):
    """Two non-parallel, non-touching strokes: no parallel group, two
    components -> Hough, snapped to the nearest global potential angle."""
    a = vector(bbox=(10.0, 10.0, 20.0, 10.0), items=[("l", (10.0, 10.0), (20.0, 10.0))], width=0.5, seqno=0)
    # Same 10 pt extent as `a`, so spatial clustering puts both in one cluster.
    b = vector(bbox=(10.0, 13.0, 20.0, 23.0), items=[("l", (10.0, 13.0), (20.0, 23.0))], width=0.5, seqno=1)
    monkeypatch.setattr(PaddleDetectBackend, "detect", lambda self, bgr: _whole_image_quad(bgr))
    monkeypatch.setattr(cvc, "hough_angle_deg", lambda crop: (2.0, np.zeros((1, 1), bool)))
    monkeypatch.setattr(
        PaddleRecBackend, "recognize_crops_raw",
        lambda self, crops: [OcrBox(text="T", confidence=1.0, flip_deg=0) for _ in crops],
    )
    debug_out: dict = {}
    cvc.parse([a, b], [], _page(page_meta), debug_out=debug_out)
    assert debug_out["global_angles"] == pytest.approx([0.0, 45.0])
    (entry,) = debug_out["rotation"]
    assert entry["rule"] == "multi-cc hough snapped to global angle"
    assert entry["hough_angle_deg"] == pytest.approx(2.0)
    assert entry["final_angle_deg"] == pytest.approx(0.0)


def _scripted_rec(monkeypatch, by_pass):
    """Patch `recognize_crops_raw` to return `by_pass[n]` (text, conf) on
    its n-th call (pass 1, then the +90/+180/+270 retries); records each
    call's batch size."""
    calls: list[int] = []

    def _raw(self, crops):
        text, conf = by_pass[len(calls)]
        calls.append(len(crops))
        return [OcrBox(text=text, confidence=conf) for _ in crops]

    monkeypatch.setattr(PaddleRecBackend, "recognize_crops_raw", _raw)
    return calls


def _one_vector_run(page_meta, vector, monkeypatch):
    v = vector(bbox=(10.0, 10.0, 20.0, 20.0), width=0.5, seqno=1)
    monkeypatch.setattr(PaddleDetectBackend, "detect", lambda self, bgr: _whole_image_quad(bgr))
    debug_out: dict = {}
    _drawing, texts = cvc.parse([v], [], _page(page_meta), debug_out=debug_out)
    return texts, debug_out


def test_parse_low_confidence_retries_all_rotations_and_keeps_best(page_meta, vector, monkeypatch):
    calls = _scripted_rec(monkeypatch, [("AB", 0.5), ("CD", 0.7), ("EF", 0.95), ("GH", 0.9)])
    texts, debug_out = _one_vector_run(page_meta, vector, monkeypatch)
    assert calls == [1, 1, 1, 1]  # pass 1 + all three retries, even after a good one
    assert [t.text for t in texts] == ["EF"]
    entry = debug_out["rotation"][0]
    assert entry["retry_count"] == 2 and entry["retried"] and entry["score"] == 0.95
    assert entry["best_angle_deg"] == 0.0  # final 0 + 180, normalised
    assert debug_out["retry_stats"] == {"0": 0, "1": 0, "2": 1, "3": 0, "failed": 0}
    assert debug_out["paddle_classifier_crops"] == []


def test_parse_single_character_is_penalised(page_meta, vector, monkeypatch):
    _scripted_rec(monkeypatch, [("I", 0.95), ("AB", 0.6), ("", 0.0), ("", 0.0)])
    texts, debug_out = _one_vector_run(page_meta, vector, monkeypatch)
    assert [t.text for t in texts] == ["AB"]  # 0.6 beats 0.95 * 0.5
    assert texts[0].confidence == 0.6  # raw confidence, not the score
    assert debug_out["rotation"][0]["best_angle_deg"] == -90.0  # final 0 + 90


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


def test_parse_all_blank_is_failed(page_meta, vector, monkeypatch):
    _scripted_rec(monkeypatch, [("", 0.0)] * 4)
    texts, debug_out = _one_vector_run(page_meta, vector, monkeypatch)
    assert texts == []
    assert debug_out["rotation"][0]["retry_count"] is None
    assert debug_out["retry_stats"]["failed"] == 1
    assert len(debug_out["ocr_blank_boxes"]) == 1
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
    v = vector(bbox=(10.0, 10.0, 20.0, 20.0), width=0.5, seqno=1)
    monkeypatch.setattr(PaddleDetectBackend, "detect", lambda self, bgr: _whole_image_quad(bgr))
    monkeypatch.setattr(
        PaddleRecBackend, "recognize_crops_raw",
        lambda self, crops: [OcrBox(text="X", confidence=1.0, flip_deg=0) for _ in crops],
    )
    compute = _FakeComputePool()
    _drawing, texts = cvc.parse([v], [], _page(page_meta), compute=compute)
    assert [t.text for t in texts] == ["X"]
    assert compute.starmap_calls[0][0] is cvc._detect_job
    assert compute.apply_calls[0][0] is cvc._recognize_crops_raw_job


def test_parse_streaming_matches_batch_render_debug(page_meta, vector, monkeypatch):
    monkeypatch.setattr(PaddleDetectBackend, "detect", lambda self, bgr: _whole_image_quad(bgr))
    monkeypatch.setattr(
        PaddleRecBackend, "recognize_crops_raw",
        lambda self, crops: [OcrBox(text="X", confidence=1.0, flip_deg=0) for _ in crops],
    )
    page = _page(page_meta)
    streamed: list[tuple] = []
    cvc.parse(_strokes(vector), [], page, verbose=True, on_debug_layer=lambda *layer: streamed.append(layer))
    debug_out: dict = {}
    cvc.parse(_strokes(vector), [], page, verbose=True, debug_out=debug_out)
    batch = cvc.render_debug(debug_out, page.meta)

    assert [(s, l) for s, l, _h, _p in streamed] == [(s, l) for s, l, _h, _p in batch]
    stages = {s for s, *_ in streamed}
    assert {"rotation", "rotation_rule_cluster", "rotation_rule_quad", "ocr", "drawing"} <= stages
    assert ("rotation", "flip angle") not in {(st, lb) for st, lb, _h, _p in streamed}
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
    layers = cvc._render_classification_layers(page_meta(), cls)
    assert [(s, l) for s, l, _h, _p in layers] == [
        ("classify_01_collinear_drawing", "dropped drawing"),
        ("classify_02_seq_overlap_merge", "kept bbox"),
    ]


def test_component_count(vector):
    assert cvc._component_count([]) == 0
    a = vector(bbox=(0.0, 0.0, 5.0, 5.0))
    b = vector(bbox=(4.0, 4.0, 9.0, 9.0))
    c = vector(bbox=(20.0, 20.0, 25.0, 25.0))
    assert cvc._component_count([a, b]) == 1
    assert cvc._component_count([a, b, c]) == 2


def test_classification_layers_intersection_stage(page_meta, vector):
    from types import SimpleNamespace as NS

    crossed = vector(bbox=(0.0, 0.0, 5.0, 0.0))
    kept = vector(bbox=(10.0, 10.0, 20.0, 20.0))
    cls = NS(clustering={"b": NS(steps=[
        NS(label="Collinear drawing", categories={"kept": NS(role="kept", groups=[[kept]])}),
        NS(label="Crossings", categories={
            "kept": NS(role="kept", groups=[[[kept]]]),
            "crossed": NS(role="dropped", groups=[[crossed]]),
        }),
    ])})
    layers = cvc._render_classification_layers(page_meta(), cls)
    labels = [(s, l) for s, l, _h, _p in layers]
    assert ("intersection", "dropped to drawing (1)") in labels
    assert not any(l == "dropped crossed" for _s, l in labels)
