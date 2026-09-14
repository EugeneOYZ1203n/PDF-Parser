from __future__ import annotations

import numpy as np
import pymupdf as fitz
import pytest

from rastervec.OCR.Paddle_OCR.ocr_backend import ClusterDetection, PaddleDetection
from rastervec.pipelines import _steps, current as current_mod
from rastervec.pipelines.current import (
    STEP_NAMES,
    cluster_buckets,
    run_pipeline,
    separate_by_layer_color_width,
)
from rastervec.pipelines._steps import filter_vectors_fast, reassign_by_overlap


# --------------------------------------------------------------------------
# filter_vectors_fast
# --------------------------------------------------------------------------
def test_filter_vectors_fast_passthrough_when_disabled(vector):
    v = vector(bbox=(0, 0, 10, 10))

    res = filter_vectors_fast([v], page=None, enable_fast=False)

    assert res.passed == [v]
    assert res.dropped == []


class _FakeDetector:
    def __init__(self, score: float):
        self.score = score

    def detect_tiled(self, image, **kwargs):
        return np.full((image.height, image.width), self.score, dtype=np.float32)


def test_filter_vectors_fast_keeps_and_drops_by_own_item_score(monkeypatch, vector, page_meta):
    monkeypatch.setattr(_steps, "FastDetector", lambda: _FakeDetector(1.0))

    class _FakePage:
        meta = page_meta(width=200, height=200)

    v = vector(bbox=(10, 10, 20, 20))

    res = filter_vectors_fast([v], _FakePage(), enable_fast=True)

    assert res.passed == [v]
    assert res.dropped == []


def test_filter_vectors_fast_drops_when_score_low(monkeypatch, vector, page_meta):
    monkeypatch.setattr(_steps, "FastDetector", lambda: _FakeDetector(0.0))

    class _FakePage:
        meta = page_meta(width=200, height=200)

    v = vector(bbox=(10, 10, 20, 20))

    res = filter_vectors_fast([v], _FakePage(), enable_fast=True)

    assert res.passed == []
    assert res.dropped == [v]


# --------------------------------------------------------------------------
# separation + seqno clustering
# --------------------------------------------------------------------------
def test_separate_by_layer_color_width_splits_on_all_three_axes(vector):
    a = vector(layer="l1", color=(0, 0, 0), width=1.0, seqno=0)
    b = vector(layer="l1", color=(0, 0, 0), width=2.0, seqno=1)  # different width
    c = vector(layer="l1", color=(1, 1, 1), width=1.0, seqno=2)  # different color
    d = vector(layer="l2", color=(0, 0, 0), width=1.0, seqno=3)  # different layer

    buckets = separate_by_layer_color_width([a, b, c, d])

    assert len(buckets) == 4
    assert sorted(len(bucket) for bucket in buckets) == [1, 1, 1, 1]


def test_cluster_buckets_merges_consecutive_seqno_within_a_bucket(vector):
    a = vector(bbox=(0, 0, 5, 5), seqno=0)
    b = vector(bbox=(5, 0, 10, 5), seqno=1)  # touching -> merges with a
    c = vector(bbox=(100, 100, 105, 105), seqno=2)  # far away -> new cluster

    clusters = cluster_buckets([[a, b, c]], base_tolerance=1.0, scale=0.0)

    assert len(clusters) == 2
    assert {v.seqno for v in clusters[0]} == {0, 1}
    assert {v.seqno for v in clusters[1]} == {2}


def test_cluster_buckets_never_merges_across_buckets(vector):
    a = vector(bbox=(0, 0, 5, 5), seqno=0)
    b = vector(bbox=(5, 0, 10, 5), seqno=1)  # would merge with a if same bucket

    clusters = cluster_buckets([[a], [b]], base_tolerance=1.0, scale=0.0)

    assert len(clusters) == 2


def test_cluster_buckets_is_order_independent(vector):
    """Three touching vectors, given in seqno-scrambled order, still all
    end up in one cluster -- the union-find merge never sorts by seqno."""
    a = vector(bbox=(0, 0, 5, 5), seqno=2)
    b = vector(bbox=(5, 0, 10, 5), seqno=0)
    c = vector(bbox=(10, 0, 15, 5), seqno=1)

    clusters = cluster_buckets([[a, b, c]], base_tolerance=1.0, scale=0.0)

    assert len(clusters) == 1
    assert {v.seqno for v in clusters[0]} == {0, 1, 2}


def test_cluster_buckets_merges_a_bridging_vector_into_one_set(vector):
    """Two separate far-apart clusters (a,b) and (c,d), then a vector `e`
    that bridges the gap between both -- all five must end up as one
    cluster (a vector matching more than one existing set unions them all
    together), regardless of processing order."""
    a = vector(bbox=(0, 0, 5, 5), seqno=0)
    b = vector(bbox=(0, 5, 5, 10), seqno=1)
    c = vector(bbox=(100, 0, 105, 5), seqno=2)
    d = vector(bbox=(100, 5, 105, 10), seqno=3)
    e = vector(bbox=(50, 0, 55, 5), seqno=4)

    clusters = cluster_buckets([[a, b, c, d, e]], base_tolerance=50.0, scale=0.0)

    assert len(clusters) == 1
    assert {v.seqno for v in clusters[0]} == {0, 1, 2, 3, 4}


def test_cluster_buckets_tolerance_cap_holds_cluster_size_down(vector):
    """A run of vectors with a steadily widening gap would chain-merge into
    one huge cluster if the dynamic tolerance kept growing unbounded with the
    growing set's own bbox (each merge widens the gap it can bridge next) --
    the cap should stop that once the gap outgrows it."""
    vectors = []
    x = 0.0
    for i in range(60):
        vectors.append(vector(bbox=(x, 0.0, x + 1.0, 1.0), seqno=i))
        x += 1.0 + i * 0.3  # widening gap to the next vector

    uncapped = cluster_buckets([vectors], base_tolerance=1.0, scale=1.0, cap=10_000.0)
    capped = cluster_buckets([vectors], base_tolerance=1.0, scale=1.0, cap=4.0)

    assert len(uncapped) == 1  # unbounded tolerance -> everything merges
    assert len(capped) > 1  # capped tolerance -> splits into multiple clusters


# --------------------------------------------------------------------------
# reassign_by_overlap
# --------------------------------------------------------------------------
class _FakeDetection:
    def __init__(self, bbox):
        self.bbox = bbox


class _FakeClusterDetection:
    def __init__(self, detections):
        self.detections = detections


def test_reassign_by_overlap_assigns_vector_fully_covered_by_a_detection(vector):
    v = vector(bbox=(0, 0, 10, 10))
    detection = _FakeDetection(bbox=(0, 0, 10, 10))
    cd = _FakeClusterDetection([detection])

    result = reassign_by_overlap([[v]], [cd])

    assert result.text == [[[v]]]
    assert result.drawing == []


def test_reassign_by_overlap_sends_uncovered_vector_to_drawing(vector):
    v = vector(bbox=(0, 0, 10, 10))
    detection = _FakeDetection(bbox=(100, 100, 110, 110))  # no overlap at all
    cd = _FakeClusterDetection([detection])

    result = reassign_by_overlap([[v]], [cd])

    assert result.text == [[[]]]
    assert result.drawing == [v]


def test_reassign_by_overlap_no_detections_sends_everything_to_drawing(vector):
    v = vector(bbox=(0, 0, 10, 10))

    result = reassign_by_overlap([[v]], [None])

    assert result.text == [[]]
    assert result.drawing == [v]


def test_reassign_by_overlap_partial_coverage_below_threshold_goes_to_drawing(vector):
    # Only 10% of v's own area overlaps the detection -- below the default
    # PADDLE_REASSIGN_MIN_COVERAGE (0.5).
    v = vector(bbox=(0, 0, 10, 10))
    detection = _FakeDetection(bbox=(9, 0, 20, 10))
    cd = _FakeClusterDetection([detection])

    result = reassign_by_overlap([[v]], [cd], min_coverage=0.5)

    assert result.text == [[[]]]
    assert result.drawing == [v]


# --------------------------------------------------------------------------
# run_pipeline: full step sequence on a synthetic PDF, with FAST disabled and
# PaddleOCR fully stubbed -- exercises the wiring, not the real models.
# --------------------------------------------------------------------------
class _StubPaddleDetectBackend:
    """Detects exactly one text box covering each cluster's own bbox, no
    rotation -- enough to exercise assignment/rotation/recognition without a
    real PaddleOCR engine."""

    def detect_on_cluster(self, vectors, *, dpi=300, padding=0.0):
        if not vectors:
            return None
        from rastervec.commons.helpers.geometry import union_bbox
        from rastervec.commons.renderer import render_vector_cluster

        bbox = union_bbox([v.bbox for v in vectors])
        image = np.asarray(render_vector_cluster(vectors, dpi, padding))
        quad = [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]
        detection = PaddleDetection(bbox=bbox, quad=quad, rotation_deg=0.0)
        return ClusterDetection(image=image, dpi=dpi, detections=[detection])


def _fake_recognize_segments(segments, *, recognize_fn=None):
    from rastervec.commons.helpers.geometry import compute_origin, union_bbox
    from rastervec.commons.models import Text

    texts = []
    for seg in segments:
        bbox = union_bbox([v.bbox for v in seg.vectors]) if seg.vectors else (0.0, 0.0, 1.0, 1.0)
        direction = (1.0, 0.0)
        texts.append(Text(
            text="TXT", bbox=bbox, direction=direction, origin=compute_origin(bbox, direction),
            font="", font_size=0.0, color=None, flags=0,
            ascender=None, descender=None, wmode=0,
            block_no=0, line_no=0, word_no=0, page_index=0, seqno=0,
            confidence=0.9, source="ocr",
        ))
    return texts


@pytest.fixture(autouse=True)
def _stub_paddle(monkeypatch):
    monkeypatch.setattr(_steps, "PaddleDetectBackend", _StubPaddleDetectBackend)
    monkeypatch.setattr(current_mod, "recognize_segments", _fake_recognize_segments)


def _drawing_pdf(tmp_pdf_path):
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    line = page.new_shape()
    line.draw_line((10, 10), (16, 16))
    line.finish(color=(0, 0, 0), width=2)
    line.commit()
    return tmp_pdf_path(doc)


def test_run_pipeline_ocrs_a_drawing_candidate(tmp_pdf_path):
    path = _drawing_pdf(tmp_pdf_path)

    res = run_pipeline(path, 0, enable_fast=False)

    assert res.engine == "current"
    assert list(res.step_durations) == STEP_NAMES
    ocr_texts = [t for t in res.texts if t.source == "ocr"]
    assert len(ocr_texts) == 1
    assert res.vectors == []  # every vector got assigned to the stub's detection


def test_run_pipeline_verbose_toggles_intermediates(tmp_pdf_path):
    path = _drawing_pdf(tmp_pdf_path)

    quiet = run_pipeline(path, 0, enable_fast=False, verbose=False)
    verbose = run_pipeline(path, 0, enable_fast=False, verbose=True)

    assert quiet.spatial_clusters is None
    assert verbose.spatial_clusters is not None
    assert verbose.cluster_detections is not None
    assert verbose.rotated_segments is not None


def test_run_pipeline_stop_after_limits_steps(tmp_pdf_path):
    path = _drawing_pdf(tmp_pdf_path)

    res = run_pipeline(path, 0, enable_fast=False, verbose=True, stop_after="clusters")

    assert set(res.step_durations) == {
        "read", "native", "vectors", "similarity", "fast", "reclassify",
        "separation", "clusters",
    }
    assert res.spatial_clusters is not None
    assert res.cluster_detections is None
