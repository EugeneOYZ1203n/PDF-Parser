from __future__ import annotations

import numpy as np
import pytest

from rastervec.models import Segment
from rastervec.pipelines import _steps
from rastervec.pipelines._steps import (
    _candidate_tile_bboxes,
    _normalize_segment,
    _sample_mask,
    build_drawing_output,
    detect_text_fast,
    elect_unique_segments,
    group_similar_segments,
)


def test_sample_mask_weights_by_own_footprint(vector):
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[0:10, 0:10] = 1.0
    v1 = vector(bbox=(0, 0, 10, 10))
    v2 = vector(bbox=(90, 90, 100, 100))
    score = _sample_mask(mask, [v1, v2], zoom=1.0)
    assert score == pytest.approx(0.5)


def test_sample_mask_none_is_zero(vector):
    assert _sample_mask(None, [vector(bbox=(0, 0, 10, 10))], zoom=1.0) == 0.0


def test_build_drawing_output_restores_seq_order(vector):
    a = vector(seqno=0)
    b = vector(seqno=2)
    c = vector(seqno=1)
    d = vector(seqno=3)

    out = build_drawing_output([a, b], [c, d])

    assert [v.seqno for v in out] == [0, 1, 2, 3]


def test_build_drawing_output_empty_inputs():
    assert build_drawing_output([], []) == []


# --------------------------------------------------------------------------
# group_similar_segments (Phase F) -- always Radon-precise angle now
# --------------------------------------------------------------------------
def test_group_similar_segments_groups_translated_copies(vector):
    # two translated copies of the same rectangle -> one similarity group
    v1 = vector(kind="re", bbox=(0, 0, 10, 5))
    v2 = vector(kind="re", bbox=(50, 50, 60, 55))
    seg1 = Segment(vectors=[v1], angle=0.0)
    seg2 = Segment(vectors=[v2], angle=0.0)

    groups = group_similar_segments([seg1, seg2], tolerance=0.05)

    assert groups == [[0, 1]]


def test_group_similar_segments_different_shapes_stay_separate(vector):
    v1 = vector(kind="re", bbox=(0, 0, 10, 5))
    v2 = vector(kind="l", bbox=(50, 50, 60, 55))
    seg1 = Segment(vectors=[v1], angle=0.0)
    seg2 = Segment(vectors=[v2], angle=0.0)

    groups = group_similar_segments([seg1, seg2], tolerance=0.05)

    assert sorted(groups) == [[0], [1]]


def test_group_similar_segments_uses_own_angle_to_normalize_rotation(vector):
    # A rectangle rotated 90 degrees relative to another, but each
    # segment's own `angle` exactly accounts for that rotation -- both
    # normalize to the same canonical shape.
    v1 = vector(kind="re", bbox=(0, 0, 10, 5))
    v2 = vector(kind="re", bbox=(50, 50, 55, 60))  # same shape, swapped w/h
    seg1 = Segment(vectors=[v1], angle=0.0)
    seg2 = Segment(vectors=[v2], angle=90.0)

    groups = group_similar_segments([seg1, seg2], tolerance=0.05)

    assert groups == [[0, 1]]


def test_group_similar_segments_single_segment():
    assert group_similar_segments([]) == []


# --------------------------------------------------------------------------
# detect_text_fast (Phase D) -- runs directly on plain clusters, before
# Radon/similarity; each cluster passes or fails entirely on its own score.
# --------------------------------------------------------------------------
def test_detect_text_fast_passthrough_when_disabled(vector):
    v = vector(bbox=(0, 0, 10, 10))

    res = detect_text_fast([[v]], page=None, enable_fast=False)

    assert res.passed == [[v]]
    assert res.dropped_vectors == []
    assert res.page_result.page_image is None
    assert res.page_result.detect_seconds is None


class _FakeDetector:
    """Stands in for `OCR.fast_detect.FastDetector` -- returns a fixed
    per-pixel score without needing real model weights."""

    def __init__(self, score: float):
        self.score = score

    def detect_tiled(self, image, **kwargs):
        return np.full((image.height, image.width), self.score, dtype=np.float32)


def test_detect_text_fast_keeps_cluster_that_passes(monkeypatch, vector, page_meta):
    monkeypatch.setattr(_steps, "FastDetector", lambda: _FakeDetector(1.0))

    class _FakePage:
        meta = page_meta(width=200, height=200)

    v = vector(bbox=(10, 10, 20, 20))

    res = detect_text_fast([[v]], _FakePage(), enable_fast=True)

    assert res.passed == [[v]]
    assert res.dropped_vectors == []


def test_detect_text_fast_drops_cluster_that_fails(monkeypatch, vector, page_meta):
    monkeypatch.setattr(_steps, "FastDetector", lambda: _FakeDetector(0.0))

    class _FakePage:
        meta = page_meta(width=200, height=200)

    v = vector(bbox=(10, 10, 20, 20))

    res = detect_text_fast([[v]], _FakePage(), enable_fast=True)

    assert res.passed == []
    assert res.dropped_vectors == [v]


def test_candidate_tile_bboxes_converts_and_pads(vector):
    cluster = [vector(bbox=(10.0, 20.0, 30.0, 40.0))]

    boxes = _candidate_tile_bboxes([cluster], zoom=2.0, tile_scale=5.0, margin=3.0)

    assert boxes == [(10.0 * 10.0 - 3.0, 20.0 * 10.0 - 3.0, 30.0 * 10.0 + 3.0, 40.0 * 10.0 + 3.0)]


def test_candidate_tile_bboxes_skips_empty_clusters():
    assert _candidate_tile_bboxes([[]], zoom=1.0, tile_scale=1.0, margin=0.0) == []


def test_detect_text_fast_forwards_candidate_bboxes(monkeypatch, vector, page_meta):
    captured = {}

    class _RecordingDetector:
        def detect_tiled(self, image, **kwargs):
            captured.update(kwargs)
            return np.ones((image.height, image.width), dtype=np.float32)

    monkeypatch.setattr(_steps, "FastDetector", lambda: _RecordingDetector())

    class _FakePage:
        meta = page_meta(width=200, height=200)

    v = vector(bbox=(10, 10, 20, 20))

    detect_text_fast([[v]], _FakePage(), enable_fast=True)

    assert len(captured.get("candidate_bboxes") or []) == 1


def test_detect_text_fast_filters_each_cluster_independently(monkeypatch, vector, page_meta):
    """A weak cluster is dropped on its own, without dragging down (or
    being saved by) an unrelated strong cluster -- there is no group-min
    check anymore, since grouping doesn't exist yet at this step."""
    class _SplitScoreDetector:
        def detect_tiled(self, image, **kwargs):
            mask = np.zeros((image.height, image.width), dtype=np.float32)
            mask[:, : image.width // 2] = 1.0  # left half strong, right half zero
            return mask

    monkeypatch.setattr(_steps, "FastDetector", lambda: _SplitScoreDetector())

    class _FakePage:
        meta = page_meta(width=200, height=200)

    strong = vector(bbox=(0, 0, 10, 10))      # left half -> high score
    weak = vector(bbox=(150, 150, 160, 160))  # right half -> zero score

    res = detect_text_fast([[strong], [weak]], _FakePage(), enable_fast=True)

    assert res.passed == [[strong]]
    assert res.dropped_vectors == [weak]


# --------------------------------------------------------------------------
# elect_unique_segments (Phase F, representative election + dedup metas)
# --------------------------------------------------------------------------
def test_elect_unique_segments_single_member_group(vector):
    v = vector(bbox=(0, 0, 10, 5))
    seg = Segment(vectors=[v], angle=0.0, image=np.zeros((2, 2), dtype=np.uint8))

    uniques, metas = elect_unique_segments([seg], [[0]])

    assert len(uniques) == 1
    assert uniques[0].angle == 0.0
    assert uniques[0].image is seg.image
    assert len(metas) == 1
    assert metas[0].unique_index == 0


def test_elect_unique_segments_multi_member_group_shares_one_representative(vector):
    v1 = vector(kind="re", bbox=(0, 0, 10, 5))
    v2 = vector(kind="re", bbox=(50, 50, 60, 55))
    rep_image = np.zeros((2, 2), dtype=np.uint8)
    seg1 = Segment(vectors=[v1], angle=0.0, image=rep_image)
    seg2 = Segment(vectors=[v2], angle=0.0, image=np.ones((2, 2), dtype=np.uint8))

    uniques, metas = elect_unique_segments([seg1, seg2], [[0, 1]])

    assert len(uniques) == 1  # one representative for the whole group
    assert uniques[0].image is rep_image  # group[0]'s own image, carried unchanged
    assert len(metas) == 2  # one meta per real occurrence, including the representative's own
    assert {m.unique_index for m in metas} == {0}


def test_elect_unique_segments_empty_input():
    assert elect_unique_segments([], []) == ([], [])


def test_normalize_segment_undoes_known_rotation(vector):
    from rastervec.helpers.geometry import transform_vector

    v = vector(kind="l", items=[("l", (0.0, 0.0), (200.0, 0.0))])
    rotated = transform_vector(v, offset=(0.0, 0.0), rotation_deg=40.0)

    seg = Segment(vectors=[rotated], angle=40.0)
    canonical, _offset = _normalize_segment(seg)

    x0, y0, x1, y1 = canonical[0].bbox
    assert (x1 - x0) > (y1 - y0)  # wide, not tall -- rotation undone
