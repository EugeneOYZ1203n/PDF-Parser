from __future__ import annotations

import numpy as np
import pytest

from rastervec.models import Segment
from rastervec.pipelines import _steps
from rastervec.pipelines._steps import (
    _candidate_tile_bboxes,
    _sample_mask,
    build_drawing_output,
    detect_text_fast,
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
# group_similar_segments (Phase E)
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
# detect_text_fast (Phase F)
# --------------------------------------------------------------------------
def test_detect_text_fast_passthrough_when_disabled(vector):
    v = vector(bbox=(0, 0, 10, 10))
    seg = Segment(vectors=[v], angle=0.0)

    res = detect_text_fast([seg], [[0]], page=None, enable_fast=False)

    assert len(res.uniques) == 1
    assert [m.unique_index for m in res.metas] == [0]
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


def test_detect_text_fast_keeps_group_when_every_member_passes(monkeypatch, vector, page_meta):
    monkeypatch.setattr(_steps, "FastDetector", lambda: _FakeDetector(1.0))

    class _FakePage:
        meta = page_meta(width=200, height=200)

    v = vector(bbox=(10, 10, 20, 20))
    seg = Segment(vectors=[v], angle=0.0)

    res = detect_text_fast([seg], [[0]], _FakePage(), enable_fast=True)

    assert len(res.uniques) == 1
    assert res.dropped_vectors == []


def test_detect_text_fast_drops_group_when_any_member_fails(monkeypatch, vector, page_meta):
    monkeypatch.setattr(_steps, "FastDetector", lambda: _FakeDetector(0.0))

    class _FakePage:
        meta = page_meta(width=200, height=200)

    v = vector(bbox=(10, 10, 20, 20))
    seg = Segment(vectors=[v], angle=0.0)

    res = detect_text_fast([seg], [[0]], _FakePage(), enable_fast=True)

    assert res.uniques == []
    assert len(res.dropped_vectors) == 1
    assert res.dropped_vectors[0] is v


def test_candidate_tile_bboxes_converts_and_pads(vector):
    seg = Segment(vectors=[vector(bbox=(10.0, 20.0, 30.0, 40.0))], angle=0.0)

    boxes = _candidate_tile_bboxes([seg], zoom=2.0, tile_scale=5.0, margin=3.0)

    assert boxes == [(10.0 * 10.0 - 3.0, 20.0 * 10.0 - 3.0, 30.0 * 10.0 + 3.0, 40.0 * 10.0 + 3.0)]


def test_candidate_tile_bboxes_skips_empty_segments():
    assert _candidate_tile_bboxes(
        [Segment(vectors=[], angle=0.0)], zoom=1.0, tile_scale=1.0, margin=0.0,
    ) == []


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
    seg = Segment(vectors=[v], angle=0.0)

    detect_text_fast([seg], [[0]], _FakePage(), enable_fast=True)

    assert len(captured.get("candidate_bboxes") or []) == 1


def test_detect_text_fast_all_must_pass_not_just_the_group_min(monkeypatch, vector, page_meta):
    """A group's weakest member failing drops the WHOLE group, even if
    another member in it would pass on its own."""
    class _SplitScoreDetector:
        def detect_tiled(self, image, **kwargs):
            mask = np.zeros((image.height, image.width), dtype=np.float32)
            mask[:, : image.width // 2] = 1.0  # left half strong, right half zero
            return mask

    monkeypatch.setattr(_steps, "FastDetector", lambda: _SplitScoreDetector())

    class _FakePage:
        meta = page_meta(width=200, height=200)

    strong = vector(bbox=(0, 0, 10, 10))    # left half -> high score
    weak = vector(bbox=(150, 150, 160, 160))  # right half -> zero score
    segments = [Segment(vectors=[strong], angle=0.0), Segment(vectors=[weak], angle=0.0)]

    res = detect_text_fast(segments, [[0, 1]], _FakePage(), enable_fast=True)

    assert res.uniques == []
    assert {id(v) for v in res.dropped_vectors} == {id(strong), id(weak)}
