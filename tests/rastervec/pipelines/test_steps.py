from __future__ import annotations

import numpy as np
import pytest

from rastervec.pipelines._steps import (
    _candidate_tile_bboxes,
    _sample_mask_items,
    build_drawing_output,
)


def test_sample_mask_items_weights_by_own_footprint(vector):
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[0:10, 0:10] = 1.0
    v1 = vector(bbox=(0, 0, 10, 10))
    v2 = vector(bbox=(90, 90, 100, 100))
    score = _sample_mask_items(mask, v1.items + v2.items, zoom=1.0)
    assert score == pytest.approx(0.5)


def test_sample_mask_items_none_is_zero(vector):
    v = vector(bbox=(0, 0, 10, 10))
    assert _sample_mask_items(None, v.items, zoom=1.0) == 0.0


def test_build_drawing_output_restores_seq_order(vector):
    a = vector(seqno=0)
    b = vector(seqno=2)
    c = vector(seqno=1)
    d = vector(seqno=3)

    out = build_drawing_output([a, b], [c, d])

    assert [v.seqno for v in out] == [0, 1, 2, 3]


def test_build_drawing_output_empty_inputs():
    assert build_drawing_output([], []) == []


def test_candidate_tile_bboxes_converts_and_pads(vector):
    cluster = [vector(bbox=(10.0, 20.0, 30.0, 40.0))]

    boxes = _candidate_tile_bboxes([cluster], zoom=10.0, margin=3.0)

    assert boxes == [(10.0 * 10.0 - 3.0, 20.0 * 10.0 - 3.0, 30.0 * 10.0 + 3.0, 40.0 * 10.0 + 3.0)]


def test_candidate_tile_bboxes_skips_empty_clusters():
    assert _candidate_tile_bboxes([[]], zoom=1.0, margin=0.0) == []
