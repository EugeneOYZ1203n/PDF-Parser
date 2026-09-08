from __future__ import annotations

import numpy as np
import pytest

from rastervec.models import VectorPath
from rastervec.pipelines._steps import (
    _sample_mask,
    build_drawing_output,
    detect_text_fast,
    spatial_regroup,
)


def _path(seq, bbox, *, stroke_color=(0, 0, 0), layer=None) -> VectorPath:
    return VectorPath(
        seq=seq, item_index=0, kind="l", fill_rule="s",
        points=[(bbox[0], bbox[1]), (bbox[2], bbox[3])], bbox=bbox,
        stroke_color=stroke_color, fill_color=None, stroke_opacity=None,
        fill_opacity=None, stroke_width=1.0, dashes=None, closed=False,
        layer=layer, page_index=0,
    )


def test_sample_mask_weights_by_own_footprint():
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[0:10, 0:10] = 1.0
    score = _sample_mask(mask, [_path(0, (0, 0, 10, 10)), _path(1, (90, 90, 100, 100))], zoom=1.0)
    assert score == pytest.approx(0.5)


def test_sample_mask_none_is_zero():
    assert _sample_mask(None, [_path(0, (0, 0, 10, 10))], zoom=1.0) == 0.0


def test_spatial_regroup_merges_touching_same_paint():
    a = [_path(0, (0, 0, 10, 10))]
    b = [_path(1, (10.5, 0, 20, 10))]
    res = spatial_regroup([a, b], {id(a): 5, id(b): 5})
    assert len(res.clusters) == 1
    assert {p.seq for p in res.clusters[0]} == {0, 1}
    assert res.similarity_id[id(res.clusters[0])] == 5


def test_spatial_regroup_drops_id_on_disagree():
    a = [_path(0, (0, 0, 10, 10))]
    b = [_path(1, (10.5, 0, 20, 10))]
    res = spatial_regroup([a, b], {id(a): 5, id(b): 6})
    assert len(res.clusters) == 1
    assert id(res.clusters[0]) not in res.similarity_id


def test_spatial_regroup_merges_touching_different_color():
    a = [_path(0, (0, 0, 10, 10), stroke_color=(0, 0, 0))]
    b = [_path(1, (2, 0, 12, 10), stroke_color=(1, 0, 0))]
    assert len(spatial_regroup([a, b], {}).clusters) == 1


def test_spatial_regroup_keeps_far_apart_separate():
    a = [_path(0, (0, 0, 10, 10))]
    b = [_path(1, (100, 0, 110, 10))]
    assert len(spatial_regroup([a, b], {}).clusters) == 2


def test_detect_text_fast_passthrough_when_disabled():
    a = [_path(0, (0, 0, 10, 10))]
    res = detect_text_fast([], [a], None, page=None, enable_fast=False)
    assert res.passed == [a]
    assert res.dropped == []
    assert res.page_result.page_image is None
    assert res.page_result.detect_seconds is None


def test_build_drawing_output_restores_seq_order():
    dv = build_drawing_output(
        [_path(0, (0, 0, 1, 1)), _path(2, (0, 0, 1, 1))],
        [[_path(1, (0, 0, 1, 1))]],
        [[_path(3, (0, 0, 1, 1))]],
    )
    assert [d.paths[0].seq for d in dv] == [0, 1, 2, 3]
