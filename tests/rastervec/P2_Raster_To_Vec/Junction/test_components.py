from __future__ import annotations

import numpy as np
import pytest

from rastervec.P2_Raster_To_Vec.Junction.components import iter_layer_components


def _two_blobs(gap: int):
    labels = np.zeros((60, 200), dtype=np.uint8)
    labels[20:30, 10:50] = 1
    labels[20:30, 50 + gap:90 + gap] = 1
    labels[5:8, 150:190] = 2  # another layer -- never part of layer 1's crops
    enhanced = np.where(labels > 0, 0, 255).astype(np.uint8)
    return labels, enhanced


@pytest.mark.parametrize("gap,expected", [(5, 1), (20, 2)])
def test_tolerance_joins_or_splits_components(gap, expected):
    labels, enhanced = _two_blobs(gap)
    comps = list(iter_layer_components(labels, enhanced, 1, tol_px=10.0))
    assert len(comps) == expected


def test_crop_contains_only_this_layer_and_offsets_are_right():
    labels, enhanced = _two_blobs(20)
    comps = list(iter_layer_components(labels, enhanced, 1, tol_px=10.0))
    total_ink = 0
    for c in comps:
        ys, xs = np.nonzero(c.gray < 128)
        total_ink += xs.size
        assert (labels[ys + c.y0, xs + c.x0] == 1).all()
    assert total_ink == int((labels == 1).sum())


def test_missing_layer_yields_nothing():
    labels, enhanced = _two_blobs(5)
    assert list(iter_layer_components(labels, enhanced, 7, tol_px=10.0)) == []


@pytest.mark.parametrize("gap", [5, 20])
def test_layer_components_count_matches_yielded(gap):
    from rastervec.P2_Raster_To_Vec.Junction.components import layer_components

    labels, enhanced = _two_blobs(gap)
    count, it = layer_components(labels, enhanced, 1, tol_px=10.0)
    assert count == len(list(it))
    assert layer_components(labels, enhanced, 7, tol_px=10.0)[0] == 0
