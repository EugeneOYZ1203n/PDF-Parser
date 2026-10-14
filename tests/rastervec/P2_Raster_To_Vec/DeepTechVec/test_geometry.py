from __future__ import annotations

import numpy as np
import pytest

from rastervec.P2_Raster_To_Vec.DeepTechVec import geometry as geo


def _cubic_eval(piece: np.ndarray, t) -> np.ndarray:
    t = np.asarray(t, float)[:, None]
    p0, c1, c2, p3 = piece
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * p3


def test_quad_to_cubic_is_exact():
    q = np.array([[0.0, 0.0], [10.0, 30.0], [40.0, 5.0]])
    t = np.linspace(0, 1, 21)
    assert np.allclose(geo.quad_eval(q, t), _cubic_eval(geo.quad_to_cubic(q), t))


def test_split_quad_matches_the_original():
    q = np.array([[0.0, 0.0], [10.0, 30.0], [40.0, 5.0]])
    sub = geo.split_quad(q, 0.25, 0.75)
    assert np.allclose(geo.quad_eval(sub, [0.0, 0.5, 1.0]), geo.quad_eval(q, [0.25, 0.5, 0.75]))


def test_clip_prim_line_and_curve():
    line = geo.Prim(np.array([[-10.0, 5.0], [50.0, 5.0]]), 2.0)
    (part,) = geo.clip_prim(line, (0, 0, 32, float("inf")))
    assert np.allclose(part.pts, [[0, 5], [32, 5]]) and part.width == 2.0
    curve = geo.Prim(np.array([[0.0, 0.0], [20.0, 40.0], [40.0, 0.0]]), 1.0)
    parts = geo.clip_prim(curve, (10, -1, 30, 100))
    assert len(parts) == 1
    xs = geo.prim_points(parts[0], 16)[:, 0]
    assert xs.min() == pytest.approx(10, abs=0.05) and xs.max() == pytest.approx(30, abs=0.05)


@pytest.mark.parametrize("total", [1000, 300, 64, 200])
def test_patch_cores_partition_the_axis(total):
    """The last patch is shifted back to the edge; ownership must still be
    exactly one patch per pixel."""
    xs = np.arange(total) + 0.5
    owners = np.zeros(total, int)
    for s in geo.tile_starts(total, 64, 16):
        a, b = geo.core_spans(total, 64, 16)[s]
        owners += (xs >= a) & (xs < b)
    assert (owners == 1).all()
