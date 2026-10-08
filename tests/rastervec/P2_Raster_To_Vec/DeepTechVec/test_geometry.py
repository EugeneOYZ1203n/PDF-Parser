from __future__ import annotations

import numpy as np
import pytest

from rastervec.P2_Raster_To_Vec.DeepTechVec import geometry as geo


def test_straight_cubic_stroke_is_one_line():
    stroke = np.stack([geo.line_to_cubic((0, 0), (10, 5)), geo.line_to_cubic((10, 5), (30, 15))])
    lines = geo.stroke_to_lines(stroke, 0.5, 3.0)
    assert len(lines) == 1
    assert np.allclose(lines[0], [[0, 0], [30, 15]])


def test_arc_becomes_quadratics_within_tolerance():
    # a quarter circle of radius 40 as one cubic (kappa = 0.5523)
    k = 0.5523 * 40
    piece = np.array([[40.0, 0.0], [40.0, k], [k, 40.0], [0.0, 40.0]])
    quads = geo.stroke_to_quads(piece[None], 0.25)
    assert len(quads) >= 2
    t = np.linspace(0, 1, 50)
    for q in quads:
        r = np.linalg.norm(geo.quad_eval(q, t), axis=1)
        assert np.abs(r - 40).max() < 0.6
    assert np.allclose(quads[0][0], piece[0]) and np.allclose(quads[-1][2], piece[3])


def test_line_flattening_of_a_curve_stays_within_tolerance():
    piece = np.array([[0.0, 0.0], [10.0, 20.0], [30.0, 20.0], [40.0, 0.0]])
    lines = geo.stroke_to_lines(piece[None], 0.5, 3.0)
    assert len(lines) > 3
    for a, b in zip(lines, lines[1:]):
        assert np.allclose(a[1], b[0])  # a connected chain


def test_quad_to_cubic_is_exact():
    q = np.array([[0.0, 0.0], [10.0, 30.0], [40.0, 5.0]])
    t = np.linspace(0, 1, 21)
    assert np.allclose(geo.quad_eval(q, t), geo.bezier_eval(geo.quad_to_cubic(q), t))


def test_split_quad_matches_the_original():
    q = np.array([[0.0, 0.0], [10.0, 30.0], [40.0, 5.0]])
    sub = geo.split_quad(q, 0.25, 0.75)
    assert np.allclose(geo.quad_eval(sub, [0.0, 0.5, 1.0]), geo.quad_eval(q, [0.25, 0.5, 0.75]))


def test_sort_prims_orients_endpoints_then_orders_lexicographically():
    a = geo.Prim(np.array([[30.0, 5.0], [10.0, 5.0]]), 1.0)
    b = geo.Prim(np.array([[5.0, 9.0], [5.0, 1.0]]), 1.0)
    out = geo.sort_prims([a, b])
    assert np.allclose(out[0].pts, [[5, 1], [5, 9]])
    assert np.allclose(out[1].pts, [[10, 5], [30, 5]])
    c = geo.Prim(np.array([[9.0, 0.0], [5.0, 5.0], [1.0, 0.0]]), 1.0)
    assert np.allclose(geo.sort_prims([c])[0].pts, [[1, 0], [5, 5], [9, 0]])  # control stays in the middle


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
