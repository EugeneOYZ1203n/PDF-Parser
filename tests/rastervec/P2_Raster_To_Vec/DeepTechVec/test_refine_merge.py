from __future__ import annotations

import numpy as np
import pytest

from rastervec.P2_Raster_To_Vec.DeepTechVec import geometry as geo
from rastervec.P2_Raster_To_Vec.DeepTechVec import merge, refine


def _line_patch():
    g = np.full((64, 64), 255, np.uint8)
    g[31:34, 8:56] = 0  # a 3 px line centred on y = 32.5, x 8..56
    return g


def _line(a, b, w=2.0):
    return geo.Prim(np.array([a, b], float), w)


def test_refinement_pulls_an_offset_line_onto_the_ink():
    (out,) = refine.refine_patches([_line_patch()], [[_line((10, 35.5), (54, 35.5), 3.0)]], iters=80)
    (p,) = out
    assert np.abs(p.pts[:, 1] - 32.5).max() < 0.5
    assert p.width == pytest.approx(3.0, abs=0.5)


def test_refinement_aligns_a_curve_too():
    q = geo.Prim(np.array([[10.0, 35.0], [32.0, 36.0], [54.0, 35.0]]), 3.0)
    (out,) = refine.refine_patches([_line_patch()], [[q]], iters=80)
    assert np.abs(geo.prim_points(out[0], 8)[:, 1] - 32.5).max() < 0.6


def test_isolated_primitive_shrinks():
    """Appendix I, Fig. 24: with no ink nearby a primitive only feels its own
    repulsion, so it collapses."""
    empty = np.full((64, 64), 255, np.uint8)
    p0 = _line((10, 20), (40, 20), 3.0)
    out = refine.refine_patches([empty], [[p0]], iters=60)[0]
    assert not out or (out[0].length() < p0.length() and out[0].width < p0.width)


def test_zero_iterations_is_identity():
    p = _line((1, 2), (3, 4))
    assert refine.refine_patches([_line_patch()], [[p]], iters=0)[0][0] is p


def test_collinear_pieces_across_a_seam_merge_into_one_line():
    out = merge.merge_lines([_line((0, 10), (40, 10.3)), _line((38, 10.2), (90, 10.5))])
    assert len(out) == 1
    assert out[0].pts[:, 0].min() == pytest.approx(0, abs=0.1) and out[0].pts[:, 0].max() == pytest.approx(90, abs=0.1)


def test_parallel_offset_lines_do_not_merge():
    assert len(merge.merge_lines([_line((0, 10), (90, 10)), _line((0, 14), (90, 14))])) == 2


def test_dangling_end_past_an_intersection_is_cut():
    out = merge.merge_lines([_line((0, 10), (50, 10)), _line((48.5, 0), (48.5, 40))])
    horiz = next(p for p in out if abs(p.pts[0, 1] - p.pts[1, 1]) < 1e-6)
    assert horiz.pts[:, 0].max() == pytest.approx(48.5)


def test_two_halves_of_a_quadratic_refit_into_one():
    q = np.array([[0.0, 0.0], [20.0, 20.0], [40.0, 0.0]])
    out = merge.merge_curves([geo.Prim(geo.split_quad(q, 0, 0.5), 2.0), geo.Prim(geo.split_quad(q, 0.5, 1), 2.0)])
    assert len(out) == 1
    t = np.linspace(0, 1, 21)
    d = np.linalg.norm(geo.quad_eval(out[0].pts, t)[:, None] - geo.quad_eval(q, np.linspace(0, 1, 200))[None], axis=2)
    assert d.min(axis=1).max() < 0.5


def test_distant_curves_stay_separate():
    a = geo.Prim(np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 0.0]]), 2.0)
    b = geo.Prim(np.array([[0.0, 30.0], [10.0, 40.0], [20.0, 30.0]]), 2.0)
    assert len(merge.merge_curves([a, b])) == 2
