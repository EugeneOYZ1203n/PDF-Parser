from __future__ import annotations

import numpy as np
import pytest

from rastervec.P2_Raster_To_Vec.ImplicitSketchVec import geometry as geo
from rastervec.P2_Raster_To_Vec.ImplicitSketchVec import postprocess as pp
from rastervec.P2_Raster_To_Vec.ImplicitSketchVec import train_data as td


def _line(a, b):
    return geo.line_to_cubic(a, b)[None]


def _pipeline(strokes, size=32, k=1):
    """GT fields (as a perfect network would predict them) -> grouped lines."""
    t = td.make_targets(strokes, size)
    er, eb = pp.refine_flags(t["edge"] % 2 == 1, t["edge"] >= 2)
    ys, xs = np.nonzero(t["vmask"])
    vp = {(int(a), int(b)): np.array([(a + t["vert"][0, b, a]) / 2, (b + t["vert"][1, b, a]) / 2])
          for b, a in zip(ys, xs)}
    g = pp.build_graph(np.argwhere(er)[:, ::-1], np.argwhere(eb)[:, ::-1], vp)
    pp.repair_breaks(g)
    ys, xs = np.nonzero(t["usm"])
    usm = {(int(a), int(b)) for b, a in zip(ys, xs)}
    pp.refine_topology(g, usm, np.concatenate([t["keypoints"][n] for n in ("end", "sharp", "junc")]))
    g = pp.downsample(g, k, usm)
    pp.split_crossings(g)
    return pp.group_lines(g, 1.0)


def _ends(pl):
    return sorted([tuple(np.round(pl[0], 0)), tuple(np.round(pl[-1], 0))])


@pytest.mark.parametrize("k", [1, 2])
def test_straight_line(k):
    (pl,) = _pipeline([_line((2, 10.3), (30, 10.3))], k=k)
    assert _ends(pl) == [(2.0, 10.0), (30.0, 10.0)]
    assert np.abs(pl[:, 1] - 10.3).max() < 0.15


def test_diagonal_line_is_whole():
    (pl,) = _pipeline([_line((2, 3.3), (29, 27.1))])
    assert np.hypot(*(pl[0] - pl[-1])) == pytest.approx(np.hypot(27, 23.8), abs=1.5)


@pytest.mark.parametrize("k", [1, 2])
def test_crossing_lines_stay_two_straight_strokes(k):
    lines = _pipeline([_line((2, 2.2), (30, 29.7)), _line((2, 29.6), (29.4, 2.3))], k=k)
    assert len(lines) == 2
    for pl in lines:
        d = pl[-1] - pl[0]
        assert np.hypot(*d) > 35  # corner to corner, not bent at the crossing


def test_plus_and_tee():
    plus = _pipeline([_line((2, 16.2), (30, 16.2)), _line((16.3, 2), (16.3, 30))])
    assert sorted(len(p) > 0 and round(abs(p[-1] - p[0]).max()) for p in plus) == [28, 28]
    tee = _pipeline([_line((2, 16.2), (30, 16.2)), _line((16.3, 16.2), (16.3, 30))])
    horiz = [p for p in tee if abs(p[-1, 1] - p[0, 1]) < 1]
    assert len(horiz) == 1 and abs(horiz[0][-1, 0] - horiz[0][0, 0]) > 27


def test_closed_outline_is_one_stroke():
    rect = np.stack([geo.line_to_cubic((4.2, 4.2), (27.8, 4.2)), geo.line_to_cubic((27.8, 4.2), (27.8, 27.8)),
                     geo.line_to_cubic((27.8, 27.8), (4.2, 27.8)), geo.line_to_cubic((4.2, 27.8), (4.2, 4.2))])
    (pl,) = _pipeline([rect])
    assert np.allclose(pl[0], pl[-1])


def test_lines_closer_than_a_cell_are_kept():
    lines = _pipeline([_line((2, 10.1), (30, 10.1)), _line((2, 10.4), (30, 10.4))])
    assert len(lines) == 1


def test_refine_flags_thins_a_doubled_band_but_not_a_staircase():
    er = np.zeros((10, 20), bool)
    er[4:6, 2:15] = True                    # two parallel rows of flags
    er2, _ = pp.refine_flags(er, np.zeros_like(er))
    assert er2.sum() < er.sum()
    stair_r = np.zeros((10, 10), bool)
    stair_b = np.zeros((10, 10), bool)
    for i in range(6):
        stair_r[i, i] = True
        stair_b[i, i + 1] = True
    r2, b2 = pp.refine_flags(stair_r, stair_b)
    assert (r2 == stair_r).all() and (b2 == stair_b).all()


def test_repair_breaks_joins_adjacent_open_ends():
    g = pp.DCGraph()
    for c in [(0, 0), (1, 0), (2, 1), (3, 1)]:
        g.add_node(c, pp.cell_centre(c))
    g.add_edge((0, 0), (1, 0))
    g.add_edge((2, 1), (3, 1))
    assert pp.repair_breaks(g) == 1 and (2, 1) in g.adj[(1, 0)]


def test_topology_with_a_keypoint_makes_a_hub():
    """Fig. 7 (I): an under-sampled region with a keypoint inside -> its
    vertices removed and every truncated branch joined to the keypoint."""
    g = pp.DCGraph()
    g.add_node((0, 0), pp.cell_centre((0, 0)))
    for outer, inner in {(-4, 0): (-2, 0), (4, 0): (2, 0), (0, -4): (0, -2), (0, 4): (0, 2)}.items():
        g.add_node(outer, pp.cell_centre(outer))
        g.add_node(inner, pp.cell_centre(inner))
        g.add_edge(outer, inner)
        g.add_edge(inner, (0, 0))
    region = {(a, b) for a in range(-2, 3) for b in range(-2, 3)}
    (r,) = pp.refine_topology(g, region, np.array([[0.25, 0.25]]))
    hubs = [n for n in g.adj if n[0] == "new"]
    assert len(hubs) == 1 and g.degree(hubs[0]) == 4 and r == region
