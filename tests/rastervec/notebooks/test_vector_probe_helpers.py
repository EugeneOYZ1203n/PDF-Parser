"""Unit tests for `rastervec/notebooks/_vector_probe_helpers.py` -- the
geometry/colour/manifest helpers behind the three vector-probe notebooks."""
from __future__ import annotations

import json

import pytest

from rastervec.commons.models import PageMeta
from rastervec.notebooks import _vector_probe_helpers as h


def _line(vector, p0, p1, **kw):
    bbox = (min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1]))
    return vector(items=[("l", p0, p1)], bbox=bbox, **kw)


def _poly(vector, *pts):
    items = [("l", pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return vector(items=items, bbox=(min(xs), min(ys), max(xs), max(ys)))


# ---- crossings ---------------------------------------------------------

def test_x_crossing_counts_one_partner_each(vector):
    a = _line(vector, (0, 0), (10, 10))
    b = _line(vector, (0, 10), (10, 0))
    assert h.crossing_partner_counts([a, b]) == [1, 1]


def test_t_junction_and_shared_corner_do_not_count(vector):
    bar = _line(vector, (0, 0), (10, 0))
    stem = _line(vector, (5, 0), (5, 10))      # T: stem's end rests on bar
    corner = _line(vector, (10, 0), (10, 10))  # L: shares bar's endpoint
    assert h.crossing_partner_counts([bar, stem, corner]) == [0, 0, 0]


def test_collinear_overlap_does_not_count(vector):
    a = _line(vector, (0, 0), (10, 0))
    b = _line(vector, (5, 0), (15, 0))
    assert h.crossing_partner_counts([a, b]) == [0, 0]


def test_self_crossing_does_not_count(vector):
    bowtie = _poly(vector, (0, 0), (10, 10), (10, 0), (0, 10))
    assert h.crossing_partner_counts([bowtie]) == [0]


def test_partner_counted_once_despite_multiple_crossings(vector):
    zigzag = _poly(vector, (0, -5), (2, 5), (4, -5), (6, 5))
    axis = _line(vector, (-1, 0), (7, 0))
    assert h.crossing_partner_counts([zigzag, axis]) == [1, 1]


def test_rect_crossed_by_line(vector):
    rect = vector(kind="re", bbox=(0, 0, 10, 10))
    line = _line(vector, (-5, 5), (5, 5))  # crosses the left edge
    assert h.crossing_partner_counts([rect, line]) == [1, 1]


def test_curve_is_flattened_for_crossing(vector):
    arch = vector(items=[("c", (0, 0), (0, -10), (10, -10), (10, 0))], bbox=(0, -7.5, 10, 0))
    line = _line(vector, (4, -10), (4, -2))  # off the t=0.5 flattening vertex
    assert h.crossing_partner_counts([arch, line]) == [1, 1]


# ---- straight lines ------------------------------------------------------

def test_straight_line_accepts_collinear_multi_item_vector(vector):
    v = _poly(vector, (0, 0), (5, 0), (10, 0))
    fit = h.straight_line(v)
    assert fit is not None
    assert fit.angle == pytest.approx(0.0)
    assert fit.length == pytest.approx(10.0)
    assert fit.midpoint == pytest.approx((5.0, 0.0))


def test_straight_line_folds_angle_into_0_180(vector):
    fit = h.straight_line(_line(vector, (10, 10), (0, 0)))
    assert fit.angle == pytest.approx(45.0)


@pytest.mark.parametrize("build", [
    lambda vector: _poly(vector, (0, 0), (5, 0), (5, 5)),
    lambda vector: vector(items=[("c", (0, 0), (1, 1), (2, 1), (3, 0))], bbox=(0, 0, 3, 1)),
    lambda vector: vector(kind="re", bbox=(0, 0, 5, 5)),
    lambda vector: _line(vector, (1, 1), (1, 1)),
])
def test_straight_line_rejects_non_straight(vector, build):
    assert h.straight_line(build(vector)) is None


# ---- grouping ----------------------------------------------------------

def _straight(vector, *segments):
    vs = [_line(vector, p0, p1) for p0, p1 in segments]
    straight, excluded = h.split_straight(vs)
    assert not excluded
    return straight


def test_collinear_dashes_group_and_offset_line_splits(vector):
    straight = _straight(
        vector,
        ((0, 0), (2, 0)), ((4, 0), (6, 0)), ((100, 0), (102, 0)),  # same line, far apart
        ((0, 3), (2, 3)),                                         # parallel, offset 3
    )
    groups = h.group_collinear(straight, angle_tol=1.0, offset_tol=0.5)
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 3]


def test_collinear_joins_across_angle_wrap(vector):
    import math
    a = math.radians(179.8)
    straight = _straight(
        vector,
        ((0, 0), (10 * math.cos(a), 10 * math.sin(a))),
        ((20, 0), (30, 0.0175)),  # ~0.1 deg
    )
    assert [len(g) for g in h.group_collinear(straight)] == [2]


def test_parallel_groups_by_angle_regardless_of_offset(vector):
    straight = _straight(
        vector,
        ((0, 0), (10, 0)), ((0, 50), (3, 50)),   # horizontal
        ((0, 0), (0, 10)), ((40, 0), (40, 7)),   # vertical
        ((0, 0), (10, 10)),                      # diagonal
    )
    groups = h.group_parallel(straight)
    assert sorted(len(g) for g in groups) == [1, 2, 2]


def test_group_stats(vector):
    straight = _straight(vector, ((0, 0), (2, 0)), ((5, 0), (9, 0)))
    count, std = h.group_stats(straight)
    assert count == 2
    assert std == pytest.approx(1.0)


# ---- colour ------------------------------------------------------------

def test_crossing_ramp_anchors():
    assert h.ramp_crossings(0) == (1.0, 0.0, 0.0)
    assert h.ramp_crossings(3) == h._CROSSING_ANCHORS[1][1]
    assert h.ramp_crossings(5) == h._CROSSING_ANCHORS[2][1]
    assert h.ramp_crossings(7) == h.ramp_crossings(12) == h._CROSSING_ANCHORS[3][1]


@pytest.mark.parametrize("angle,value", [(0, 0.25), (45, 0.5), (90, 0.75), (135, 0.5), (179.999, 0.25)])
def test_angle_value_triangle_wave(angle, value):
    assert h.angle_value(angle) == pytest.approx(value, abs=1e-4)


def test_green_red_endpoints():
    g, r = h.green_red(0.0), h.green_red(1.0)
    assert g[1] > g[0] and g[2] == 0
    assert r[0] > r[1] and r[2] == 0


# ---- report ------------------------------------------------------------

def test_debug_report_manifest_matches_viewer_contract(vector, tmp_path, monkeypatch):
    monkeypatch.setattr(h, "output_dir", lambda *parts: tmp_path)
    meta = PageMeta(index=2, number=3, mediabox=(0, 0, 100, 100), rotation=90, width=100, height=100)
    rep = h.DebugReport("probe", str(tmp_path / "src.pdf"), meta)
    rep.add_vectors("stage a", "all (ramp)", (1, 0, 0), [_line(vector, (0, 0), (10, 10))], lambda v: (1, 0, 0))
    rep.add_vectors("stage a", "empty", (1, 0, 0), [], lambda v: (1, 0, 0))
    rep.finish()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pages"] == [2]
    assert manifest["source_pdf"].endswith("src.pdf")
    assert [e["layer"] for e in manifest["layers"]] == ["all (ramp)"]
    entry = manifest["layers"][0]
    assert entry["color"] == "#ff0000"
    assert (tmp_path / entry["file"]).is_file()


def test_collinear_offsets_do_not_chain(vector):
    # Offsets 0.3 pt apart: single-linkage at 0.5 would chain all six into
    # one 1.5 pt-wide group; anchored grouping caps each group's span at 0.5.
    straight = _straight(vector, *[((0, 0.3 * i), (5, 0.3 * i)) for i in range(6)])
    groups = h.group_collinear(straight, offset_tol=0.5)
    assert sorted(len(g) for g in groups) == [2, 2, 2]


def test_parallel_tolerates_angle_jitter_beyond_one_tolerance(vector):
    # Short hatch segments jitter +-1 deg around 135; single-linkage on angle
    # keeps them in one group even though the total spread exceeds angle_tol.
    import math
    segs = []
    for i in range(21):
        a = math.radians(134.0 + 0.1 * i)
        segs.append(((0, 5 * i), (6 * math.cos(a), 5 * i + 6 * math.sin(a))))
    assert [len(g) for g in h.group_parallel(_straight(vector, *segs), angle_tol=1.0)] == [21]
