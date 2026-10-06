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

@pytest.mark.parametrize("angle,value", [(0, 0.25), (45, 0.5), (90, 0.75), (135, 0.5), (179.999, 0.25)])
def test_angle_value_triangle_wave(angle, value):
    assert h.angle_value(angle) == pytest.approx(value, abs=1e-4)


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


# ---- crossing segments ---------------------------------------------------

def test_segment_count_line_through_rect(vector):
    rect = vector(kind="re", bbox=(0, 0, 10, 10))
    line = _line(vector, (-5, 5), (15, 5))  # crosses left + right edges
    assert h.crossing_segment_counts([rect, line]) == [1, 2]


def test_segment_count_one_foreign_segment_counts_once(vector):
    zigzag = _poly(vector, (0, -5), (2, 5), (4, -5), (6, 5))
    axis = _line(vector, (-1, 0), (7, 0))
    # axis is crossed by 3 zigzag segments; zigzag only by the one axis segment
    assert h.crossing_segment_counts([zigzag, axis]) == [1, 3]


def test_segment_count_ignores_touching_and_self(vector):
    bar = _line(vector, (0, 0), (10, 0))
    stem = _line(vector, (5, 0), (5, 10))
    corner = _line(vector, (10, 0), (10, 10))
    assert h.crossing_segment_counts([bar, stem, corner]) == [0, 0, 0]
    bowtie = _poly(vector, (0, 0), (10, 10), (10, 0), (0, 10))
    assert h.crossing_segment_counts([bowtie]) == [0]


def test_segment_count_sums_over_partners(vector):
    axis = _line(vector, (0, 5), (20, 5))
    verticals = [_line(vector, (x, 0), (x, 10)) for x in (2, 6, 10)]
    assert h.crossing_segment_counts([axis, *verticals]) == [3, 1, 1, 1]


# ---- percentile buckets -----------------------------------------

def test_percentile_buckets_even_split():
    values = list(range(1, 11))
    buckets = h.percentile_buckets(values, 5)
    assert [(lo, hi) for lo, hi, _ in buckets] == [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
    assert sorted(i for _, _, m in buckets for i in m) == list(range(10))


def test_percentile_buckets_keep_ties_together_and_drop_empty():
    values = [1] * 8 + [2, 9]
    buckets = h.percentile_buckets(values, 5)
    ranges = [(lo, hi) for lo, hi, _ in buckets]
    assert ranges[0] == (1, 1) and len(buckets[0][2]) == 8
    assert len(buckets) < 5
    for lo, hi, m in buckets:
        assert all(lo <= values[i] <= hi for i in m)


def test_percentile_buckets_empty():
    assert h.percentile_buckets([], 5) == []


# ---- exact value layers -------------------------------------------------

def test_exact_value_layers_merges_high_tail_under_fraction():
    counts = {1: 50, 2: 30, 3: 10, 4: 3, 7: 2, 12: 1}  # 96 total; 5% = 4.8
    values = [v for v, n in counts.items() for _ in range(n)]
    layers = h.exact_value_layers(values, 0.05)
    # tail {7, 12} = 3 < 4.8; adding 4 -> 6 >= 4.8, so 4 keeps its own layer
    assert [(lo, hi) for lo, hi, _ in layers] == [(1, 1), (2, 2), (3, 3), (4, 4), (7, 12)]
    assert len(layers[-1][2]) == 3
    assert sorted(i for _, _, m in layers for i in m) == list(range(len(values)))
    for lo, hi, m in layers:
        assert all(lo <= values[i] <= hi for i in m)


def test_exact_value_layers_single_value_and_no_merge():
    assert [(lo, hi) for lo, hi, _ in h.exact_value_layers([3, 3, 3])] == [(3, 3)]
    values = [1] * 50 + [2, 5]
    assert [(lo, hi) for lo, hi, _ in h.exact_value_layers(values, 0.0)] == [(1, 1), (2, 2), (5, 5)]


def test_exact_value_layers_empty():
    assert h.exact_value_layers([]) == []


def test_is_all_lines(vector):
    assert h.is_all_lines(_poly(vector, (0, 0), (5, 5), (10, 0)))
    assert not h.is_all_lines(vector(kind="re", bbox=(0, 0, 10, 10)))
    assert not h.is_all_lines(vector(items=[("l", (0, 0), (1, 1)), ("c", (1, 1), (2, 2), (3, 3), (4, 4))],
                                     bbox=(0, 0, 4, 4)))
    assert not h.is_all_lines(vector(items=[], bbox=(0, 0, 1, 1)))


# ---- gradient4 -----------------------------------------------------------

def test_gradient4_anchor_colours():
    stops = h._GRADIENT_STOPS
    assert h.gradient4(0.0) == pytest.approx(stops[0])          # green
    assert h.gradient4(1 / 3) == pytest.approx(stops[1])        # yellow
    assert h.gradient4(2 / 3) == pytest.approx(stops[2])        # red
    assert h.gradient4(1.0) == pytest.approx(stops[3])          # purple


def test_gradient4_clamps_and_interpolates():
    assert h.gradient4(-1.0) == pytest.approx(h._GRADIENT_STOPS[0])
    assert h.gradient4(2.0) == pytest.approx(h._GRADIENT_STOPS[3])
    mid = h.gradient4(1 / 6)  # halfway green -> yellow
    expected = tuple((a + b) / 2 for a, b in zip(h._GRADIENT_STOPS[0], h._GRADIENT_STOPS[1]))
    assert mid == pytest.approx(expected)


# ---- group summaries -----------------------------------------------------

def test_summarise_groups(vector):
    straight = _straight(vector, ((0, 0), (2, 0)), ((5, 0), (9, 0)), ((0, 9), (3, 9)))
    groups = [straight[:2], straight[2:]]
    summary = h.summarise_groups(groups)
    assert len(summary.multi) == 1 and len(summary.singles) == 1
    assert (summary.c_lo, summary.c_hi) == (2, 2)
    assert summary.s_lo == pytest.approx(1.0) and summary.s_hi == pytest.approx(1.0)


def test_add_gradient_layers_writes_gradient_and_other_stages(vector, tmp_path, monkeypatch):
    monkeypatch.setattr(h, "output_dir", lambda *parts: tmp_path)
    meta = PageMeta(index=0, number=1, mediabox=(0, 0, 100, 100), rotation=0, width=100, height=100)
    rep = h.DebugReport("probe", str(tmp_path / "src.pdf"), meta)
    straight = _straight(vector, ((0, 0), (2, 0)), ((5, 0), (9, 0)), ((0, 9), (3, 9)))
    excluded = [vector(kind="re", bbox=(50, 50, 60, 60))]
    h.add_gradient_layers(rep, h.summarise_groups([straight[:2], straight[2:]]), excluded,
                          excluded_label="excluded (non-straight)")
    assert [(e["stage"], e["layer"].split(" (")[0]) for e in rep.layers] == [
        ("gradient", "vector count"),
        ("gradient", "length std"),
        ("other", "singletons"),
        ("other", "excluded"),
    ]
