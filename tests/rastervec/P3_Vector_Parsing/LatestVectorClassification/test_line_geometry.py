from __future__ import annotations

import math

import pytest

from rastervec.P3_Vector_Parsing.LatestVectorClassification import line_geometry as lg


def _line(vector, x0, y0, x1, y1, **kw):
    return vector(bbox=(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
                  items=[("l", (x0, y0), (x1, y1))], **kw)


def test_straight_line_fits_polyline_and_rejects_bend(vector):
    v = vector(items=[("l", (0.0, 0.0), (5.0, 5.0)), ("l", (5.0, 5.0), (10.0, 10.0))])
    fit = lg.straight_line(v, 0.25)
    assert fit is not None
    assert fit.angle == pytest.approx(45.0)
    assert fit.length == pytest.approx(10.0 * 2 ** 0.5)
    bent = vector(items=[("l", (0.0, 0.0), (5.0, 0.0)), ("l", (5.0, 0.0), (5.0, 5.0))])
    assert lg.straight_line(bent, 0.25) is None
    assert lg.straight_line(vector(kind="re", bbox=(0, 0, 5, 5)), 0.25) is None


def test_group_collinear_splits_parallel_offsets(vector):
    straight, _ = lg.split_straight(
        [_line(vector, x, 0, x + 2, 0) for x in range(0, 20, 4)]
        + [_line(vector, x, 5, x + 2, 5) for x in range(0, 20, 4)],
        0.25,
    )
    groups = lg.group_collinear(straight, 1.0, 0.5)
    assert sorted(len(g) for g in groups) == [5, 5]


def test_group_parallel_wraps_around_180(vector):
    straight, _ = lg.split_straight([
        _line(vector, 0, 0, 10, 0.05),     # ~0.3 deg
        _line(vector, 0, 0.05, 10, 0),     # ~179.7 deg
    ], 0.25)
    assert len(lg.group_parallel(straight, 1.0)) == 1


def test_golden_hues_are_distinct():
    hues = lg.golden_hues(5)
    assert len(set(round(h, 6) for h in hues)) == 5 and all(0.0 <= h < 1.0 for h in hues)


def _curve(vector, p0, p1, p2, p3):
    xs, ys = [p[0] for p in (p0, p1, p2, p3)], [p[1] for p in (p0, p1, p2, p3)]
    return vector(kind="c", bbox=(min(xs), min(ys), max(xs), max(ys)), items=[("c", p0, p1, p2, p3)])


def test_line_crossing_counts_distinct_segments_and_skips_non_line_vectors(vector):
    hatch = [_line(vector, x, -5, x, 5) for x in range(1, 6)]
    bar = _line(vector, 0, 0, 10, 0)
    touching = _line(vector, 10, 0, 10, 5)  # endpoint on the bar's end -- not a crossing
    rect = vector(kind="re", bbox=(20, 20, 30, 30), items=[("re", (20, 20, 30, 30))])
    counts = lg.line_crossing_counts([bar, touching, *hatch, rect], eps=0.01)
    assert counts[0] == 5
    assert counts[1] == 0
    assert counts[2:7] == [1] * 5
    assert counts[7] is None  # not "l"-only -- never a candidate


def test_line_through_rect_counts_each_crossed_edge(vector):
    rect = vector(kind="re", bbox=(2, -2, 8, 2), items=[("re", (2, -2, 8, 2))])
    bar = _line(vector, 0, 0, 10, 0)
    assert lg.line_crossing_counts([bar, rect], eps=0.01)[0] == 2


def test_line_through_quad_counts_each_crossed_edge(vector):
    quad = vector(kind="qu", bbox=(2, -2, 8, 2), items=[("qu", ((2, -2), (8, -2), (8, 2), (2, 2)))])
    bar = _line(vector, 0, 0, 10, 0)
    assert lg.line_crossing_counts([bar, quad], eps=0.01)[0] == 2


def test_s_curve_crossing_three_times_counts_three(vector):
    bar = _line(vector, 0, 0, 10, 0)
    s_curve = _curve(vector, (1, -5), (12, 15), (-2, -15), (9, 5))
    assert lg.line_cubic_crossings((0, 0, 10, 0), ((1, -5), (12, 15), (-2, -15), (9, 5)), 0.01) == 3
    assert lg.line_crossing_counts([bar, s_curve], eps=0.01)[0] == 3


def test_tangent_curve_does_not_cross():
    # Bezier touching y=0 at its apex (x=5) from below, never crossing.
    assert lg.line_cubic_crossings((0, 0, 10, 0), ((0, -4), (3, 4 / 3), (7, 4 / 3), (10, -4)), 0.01) == 0
    # A curve wholly on one side.
    assert lg.line_cubic_crossings((0, 0, 10, 0), ((0, -4), (3, -1), (7, -1), (10, -4)), 0.01) == 0


def test_curve_crossing_outside_the_segment_does_not_count():
    # Crosses the infinite line y=0 at x=15, beyond the segment's end.
    assert lg.line_cubic_crossings((0, 0, 10, 0), ((15, -5), (15, -1), (15, 1), (15, 5)), 0.01) == 0


def test_grid_direction_mixed_and_rectilinear(vector):
    ell = vector(bbox=(0, 0, 10, 10), items=[("l", (0, 0), (10, 0)), ("l", (10, 0), (10, 10))])
    d, length = lg.grid_direction(ell, 2.0)
    assert d == pytest.approx(0.0, abs=1e-9) or d == pytest.approx(90.0, abs=1e-9)
    assert length == pytest.approx(20.0)
    slanted = vector(bbox=(0, 0, 18.66, 5), items=[("l", (0, 0), (10, 0)), ("l", (10, 0), (18.66, 5))])
    assert lg.grid_direction(slanted, 2.0)[0] is None


def test_dominant_grid_single_candidate_is_its_own_majority(vector):
    bar = _line(vector, 0, 0, 10, 0)
    assert lg.dominant_grid([bar], tol=2.0, dominance=0.5) == [bar]


def test_ink_fraction_full_partial_and_outside(vector):
    quad = ((0, 0), (10, 0), (10, 10), (0, 10))
    inside = _line(vector, 1, 5, 9, 5)
    half = _line(vector, 5, 5, 15, 5)
    outside = _line(vector, 20, 5, 30, 5)
    assert lg.ink_fraction_in_quad(inside, quad, curve_samples=16) == pytest.approx(1.0)
    assert lg.ink_fraction_in_quad(half, quad, curve_samples=16) == pytest.approx(0.5)
    assert lg.ink_fraction_in_quad(outside, quad, curve_samples=16) == 0.0


def test_ink_fraction_uses_the_rotated_quad_not_its_envelope(vector):
    # A 30-deg-rotated thin quad around the origin; a horizontal line inside
    # its envelope but mostly outside the quad itself.
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    hw, hh = 20.0, 2.0
    quad = [(x * c - y * s, x * s + y * c) for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
    line = _line(vector, -15, 0, 15, 0)
    xs, ys = [p[0] for p in quad], [p[1] for p in quad]
    assert min(xs) < -15 and max(xs) > 15 and min(ys) < 0 < max(ys)  # inside the envelope
    assert lg.ink_fraction_in_quad(line, quad, curve_samples=16) < 0.5


def test_ink_fraction_zero_length_vector_uses_bbox_centre(vector):
    quad = ((0, 0), (10, 0), (10, 10), (0, 10))
    dot = vector(bbox=(5, 5, 5, 5), items=[("l", (5, 5), (5, 5))])
    far = vector(bbox=(50, 5, 50, 5), items=[("l", (50, 5), (50, 5))])
    assert lg.ink_fraction_in_quad(dot, quad, curve_samples=16) == 1.0
    assert lg.ink_fraction_in_quad(far, quad, curve_samples=16) == 0.0


def test_quad_area_and_bbox_inside_rotated_quad():
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    quad = [(x * c - y * s, x * s + y * c) for x, y in ((-20, -5), (20, -5), (20, 5), (-20, 5))]
    assert lg.quad_area(quad) == pytest.approx(400.0)
    assert lg.bbox_inside_quad((-1, -1, 1, 1), quad)
    # Inside the envelope (corner region), outside the rotated quad.
    assert not lg.bbox_inside_quad((-16, 4, -14, 6), quad)


def test_piece_overlap_fraction_counts_pieces_touching_the_quad(vector):
    quad = ((0, 0), (10, 0), (10, 10), (0, 10))
    v = vector(bbox=(1, 1, 30, 5), items=[
        ("l", (1, 1), (5, 1)),     # inside
        ("l", (5, 5), (15, 5)),    # crosses the edge
        ("l", (20, 5), (30, 5)),   # outside
        ("l", (-5, 5), (15, 5)),   # passes through, no endpoint inside
    ])
    assert lg.piece_overlap_fraction(v, quad) == pytest.approx(0.75)
    outside = _line(vector, 20, 5, 30, 5)
    assert lg.piece_overlap_fraction(outside, quad) == 0.0


def test_piece_overlap_fraction_uses_cubic_chord_and_nan_when_empty(vector):
    quad = ((0, 0), (10, 0), (10, 10), (0, 10))
    curve = vector(bbox=(2, 2, 8, 8), items=[("c", (2, 2), (2, 8), (8, 8), (8, 2))])
    assert lg.piece_overlap_fraction(curve, quad) == 1.0
    dot = vector(bbox=(5, 5, 5, 5), items=[("l", (5, 5), (5, 5))])
    assert math.isnan(lg.piece_overlap_fraction(dot, quad))
