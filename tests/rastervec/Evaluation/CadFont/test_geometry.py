from __future__ import annotations

from rastervec.Evaluation.CadFont.geometry import (
    cubic_bezier_points,
    estimate_curve_length,
    flatten_item_to_segments,
    merge_close_points,
    split_at_intersections,
)


def test_cubic_bezier_points_n8_returns_9_points_with_exact_endpoints():
    p0, p1, p2, p3 = (0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)
    pts = cubic_bezier_points(p0, p1, p2, p3, n=8)
    assert len(pts) == 9
    assert pts[0] == p0
    assert pts[-1] == p3


def test_cubic_bezier_points_midpoint_matches_closed_form():
    p0, p1, p2, p3 = (0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)
    pts = cubic_bezier_points(p0, p1, p2, p3, n=2)
    # t=0.5 Bernstein: 0.125*p0 + 0.375*p1 + 0.375*p2 + 0.125*p3
    expected_x = 0.125 * 0 + 0.375 * 0 + 0.375 * 1 + 0.125 * 1
    expected_y = 0.125 * 0 + 0.375 * 1 + 0.375 * 1 + 0.125 * 0
    mx, my = pts[1]
    assert abs(mx - expected_x) < 1e-9
    assert abs(my - expected_y) < 1e-9


def test_estimate_curve_length_straight_line_matches_chord_distance():
    # A "curve" whose control points all lie on one line degenerates to a
    # straight segment -- its arc length should match the chord distance.
    p0, p1, p2, p3 = (0.0, 0.0), (2.0, 0.0), (5.0, 0.0), (9.0, 0.0)
    length = estimate_curve_length(p0, p1, p2, p3)
    assert abs(length - 9.0) < 1e-6


def test_estimate_curve_length_curved_path_exceeds_chord_distance():
    p0, p1, p2, p3 = (0.0, 0.0), (0.0, 5.0), (5.0, 5.0), (5.0, 0.0)
    length = estimate_curve_length(p0, p1, p2, p3)
    chord = 5.0  # straight-line distance from p0 to p3
    assert length > chord


def test_flatten_item_to_segments_line():
    item = ("l", (0.0, 0.0), (1.0, 1.0))
    segs, vertices = flatten_item_to_segments(item)
    assert segs == [((0.0, 0.0), (1.0, 1.0))]
    assert vertices == {(0.0, 0.0), (1.0, 1.0)}


def test_flatten_item_to_segments_curve_is_chained_and_adaptively_spaced():
    item = ("c", (0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0))
    segs, vertices = flatten_item_to_segments(item, curve_spacing=1.0)
    # Chained, with the curve's own endpoints at the very start/end.
    assert segs[0][0] == (0.0, 0.0)
    assert segs[-1][1] == (10.0, 0.0)
    for a, b in zip(segs, segs[1:]):
        assert a[1] == b[0]
    # Only the true endpoints are original vertices, never an interior
    # sample point.
    assert vertices == {(0.0, 0.0), (10.0, 0.0)}
    # Roughly 1pt-spaced samples over a curve with a decent arc length ->
    # meaningfully more than the old fixed-8 subdivision.
    assert len(segs) > 8


def test_flatten_item_to_segments_curve_coarser_spacing_gives_fewer_segments():
    item = ("c", (0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0))
    fine_segs, _ = flatten_item_to_segments(item, curve_spacing=0.5)
    coarse_segs, _ = flatten_item_to_segments(item, curve_spacing=5.0)
    assert len(coarse_segs) < len(fine_segs)


def test_flatten_item_to_segments_curve_never_below_two_segments():
    # A very short curve at a coarse spacing must still get >= 2 segments.
    item = ("c", (0.0, 0.0), (0.0, 0.1), (0.1, 0.1), (0.1, 0.0))
    segs, _ = flatten_item_to_segments(item, curve_spacing=10.0)
    assert len(segs) >= 2


def test_flatten_item_to_segments_rect_is_closed_4_edges():
    item = ("re", (0.0, 0.0, 4.0, 2.0))
    segs, vertices = flatten_item_to_segments(item)
    assert len(segs) == 4
    corners = {p for seg in segs for p in seg}
    assert corners == {(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)}
    assert segs[-1][1] == segs[0][0]
    assert vertices == corners


def test_flatten_item_to_segments_quad_is_closed_4_edges_in_order():
    pts = ((0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0))
    item = ("qu", pts)
    segs, vertices = flatten_item_to_segments(item)
    assert len(segs) == 4
    assert segs[0] == (pts[0], pts[1])
    assert segs[-1] == (pts[3], pts[0])
    assert vertices == set(pts)


def test_split_at_intersections_x_crossing_splits_into_4():
    segs = [((0.0, 0.0), (10.0, 10.0)), ((0.0, 10.0), (10.0, 0.0))]
    split = split_at_intersections(segs)
    assert len(split) == 4
    all_points = {p for seg in split for p in seg}
    assert (5.0, 5.0) in {(round(x, 5), round(y, 5)) for x, y in all_points}


def test_split_at_intersections_t_touch_no_duplicate_point():
    # One segment's endpoint touches the interior of another: the
    # horizontal segment gets split in two at (5, 0), the vertical segment
    # is left unchanged since (5, 0) is already one of its own endpoints
    # (no duplicate split point introduced there).
    segs = [((0.0, 0.0), (10.0, 0.0)), ((5.0, -5.0), (5.0, 0.0))]
    split = split_at_intersections(segs)
    assert len(split) == 3
    assert ((5.0, -5.0), (5.0, 0.0)) in split
    assert ((0.0, 0.0), (5.0, 0.0)) in split
    assert ((5.0, 0.0), (10.0, 0.0)) in split


def test_merge_close_points_collapses_near_duplicate_endpoints():
    segs = [((0.0, 0.0), (5.0, 5.0)), ((5.001, 5.0005), (10.0, 0.0))]
    nodes, edges, point_to_node = merge_close_points(segs, point_merge_tol=0.01)
    assert len(nodes) == 3
    assert len(edges) == 2
    # both edges reference the same merged node
    endpoints_used = {n for edge in edges for n in edge}
    shared = [n for n in endpoints_used if sum(1 for e in edges if n in e) == 2]
    assert len(shared) == 1
    # the raw-point -> node-index map resolves both near-duplicate points
    # to that same shared node.
    shared_node = shared[0]
    assert point_to_node[(5.0, 5.0)] == shared_node
    assert point_to_node[(5.001, 5.0005)] == shared_node


def test_merge_close_points_default_tolerance_is_small():
    # Points that are visually close but not true floating-point
    # duplicates must NOT collapse under the new small default tolerance
    # (dedup-only now, not visual consolidation).
    segs = [((0.0, 0.0), (5.0, 5.0)), ((5.1, 5.05), (10.0, 0.0))]
    nodes, edges, _ = merge_close_points(segs)
    assert len(nodes) == 4
