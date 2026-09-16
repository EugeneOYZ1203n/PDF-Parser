from __future__ import annotations

from rastervec.Evaluation.CadFont.geometry import (
    cubic_bezier_points,
    flatten_item_to_segments,
    merge_close_points,
    merge_colinear_segments,
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


def test_flatten_item_to_segments_line():
    item = ("l", (0.0, 0.0), (1.0, 1.0))
    segs = flatten_item_to_segments(item)
    assert segs == [((0.0, 0.0), (1.0, 1.0))]


def test_flatten_item_to_segments_curve_n8_gives_8_chained_segments():
    item = ("c", (0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))
    segs = flatten_item_to_segments(item, curve_samples=8)
    assert len(segs) == 8
    for a, b in zip(segs, segs[1:]):
        assert a[1] == b[0]
    assert segs[0][0] == (0.0, 0.0)
    assert segs[-1][1] == (1.0, 0.0)


def test_flatten_item_to_segments_rect_is_closed_4_edges():
    item = ("re", (0.0, 0.0, 4.0, 2.0))
    segs = flatten_item_to_segments(item)
    assert len(segs) == 4
    corners = {p for seg in segs for p in seg}
    assert corners == {(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)}
    assert segs[-1][1] == segs[0][0]


def test_flatten_item_to_segments_quad_is_closed_4_edges_in_order():
    pts = ((0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0))
    item = ("qu", pts)
    segs = flatten_item_to_segments(item)
    assert len(segs) == 4
    assert segs[0] == (pts[0], pts[1])
    assert segs[-1] == (pts[3], pts[0])


def test_merge_colinear_segments_touching_horizontal_merges():
    segs = [((0.0, 0.0), (5.0, 0.0)), ((5.0, 0.0), (10.0, 0.0))]
    merged = merge_colinear_segments(segs, angle_tol_deg=2.0, perp_tol=0.75, gap_tol=1.0)
    assert len(merged) == 1
    (x0, y0), (x1, y1) = merged[0]
    assert {round(x0), round(x1)} == {0, 10}
    assert y0 == y1 == 0.0


def test_merge_colinear_segments_far_apart_does_not_merge():
    segs = [((0.0, 0.0), (5.0, 0.0)), ((50.0, 0.0), (60.0, 0.0))]
    merged = merge_colinear_segments(segs, gap_tol=1.0)
    assert len(merged) == 2


def test_merge_colinear_segments_perpendicular_never_merges():
    segs = [((0.0, 0.0), (5.0, 0.0)), ((5.0, 0.0), (5.0, 5.0))]
    merged = merge_colinear_segments(segs)
    assert len(merged) == 2


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
    segs = [((0.0, 0.0), (5.0, 5.0)), ((5.1, 5.05), (10.0, 0.0))]
    nodes, edges = merge_close_points(segs, point_merge_tol=0.5)
    assert len(nodes) == 3
    assert len(edges) == 2
    # both edges reference the same merged node
    endpoints_used = {n for edge in edges for n in edge}
    shared = [n for n in endpoints_used if sum(1 for e in edges if n in e) == 2]
    assert len(shared) == 1
