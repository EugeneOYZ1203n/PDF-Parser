from __future__ import annotations

from rastervec.Evaluation.CadFont.geometry import (
    connect_nearby_points,
    cubic_bezier_points,
    dedupe_exact_points,
    flatten_item_to_segments,
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
    segs, vertices = flatten_item_to_segments(item)
    assert segs == [((0.0, 0.0), (1.0, 1.0))]
    assert vertices == {(0.0, 0.0), (1.0, 1.0)}


def test_flatten_item_to_segments_curve_default_is_5_points_4_segments():
    item = ("c", (0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0))
    segs, vertices = flatten_item_to_segments(item)
    # 5 points (2 endpoints + 3 generated interior) -> 4 chained segments.
    assert len(segs) == 4
    assert segs[0][0] == (0.0, 0.0)
    assert segs[-1][1] == (10.0, 0.0)
    for a, b in zip(segs, segs[1:]):
        assert a[1] == b[0]
    # Only the true endpoints are original vertices, never an interior
    # sample point.
    assert vertices == {(0.0, 0.0), (10.0, 0.0)}


def test_flatten_item_to_segments_curve_bezier_sample_count_is_configurable():
    item = ("c", (0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0))
    segs, _ = flatten_item_to_segments(item, bezier_sample_count=9)
    assert len(segs) == 8


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


def test_dedupe_exact_points_collapses_only_bit_identical_points():
    # One shared item polyline (consecutive segments sharing an exact
    # coordinate) plus a second, near-but-not-identical point from a
    # different item -- only the literal duplicate collapses.
    segs = [
        ((0.0, 0.0), (5.0, 5.0)),
        ((5.0, 5.0), (10.0, 0.0)),
        ((5.001, 5.0005), (20.0, 20.0)),
    ]
    nodes, edges, point_to_node = dedupe_exact_points(segs)
    assert len(nodes) == 5  # (0,0) (5,5) (10,0) (5.001,5.0005) (20,20)
    assert point_to_node[(5.0, 5.0)] != point_to_node[(5.001, 5.0005)]
    assert len(edges) == 3


def test_dedupe_exact_points_no_self_loop_for_degenerate_segment():
    segs = [((1.0, 1.0), (1.0, 1.0)), ((1.0, 1.0), (2.0, 2.0))]
    nodes, edges, _ = dedupe_exact_points(segs)
    assert len(nodes) == 2
    assert len(edges) == 1


def test_connect_nearby_points_adds_edge_for_close_unconnected_pair():
    nodes = [(0.0, 0.0), (10.0, 10.0), (10.001, 10.0005)]
    edges: list[tuple[int, int]] = []
    out = connect_nearby_points(nodes, edges, epsilon=0.01)
    assert (1, 2) in out or (2, 1) in out
    assert len(out) == 1


def test_connect_nearby_points_ignores_pairs_beyond_epsilon():
    nodes = [(0.0, 0.0), (10.0, 10.0)]
    out = connect_nearby_points(nodes, [], epsilon=0.01)
    assert out == []


def test_connect_nearby_points_never_duplicates_an_existing_edge():
    nodes = [(0.0, 0.0), (0.001, 0.0)]
    edges = [(0, 1)]
    out = connect_nearby_points(nodes, edges, epsilon=1.0)
    assert out == [(0, 1)]


def test_connect_nearby_points_never_touches_positions():
    nodes = [(0.0, 0.0), (0.001, 0.0)]
    original = list(nodes)
    connect_nearby_points(nodes, [], epsilon=1.0)
    assert nodes == original
