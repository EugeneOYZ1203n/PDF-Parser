from __future__ import annotations

from rastervec.Evaluation.CadFont.geometry import flatten_item_to_segments
from rastervec.Evaluation.CadFont.graph import (
    CharGraph,
    build_char_graph,
    complexity,
    select_anchor_points,
)


def _h_shape_graph() -> CharGraph:
    # Two verticals (0-4, 4-1 and 2-5, 5-3) joined by one horizontal
    # crossbar (4-5) at their midpoints -- an "H".
    nodes = [
        (0.0, 0.0), (0.0, 10.0),   # 0: left-bottom, 1: left-top
        (10.0, 0.0), (10.0, 10.0),  # 2: right-bottom, 3: right-top
        (0.0, 5.0), (10.0, 5.0),    # 4: left-mid junction, 5: right-mid junction
    ]
    edges = [(0, 4), (4, 1), (2, 5), (5, 3), (4, 5)]
    return CharGraph(nodes=nodes, edges=edges)


def test_h_shape_degrees_and_max_degree():
    g = _h_shape_graph()
    assert g.degrees() == [1, 1, 1, 1, 3, 3]
    assert g.max_degree() == 3
    assert g.num_nodes() == 6
    assert g.num_edges() == 5


def test_h_shape_complexity_is_nodes_times_edges_times_max_degree():
    g = _h_shape_graph()
    assert complexity(g) == 6 * 5 * 3 == 90


def test_select_anchor_points_worked_example_matches_spec():
    # degrees [4, 2, 2, 1, 1] -- node0 highest degree; node2 sits collinear
    # with node0/node1 (all three on the x-axis) so it must be rejected as
    # the 3rd anchor in favor of node3 (off-axis), reproducing the spec's
    # own worked example: "1st, 2nd and 4th points" selected as anchors.
    nodes = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (5.0, 5.0), (3.0, 0.0)]
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2)]
    g = CharGraph(nodes=nodes, edges=edges)
    assert g.degrees() == [4, 2, 2, 1, 1]

    anchors = select_anchor_points(g)

    assert anchors == (0, 1, 3)


def test_select_anchor_points_two_node_graph_returns_two_anchors():
    g = CharGraph(nodes=[(0.0, 0.0), (1.0, 1.0)], edges=[(0, 1)])
    assert select_anchor_points(g) == (0, 1)


def test_select_anchor_points_fully_collinear_graph_caps_at_two():
    nodes = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    edges = [(0, 1), (1, 2), (2, 3)]
    g = CharGraph(nodes=nodes, edges=edges)
    anchors = select_anchor_points(g)
    assert len(anchors) == 2


def test_build_char_graph_plus_sign_end_to_end():
    # Both segments are "l"-equivalent -- all 4 raw endpoints are original
    # vertices, so together with the degree-4 crossing every resulting node
    # is protected and nothing is eligible for simplification, regardless
    # of area_tol.
    segments = [((-5.0, 0.0), (5.0, 0.0)), ((0.0, -5.0), (0.0, 5.0))]
    original_vertices = {(-5.0, 0.0), (5.0, 0.0), (0.0, -5.0), (0.0, 5.0)}
    g = build_char_graph(segments, original_vertices, area_tol=1.0)

    assert g.num_nodes() == 5
    assert g.num_edges() == 4
    assert sorted(g.degrees()) == [1, 1, 1, 1, 4]
    assert g.max_degree() == 4


def test_build_char_graph_simplifies_nearly_straight_curve_to_its_endpoints():
    # A slightly wiggly polyline between two original endpoints, with no
    # real junction anywhere -- every interior point should collapse away
    # under a generous area_tol, leaving just the two protected endpoints.
    points = [(0.0, 0.0), (1.0, 0.001), (2.0, -0.001), (3.0, 0.002), (4.0, 0.0)]
    segments = list(zip(points, points[1:]))
    original_vertices = {points[0], points[-1]}

    g = build_char_graph(segments, original_vertices, area_tol=0.5)

    assert g.num_nodes() == 2
    assert g.num_edges() == 1
    assert set(g.nodes) == {(0.0, 0.0), (4.0, 0.0)}


def test_build_char_graph_preserves_junction_while_simplifying_its_arms():
    # A near-straight horizontal run from (0,0) to (10,0) with a vertical
    # branch grafted on at (5,0) -- a real degree-3 junction. Both
    # horizontal arms have small wiggle that should simplify away, but the
    # junction itself (and the trivial 2-point vertical arm) must survive
    # untouched.
    horizontal = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0001), (5.0, 0.0),
                  (6.0, -0.0001), (8.0, 0.0), (10.0, 0.0)]
    segments = list(zip(horizontal, horizontal[1:]))
    segments.append(((5.0, 0.0), (5.0, 5.0)))
    original_vertices = {(0.0, 0.0), (10.0, 0.0), (5.0, 0.0), (5.0, 5.0)}

    g = build_char_graph(segments, original_vertices, area_tol=0.01)

    assert g.num_nodes() == 4
    assert set(g.nodes) == {(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (5.0, 5.0)}
    assert g.num_edges() == 3
    assert g.max_degree() == 3
    junction_idx = g.nodes.index((5.0, 0.0))
    assert g.degree(junction_idx) == 3


def test_build_char_graph_closed_loop_curve_with_no_junction_stays_a_single_cycle():
    # A single "c" item whose start and end coincide (p0 == p3) -- a pure
    # cycle with no real junction anywhere. Its own single start/end point
    # is still an original vertex, so it anchors the simplification walk
    # without any special-case loop handling.
    item = ("c", (0.0, 0.0), (10.0, 10.0), (-10.0, 10.0), (0.0, 0.0))
    segments, original_vertices = flatten_item_to_segments(item, curve_spacing=1.0)
    assert original_vertices == {(0.0, 0.0)}

    g = build_char_graph(segments, original_vertices, area_tol=1.0)

    assert (0.0, 0.0) in g.nodes
    assert g.num_edges() == g.num_nodes()  # a single closed cycle, no self-loop
    assert g.max_degree() == 2
    assert 3 <= g.num_nodes() < len(segments)  # simplified, but still a real polygon
