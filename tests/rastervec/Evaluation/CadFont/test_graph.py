from __future__ import annotations

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
    segments = [((-5.0, 0.0), (5.0, 0.0)), ((0.0, -5.0), (0.0, 5.0))]
    g = build_char_graph(segments)

    assert g.num_nodes() == 5
    assert g.num_edges() == 4
    assert sorted(g.degrees()) == [1, 1, 1, 1, 4]
    assert g.max_degree() == 4
