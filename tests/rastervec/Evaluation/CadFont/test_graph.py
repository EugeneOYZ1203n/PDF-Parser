from __future__ import annotations

from rastervec.Evaluation.CadFont.geometry import flatten_item_to_segments
from rastervec.Evaluation.CadFont.graph import (
    CharGraph,
    build_char_graph,
    complexity,
    critical_point_indices,
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
    # degrees [4, 2, 2, 1, 1] -- node0 is the unique max-degree node, so it
    # wins anchor 1 outright. Anchor 2 is then chosen purely by distance
    # from anchor 1 (degree no longer matters): node3 (5,5) is farthest
    # (~7.07) vs node1 (1.0), node2 (2.0), node4 (3.0). Anchor 3 maximizes
    # triangle area with (node0, node3) among the remaining {1, 2, 4}:
    # node4 (3,0) gives area 7.5, beating node2's 5.0 and node1's 2.5.
    # No original_vertex_indices set -> unrestricted (today's) behavior.
    nodes = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (5.0, 5.0), (3.0, 0.0)]
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2)]
    g = CharGraph(nodes=nodes, edges=edges)
    assert g.degrees() == [4, 2, 2, 1, 1]

    _, anchors = select_anchor_points(g)

    assert anchors == (0, 3, 4)


def test_select_anchor_points_anchor1_tiebreak_by_centroid_distance():
    # Two nodes tie for max degree (3 each); node2 is farther from the
    # eligible-set centroid than node1, so it must win anchor 1.
    nodes = [(0.0, 0.0), (0.0, 1.0), (100.0, 1.0), (0.0, -1.0), (1.0, 1.0), (-1.0, -1.0)]
    edges = [(1, 0), (1, 4), (1, 5), (2, 0), (2, 3), (2, 5)]
    g = CharGraph(nodes=nodes, edges=edges)
    assert g.degree(1) == 3
    assert g.degree(2) == 3

    _, anchors = select_anchor_points(g, max_anchors=1)

    assert anchors == (2,)


def test_select_anchor_points_anchor2_ignores_degree_prefers_distance():
    # node0 is the unique max-degree node (3) -> anchor 1. Among the
    # remaining nodes, node1 and node3 both have higher degree (2) than
    # node2 (1), but node2 is by far the farthest from anchor1 -- anchor 2
    # must be chosen by pure distance, ignoring degree entirely.
    nodes = [(0.0, 0.0), (1.0, 0.0), (0.0, 50.0), (2.0, 0.0)]
    edges = [(0, 1), (0, 2), (0, 3), (1, 3)]
    g = CharGraph(nodes=nodes, edges=edges)
    assert g.degree(0) == 3
    assert g.degree(1) == 2 and g.degree(3) == 2
    assert g.degree(2) == 1

    _, anchors = select_anchor_points(g, max_anchors=2)

    assert anchors[0] == 0
    assert anchors[1] == 2


def test_select_anchor_points_anchor3_maximizes_triangle_area():
    # anchor1=node0 (unique max degree 3), anchor2=node1 (farthest from
    # node0, at distance 10 vs ~5.02/~7.07 for nodes 2/3). Among the
    # remaining candidates for anchor 3, node2 sits nearly on the 0-1 line
    # (small triangle area 2.5) while node3 is well off-axis (area 25) --
    # node3 must win even though it has lower degree (1) than node2 (2).
    nodes = [(0.0, 0.0), (10.0, 0.0), (5.0, 0.5), (5.0, 5.0)]
    edges = [(0, 1), (0, 2), (0, 3), (1, 2)]
    g = CharGraph(nodes=nodes, edges=edges)
    assert g.degree(0) == 3
    assert g.degree(2) == 2 and g.degree(3) == 1

    _, anchors = select_anchor_points(g, max_anchors=3)

    assert anchors[:2] == (0, 1)
    assert anchors[2] == 3


def test_critical_point_indices_restricted_mode():
    nodes = [(0.0, 0.0), (10.0, 0.0), (5.0, 0.5), (5.0, 5.0)]
    edges = [(0, 2), (1, 2), (2, 3)]
    g = CharGraph(nodes=nodes, edges=edges, original_vertex_indices=frozenset({0, 1, 3}))
    assert g.degree(2) == 3
    assert critical_point_indices(g) == frozenset({0, 1, 2, 3})


def test_critical_point_indices_unrestricted_mode_is_every_node():
    g = CharGraph(nodes=[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)], edges=[(0, 1), (1, 2)])
    assert critical_point_indices(g) == frozenset({0, 1, 2})


def test_select_anchor_points_two_node_graph_returns_two_anchors():
    g = CharGraph(nodes=[(0.0, 0.0), (1.0, 1.0)], edges=[(0, 1)])
    _, anchors = select_anchor_points(g)
    assert anchors == (0, 1)


def test_select_anchor_points_fully_collinear_graph_caps_at_two():
    nodes = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    edges = [(0, 1), (1, 2), (2, 3)]
    g = CharGraph(nodes=nodes, edges=edges)
    _, anchors = select_anchor_points(g)
    assert len(anchors) == 2


def test_select_anchor_points_excludes_synthetic_survivor_in_favor_of_original_vertex():
    # A chain 0 - 2 - 1 where 0 and 1 are the item's real endpoints
    # (original vertices) and 2 is a higher-degree-looking simplification
    # survivor that is NOT an original vertex and has no real junction
    # (degree 2). With original_vertex_indices set, node 2 must never be
    # picked even though nothing here ranks it above the real vertices by
    # a wide margin -- eligibility excludes it outright.
    nodes = [(0.0, 0.0), (10.0, 0.0), (5.0, 0.5)]
    edges = [(0, 2), (2, 1)]
    g = CharGraph(nodes=nodes, edges=edges, original_vertex_indices=frozenset({0, 1}))

    _, anchors = select_anchor_points(g, max_anchors=2)

    assert 2 not in anchors
    assert set(anchors) == {0, 1}


def test_select_anchor_points_real_junction_not_an_original_vertex_is_eligible():
    # An X-crossing of two otherwise-unrelated segments: node 4 (the
    # crossing) has degree 4 but is NOT itself an original vertex --
    # still eligible per the "original vertex OR degree > 2" rule.
    nodes = [(-5.0, -5.0), (5.0, 5.0), (-5.0, 5.0), (5.0, -5.0), (0.0, 0.0)]
    edges = [(0, 4), (4, 1), (2, 4), (4, 3)]
    g = CharGraph(nodes=nodes, edges=edges, original_vertex_indices=frozenset({0, 1, 2, 3}))

    _, anchors = select_anchor_points(g)

    assert 4 in anchors


def test_select_anchor_points_synthesizes_baseline_anchor_for_straight_stroke():
    # A single straight stroke: exactly 2 eligible (original-vertex)
    # anchors, no junction -- a 3rd, synthetic point must be added on the
    # baseline (y=0), deterministically.
    g = CharGraph(
        nodes=[(0.0, 0.0), (10.0, 0.0)], edges=[(0, 1)],
        original_vertex_indices=frozenset({0, 1}),
    )

    new_graph, anchors = select_anchor_points(g)

    assert len(anchors) == 3
    synthetic_idx = anchors[2]
    assert synthetic_idx == 2
    assert new_graph.synthetic_anchor_indices == frozenset({2})
    assert new_graph.nodes[2] == (-10.0, 0.0)
    # original graph fields carried over unchanged
    assert new_graph.original_vertex_indices == frozenset({0, 1})
    assert new_graph.edges == [(0, 1)]


def test_select_anchor_points_unrestricted_graph_returns_itself_unchanged():
    # No original_vertex_indices -> unrestricted mode -> no synthesis, the
    # returned graph is the same object passed in.
    g = CharGraph(nodes=[(0.0, 0.0), (1.0, 1.0)], edges=[(0, 1)])
    returned_graph, _ = select_anchor_points(g)
    assert returned_graph is g


def test_build_char_graph_plus_sign_end_to_end():
    # Both segments are "l" items -- one vertex_group per item, all 4 raw
    # endpoints are original vertices. Each of the 4 arms is a trivial
    # 2-node chain (leaf -> the degree-4 crossing junction) with no
    # interior point at all, so nothing is eligible for simplification
    # here regardless of epsilon -- not because every node is protected.
    segments = [((-5.0, 0.0), (5.0, 0.0)), ((0.0, -5.0), (0.0, 5.0))]
    vertex_groups = [frozenset({(-5.0, 0.0), (5.0, 0.0)}), frozenset({(0.0, -5.0), (0.0, 5.0)})]
    g, stats = build_char_graph(segments, vertex_groups)

    assert g.num_nodes() == 5
    assert g.num_edges() == 4
    assert sorted(g.degrees()) == [1, 1, 1, 1, 4]
    assert g.max_degree() == 4
    assert g.original_vertex_indices == frozenset(range(4)) or len(g.original_vertex_indices) == 4
    assert stats.points_removed == 0


def test_build_char_graph_simplifies_nearly_straight_curve_to_its_endpoints():
    # A slightly wiggly polyline between two original endpoints (one "c"
    # item's own flatten output), with no real junction anywhere -- the
    # wiggle amplitude (~0.002) is well under half the shortest segment
    # length here (each ~1 unit long -> epsilon ~0.5), so every interior
    # point collapses away, leaving just the two protected endpoints.
    points = [(0.0, 0.0), (1.0, 0.001), (2.0, -0.001), (3.0, 0.002), (4.0, 0.0)]
    segments = list(zip(points, points[1:]))
    vertex_groups = [frozenset({points[0], points[-1]})]

    g, stats = build_char_graph(segments, vertex_groups)

    assert g.num_nodes() == 2
    assert g.num_edges() == 1
    assert set(g.nodes) == {(0.0, 0.0), (4.0, 0.0)}
    assert len(g.original_vertex_indices) == 2
    assert stats.points_removed == 3
    assert stats.points_before_rdp == 5
    assert stats.points_after_rdp == 2


def test_build_char_graph_preserves_junction_while_simplifying_its_arms():
    # A near-straight horizontal run from (0,0) to (10,0) (one item) with a
    # vertical branch (a second item) grafted on at (5,0) -- a real
    # degree-3 junction. Both horizontal arms have small wiggle (~0.0001,
    # well under the ~1-unit-segment-derived epsilon) that should simplify
    # away, but the junction itself (and the trivial 2-point vertical arm)
    # must survive untouched.
    horizontal = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0001), (5.0, 0.0),
                  (6.0, -0.0001), (8.0, 0.0), (10.0, 0.0)]
    segments = list(zip(horizontal, horizontal[1:]))
    segments.append(((5.0, 0.0), (5.0, 5.0)))
    vertex_groups = [
        frozenset({(0.0, 0.0), (10.0, 0.0)}),
        frozenset({(5.0, 0.0), (5.0, 5.0)}),
    ]

    g, _stats = build_char_graph(segments, vertex_groups)

    assert g.num_nodes() == 4
    assert set(g.nodes) == {(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (5.0, 5.0)}
    assert g.num_edges() == 3
    assert g.max_degree() == 3
    junction_idx = g.nodes.index((5.0, 0.0))
    assert g.degree(junction_idx) == 3
    assert len(g.original_vertex_indices) == 4


def test_build_char_graph_closed_loop_curve_with_no_junction_stays_a_single_cycle():
    # A single "c" item whose start and end coincide (p0 == p3) -- a pure
    # cycle with no real junction anywhere. Its own single start/end point
    # is still an original vertex, so it anchors the simplification walk
    # without any special-case loop handling.
    item = ("c", (0.0, 0.0), (10.0, 10.0), (-10.0, 10.0), (0.0, 0.0))
    segments, original_vertices = flatten_item_to_segments(item)
    assert original_vertices == {(0.0, 0.0)}
    vertex_groups = [frozenset(original_vertices)]

    g, _stats = build_char_graph(segments, vertex_groups)

    assert (0.0, 0.0) in g.nodes
    assert g.num_edges() == g.num_nodes()  # a single closed cycle, no self-loop
    assert g.max_degree() == 2
    assert 3 <= g.num_nodes() <= len(segments)  # simplified (or unchanged), but still a real polygon


def test_build_char_graph_excludes_simplification_survivor_from_original_vertex_indices():
    # A sharp bend point (2,3) well off the (0,0)-(4,0) chord -- far beyond
    # half the shortest segment length here, so it survives RDP -- must NOT
    # be counted as an original vertex, even though it's a real node in the
    # final graph.
    points = [(0.0, 0.0), (2.0, 3.0), (4.0, 0.0)]
    segments = list(zip(points, points[1:]))
    vertex_groups = [frozenset({points[0], points[-1]})]

    g, _stats = build_char_graph(segments, vertex_groups)

    assert g.num_nodes() == 3  # the bend point survives (not collinear)
    survivor_idx = g.nodes.index((2.0, 3.0))
    assert survivor_idx not in g.original_vertex_indices
    assert len(g.original_vertex_indices) == 2


def test_compute_epsilon_is_half_the_shortest_segment():
    from rastervec.Evaluation.CadFont.graph import _compute_epsilon

    segments = [((0.0, 0.0), (10.0, 0.0)), ((0.0, 0.0), (0.0, 4.0))]
    assert abs(_compute_epsilon(segments) - 2.0) < 1e-9


def test_rdp_chain_drops_in_tolerance_wiggle():
    from rastervec.Evaluation.CadFont.graph import _rdp_chain

    nodes = [(0.0, 0.0), (5.0, 0.4), (10.0, 0.0)]
    chain = [0, 1, 2]
    assert _rdp_chain(nodes, chain, epsilon=0.5) == [0, 2]


def test_rdp_chain_keeps_out_of_tolerance_bend():
    from rastervec.Evaluation.CadFont.graph import _rdp_chain

    nodes = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]
    chain = [0, 1, 2]
    assert _rdp_chain(nodes, chain, epsilon=0.5) == [0, 1, 2]


def test_build_char_graph_drops_geometrically_redundant_interior_original_vertex():
    # Three collinear "l" items chained end to end: (0,0)-(3,0)-(6,0)-(10,0).
    # All 4 corner points are original vertices (each item's own true
    # endpoints), no real junction anywhere. The two interior points sit
    # exactly on the (0,0)-(10,0) line -- perpendicular distance 0, well
    # under epsilon -- so RDP must drop them despite being original
    # vertices, since the component still keeps its floor of 2 (the two
    # outer endpoints) without them.
    points = [(0.0, 0.0), (3.0, 0.0), (6.0, 0.0), (10.0, 0.0)]
    segments = list(zip(points, points[1:]))
    vertex_groups = [
        frozenset({(0.0, 0.0), (3.0, 0.0)}),
        frozenset({(3.0, 0.0), (6.0, 0.0)}),
        frozenset({(6.0, 0.0), (10.0, 0.0)}),
    ]

    g, stats = build_char_graph(segments, vertex_groups)

    assert g.num_nodes() == 2
    assert set(g.nodes) == {(0.0, 0.0), (10.0, 0.0)}
    assert len(g.original_vertex_indices) == 2
    assert stats.points_removed == 2


def test_build_char_graph_closed_rectangle_with_no_junction_keeps_all_four_corners():
    # A lone rectangle, no other intersecting geometry -> a closed loop
    # with zero real junctions. `_force_protect_junctionless_components`
    # must give `_find_chains` a valid anchor pair (this component's
    # farthest-apart corner pair, a diagonal) rather than leaving it to an
    # arbitrary, processing-order-dependent starting edge. Each corner is a
    # genuine 90-degree turn (well beyond epsilon), so all 4 survive RDP
    # regardless of which pair got force-protected.
    item = ("re", (0.0, 0.0, 10.0, 5.0))
    segments, original_vertices = flatten_item_to_segments(item)
    vertex_groups = [frozenset(original_vertices)]

    g, _stats = build_char_graph(segments, vertex_groups)

    assert g.num_nodes() == 4
    assert g.num_edges() == 4
    assert g.max_degree() == 2
    assert len(g.original_vertex_indices) == 4


def test_rescue_original_vertex_floor_restores_largest_deviation_first():
    from rastervec.Evaluation.CadFont.graph import _rescue_original_vertex_floor

    # One chain, one component: endpoints 0/4 are NOT original vertices
    # (stand-ins for real junctions elsewhere in a bigger graph); nodes
    # 1, 2, 3 are all original vertices that plain RDP already collapsed
    # away entirely (chain_kept starts at just the two endpoints). With
    # nothing else in this component to hold the floor, exactly 2 must be
    # restored -- the largest perpendicular deviation from the (0, 4)
    # baseline first (node 2, deviation 3.0), then whichever is largest
    # against the now-updated local segments (node 3, deviation ~1.115,
    # beating node 1's ~0.686).
    nodes = [(0.0, 0.0), (3.0, 1.0), (5.0, 3.0), (7.0, 0.5), (10.0, 0.0)]
    chain_full = [[0, 1, 2, 3, 4]]
    chain_kept = [[0, 4]]
    original_vertex_node_indices = {1, 2, 3}
    components = [0, 0, 0, 0, 0]

    _rescue_original_vertex_floor(nodes, chain_full, chain_kept, original_vertex_node_indices, components)

    assert chain_kept[0][0] == 0 and chain_kept[0][-1] == 4
    restored = {i for i in chain_kept[0] if i in original_vertex_node_indices}
    assert restored == {2, 3}


def test_rescue_original_vertex_floor_noop_when_component_has_zero_originals():
    from rastervec.Evaluation.CadFont.graph import _rescue_original_vertex_floor

    nodes = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]
    chain_full = [[0, 1, 2]]
    chain_kept = [[0, 2]]
    components = [0, 0, 0]

    _rescue_original_vertex_floor(nodes, chain_full, chain_kept, set(), components)

    assert chain_kept[0] == [0, 2]


def test_rescue_original_vertex_floor_restores_the_single_original_when_only_one_exists():
    from rastervec.Evaluation.CadFont.graph import _rescue_original_vertex_floor

    # A component with only 1 original vertex total: the floor caps at
    # min(2, 1) == 1, not a forced 2 that can't exist -- but that 1 must
    # still be restored, not silently dropped to 0.
    nodes = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)]
    chain_full = [[0, 1, 2]]
    chain_kept = [[0, 2]]
    components = [0, 0, 0]

    _rescue_original_vertex_floor(nodes, chain_full, chain_kept, {1}, components)

    assert chain_kept[0] == [0, 1, 2]


def test_rescue_original_vertex_floor_treats_disconnected_components_independently():
    from rastervec.Evaluation.CadFont.graph import _rescue_original_vertex_floor

    # Two unrelated components, each shaped like the "largest deviation
    # first" test above. The floor is NOT a single global budget of 2
    # shared across the whole graph -- each component must independently
    # reach its own floor of 2.
    nodes = [
        (0.0, 0.0), (3.0, 1.0), (5.0, 3.0), (7.0, 0.5), (10.0, 0.0),
        (100.0, 0.0), (103.0, 1.0), (105.0, 3.0), (107.0, 0.5), (110.0, 0.0),
    ]
    chain_full = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    chain_kept = [[0, 4], [5, 9]]
    original_vertex_node_indices = {1, 2, 3, 6, 7, 8}
    components = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    _rescue_original_vertex_floor(nodes, chain_full, chain_kept, original_vertex_node_indices, components)

    assert len([i for i in chain_kept[0] if i in original_vertex_node_indices]) == 2
    assert len([i for i in chain_kept[1] if i in original_vertex_node_indices]) == 2
