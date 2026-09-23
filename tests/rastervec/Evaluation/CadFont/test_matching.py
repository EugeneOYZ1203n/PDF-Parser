from __future__ import annotations

from rastervec.Evaluation.CadFont.character_bank import CharacterTemplate
from rastervec.Evaluation.CadFont.graph import CharGraph
from rastervec.Evaluation.CadFont.matching import (
    build_candidate_graph,
    enumerate_anchor_correspondences,
    fit_similarity_transform,
    map_template_nodes_to_candidate,
    match_template_against_candidate,
    score_match,
)


# ---------------------------------------------------------------------------
# fit_similarity_transform
# ---------------------------------------------------------------------------


def test_fit_similarity_transform_recovers_known_transform():
    # scale=2, rotation=90deg, translation=(100, 50) applied to a right
    # triangle, then fit back out from the (src, dst) pairs.
    src = [(0.0, 0.0), (10.0, 0.0), (0.0, 5.0)]
    dst = [(100.0, 50.0), (100.0, 70.0), (90.0, 50.0)]

    transform = fit_similarity_transform(src, dst)

    assert transform is not None
    assert abs(transform.scale - 2.0) < 1e-9
    assert abs(transform.rotation_deg - 90.0) < 1e-6
    assert abs(transform.translation[0] - 100.0) < 1e-6
    assert abs(transform.translation[1] - 50.0) < 1e-6


def test_fit_similarity_transform_works_for_two_points():
    src = [(0.0, 0.0), (1.0, 0.0)]
    dst = [(100.0, 50.0), (100.0, 52.0)]  # same scale=2/rotation=90/translation=(100,50)

    transform = fit_similarity_transform(src, dst)

    assert transform is not None
    recovered = transform.apply_many(src)
    for (rx, ry), (dx, dy) in zip(recovered, dst):
        assert abs(rx - dx) < 1e-6 and abs(ry - dy) < 1e-6


def test_fit_similarity_transform_returns_none_for_coincident_source_points():
    src = [(5.0, 5.0), (5.0, 5.0), (5.0, 5.0)]
    dst = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]

    assert fit_similarity_transform(src, dst) is None


# ---------------------------------------------------------------------------
# build_candidate_graph
# ---------------------------------------------------------------------------


def _l_shape_vectors(vector):
    # An "L": horizontal + vertical stroke meeting at (100, 100), placed
    # well away from the origin so a stray baseline-relative shift (this
    # function must NOT apply one) would be obvious.
    horiz = vector(items=[("l", (100.0, 100.0), (110.0, 100.0))], seqno=0, bbox=(100.0, 100.0, 110.0, 100.0))
    vert = vector(items=[("l", (100.0, 100.0), (100.0, 105.0))], seqno=1, bbox=(100.0, 100.0, 100.0, 105.0))
    return [horiz, vert]


def test_build_candidate_graph_stays_in_page_space(vector):
    vectors = _l_shape_vectors(vector)

    g = build_candidate_graph(vectors)

    assert g.num_nodes() == 3
    assert g.num_edges() == 2
    assert set(g.nodes) == {(100.0, 100.0), (110.0, 100.0), (100.0, 105.0)}
    junction_idx = g.nodes.index((100.0, 100.0))
    assert g.degree(junction_idx) == 2


# ---------------------------------------------------------------------------
# enumerate_anchor_correspondences
# ---------------------------------------------------------------------------


def test_enumerate_anchor_correspondences_ordered_and_degree_filtered():
    # Template anchors have degrees [3, 1, 1].
    template = CharGraph(
        nodes=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)],
        edges=[(0, 1), (0, 2), (0, 3)],
    )
    anchor_indices = (0, 1, 2)
    assert [template.degree(i) for i in anchor_indices] == [3, 1, 1]

    # Candidate: node 0 has degree 3 (the only one meeting slot 0's
    # requirement); nodes 1-3 have degree 1; node 4 is isolated (degree 0,
    # excluded from every slot).
    candidate = CharGraph(
        nodes=[(10.0, 10.0), (11.0, 10.0), (12.0, 10.0), (13.0, 10.0), (14.0, 10.0)],
        edges=[(0, 1), (0, 2), (0, 3)],
    )
    assert candidate.degrees() == [3, 1, 1, 1, 0]

    correspondences = enumerate_anchor_correspondences(template, anchor_indices, candidate)

    expected = {
        (0, 1, 2), (0, 1, 3), (0, 2, 1), (0, 2, 3), (0, 3, 1), (0, 3, 2),
    }
    assert set(correspondences) == expected
    assert len(correspondences) == len(set(correspondences))  # no duplicates
    assert all(c[0] == 0 for c in correspondences)  # slot 0 only ever node 0
    assert all(4 not in c for c in correspondences)  # isolated node never eligible


# ---------------------------------------------------------------------------
# map_template_nodes_to_candidate
# ---------------------------------------------------------------------------


def test_map_template_nodes_to_candidate_matches_non_critical_node():
    # A single critical template point that, once transformed, lands
    # exactly on a candidate node that is NOT itself critical (a
    # pass-through, degree-2, non-original-vertex node) -- this must still
    # be a valid match.
    template = CharGraph(
        nodes=[(5.0, 0.0)], edges=[], original_vertex_indices=frozenset({0}),
    )
    candidate = CharGraph(
        nodes=[(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)],
        edges=[(0, 1), (1, 2)],
        original_vertex_indices=frozenset({0, 2}),  # node 1 is NOT an original vertex
    )
    from rastervec.Evaluation.CadFont.graph import critical_point_indices
    assert critical_point_indices(candidate) == frozenset({0, 2})  # node 1 excluded

    from rastervec.Evaluation.CadFont.matching import SimilarityTransform
    identity = SimilarityTransform(scale=1.0, rotation_deg=0.0, translation=(0.0, 0.0))

    node_map = map_template_nodes_to_candidate(template, identity, candidate)

    assert node_map[0] == 1
    assert 1 not in critical_point_indices(candidate)


# ---------------------------------------------------------------------------
# score_match
# ---------------------------------------------------------------------------


def _l_template_graph():
    return CharGraph(
        nodes=[(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)],
        edges=[(0, 1), (0, 2)],
        original_vertex_indices=frozenset({0, 1, 2}),
    )


def test_score_match_near_zero_for_exact_embedding():
    from rastervec.Evaluation.CadFont.graph import critical_point_indices
    from rastervec.Evaluation.CadFont.matching import SimilarityTransform

    template = _l_template_graph()
    critical = critical_point_indices(template) | template.synthetic_anchor_indices
    identity = SimilarityTransform(scale=1.0, rotation_deg=0.0, translation=(0.0, 0.0))
    candidate = CharGraph(nodes=list(template.nodes), edges=list(template.edges))
    node_map = map_template_nodes_to_candidate(template, identity, candidate)

    score = score_match(template, critical, identity, candidate, node_map)

    assert score.mse_term < 1e-12
    assert score.edge_presence_cost == 0.0
    assert score.degree_cost == 0.0
    assert score.total < 1e-12


def test_score_match_worsens_with_perturbed_node():
    from rastervec.Evaluation.CadFont.graph import critical_point_indices
    from rastervec.Evaluation.CadFont.matching import SimilarityTransform

    template = _l_template_graph()
    critical = critical_point_indices(template) | template.synthetic_anchor_indices
    identity = SimilarityTransform(scale=1.0, rotation_deg=0.0, translation=(0.0, 0.0))

    exact = CharGraph(nodes=list(template.nodes), edges=list(template.edges))
    perturbed = CharGraph(nodes=[(0.0, 0.0), (10.1, 0.0), (0.0, 10.0)], edges=list(template.edges))

    exact_score = score_match(
        template, critical, identity, exact,
        map_template_nodes_to_candidate(template, identity, exact),
    )
    perturbed_score = score_match(
        template, critical, identity, perturbed,
        map_template_nodes_to_candidate(template, identity, perturbed),
    )

    assert perturbed_score.mse_term > exact_score.mse_term
    assert perturbed_score.total > exact_score.total


def test_score_match_worsens_with_missing_edge():
    from rastervec.Evaluation.CadFont.graph import critical_point_indices
    from rastervec.Evaluation.CadFont.matching import SimilarityTransform

    template = _l_template_graph()
    critical = critical_point_indices(template) | template.synthetic_anchor_indices
    identity = SimilarityTransform(scale=1.0, rotation_deg=0.0, translation=(0.0, 0.0))

    exact = CharGraph(nodes=list(template.nodes), edges=list(template.edges))
    dropped_edge = CharGraph(nodes=list(template.nodes), edges=[(0, 1)])  # missing (0, 2)

    exact_score = score_match(
        template, critical, identity, exact,
        map_template_nodes_to_candidate(template, identity, exact),
    )
    dropped_score = score_match(
        template, critical, identity, dropped_edge,
        map_template_nodes_to_candidate(template, identity, dropped_edge),
    )

    assert dropped_score.edge_presence_cost > exact_score.edge_presence_cost
    assert dropped_score.total > exact_score.total


# ---------------------------------------------------------------------------
# match_template_against_candidate (end-to-end)
# ---------------------------------------------------------------------------


def test_match_template_against_candidate_finds_best_at_translated_copy():
    # A "T"-ish template: node 1 is the degree-3 junction, nodes 0/2/3 are
    # degree-1 leaves.
    template_graph = CharGraph(
        nodes=[(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (5.0, 5.0)],
        edges=[(0, 1), (1, 2), (1, 3)],
        original_vertex_indices=frozenset({0, 1, 2, 3}),
    )
    assert template_graph.degrees() == [1, 3, 1, 1]
    template = CharacterTemplate(
        label_id="t1", text="T", baseline_id="b1",
        graph=template_graph, complexity=1.0, anchor_node_indices=(1, 0, 3),
    )

    # Candidate: the exact same shape translated by (100, 100).
    candidate_graph = CharGraph(
        nodes=[(100.0, 100.0), (105.0, 100.0), (110.0, 100.0), (105.0, 105.0)],
        edges=[(0, 1), (1, 2), (1, 3)],
    )

    matches = match_template_against_candidate(template, candidate_graph)

    assert len(matches) == 6  # 1 (slot0) * 3 (slot1) * 2 (slot2 remaining)
    best = min(matches, key=lambda m: m.score.total)
    assert best.score.total < 1e-6
    assert abs(best.transform.scale - 1.0) < 1e-6
    assert abs(best.transform.rotation_deg) < 1e-6
    assert abs(best.transform.translation[0] - 100.0) < 1e-6
    assert abs(best.transform.translation[1] - 100.0) < 1e-6
