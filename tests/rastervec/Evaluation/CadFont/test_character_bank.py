from __future__ import annotations

import math

from rastervec.commons.helpers.geometry import item_points
from rastervec.Evaluation.CadFont.character_bank import (
    build_character_bank,
    to_baseline_relative_vectors,
)
from rastervec.Evaluation.Labelling.label_schema import (
    Baseline,
    LabelEntry,
    LabelSet,
    vector_signatures_for,
)


def test_to_baseline_relative_vectors_maps_origin_to_zero_zero(vector):
    theta = math.radians(30.0)
    origin = (100.0, 50.0)
    direction = (math.cos(theta), math.sin(theta))
    p_a = origin
    p_b = (origin[0] + 10 * direction[0], origin[1] + 10 * direction[1])
    v = vector(items=[("l", p_a, p_b)], bbox=(80.0, 40.0, 120.0, 60.0))

    out = to_baseline_relative_vectors([v], origin, direction)

    assert len(out) == 1
    (x0, y0), (x1, y1) = item_points(out[0].items[0])
    assert abs(x0) < 1e-6 and abs(y0) < 1e-6
    assert abs(x1 - 10.0) < 1e-6 and abs(y1) < 1e-6


def test_to_baseline_relative_vectors_leftmost_point_shifts_to_zero(vector):
    origin = (10.0, 10.0)
    direction = (1.0, 0.0)
    p_a = (13.0, 10.0)  # +3 along the baseline from origin
    p_b = (8.0, 10.0)   # -2 along the baseline from origin -- leftmost
    v = vector(items=[("l", p_a, p_b)], bbox=(8.0, 10.0, 13.0, 10.0))

    out = to_baseline_relative_vectors([v], origin, direction)

    (x0, y0), (x1, y1) = item_points(out[0].items[0])
    assert min(x0, x1) == 0.0
    assert abs(max(x0, x1) - min(x0, x1) - 5.0) < 1e-9
    assert abs(y0) < 1e-9 and abs(y1) < 1e-9


def _t_shape_vectors(vector):
    # A "T": a horizontal stroke with a vertical stroke meeting its
    # interior at (2, 0) -- 4 nodes, 3 edges, max degree 3 once built.
    top = vector(items=[("l", (0.0, 0.0), (5.0, 0.0))], seqno=0, bbox=(0.0, 0.0, 5.0, 0.0))
    stem = vector(items=[("l", (2.0, 0.0), (2.0, 5.0))], seqno=1, bbox=(2.0, 0.0, 2.0, 5.0))
    return [top, stem]


def _i_shape_vector(vector):
    # A single stroke -- 2 nodes, 1 edge, max degree 1.
    return vector(items=[("l", (20.0, 0.0), (20.0, 5.0))], seqno=2, bbox=(20.0, 0.0, 20.0, 5.0))


def test_build_character_bank_sorts_by_complexity_descending(vector):
    t_vectors = _t_shape_vectors(vector)
    i_vector = _i_shape_vector(vector)
    baseline = Baseline(page_index=0, baseline_id="b1", origin=(0.0, 0.0), direction=(1.0, 0.0))

    entry_t = LabelEntry(
        page_index=0, cluster_bbox=(0.0, 0.0, 5.0, 5.0), cluster_signature="s1",
        label_id="labelT", text="T", source="cad_font",
        vector_signatures=vector_signatures_for(t_vectors), baseline_id="b1",
    )
    entry_i = LabelEntry(
        page_index=0, cluster_bbox=(20.0, 0.0, 20.0, 5.0), cluster_signature="s2",
        label_id="labelI", text="I", source="cad_font",
        vector_signatures=vector_signatures_for([i_vector]), baseline_id="b1",
    )
    label_set = LabelSet(pdf_path="x.pdf", entries=[entry_t, entry_i], baselines=[baseline])
    vectors_by_page = {0: [*t_vectors, i_vector]}

    templates = build_character_bank(label_set, vectors_by_page)

    assert [t.text for t in templates] == ["T", "I"]
    assert templates[0].complexity > templates[1].complexity
    assert templates[0].baseline_id == "b1"
    assert len(templates[0].anchor_node_indices) >= 2


def test_build_character_bank_skips_entry_with_no_baseline(vector):
    i_vector = _i_shape_vector(vector)
    entry = LabelEntry(
        page_index=0, cluster_bbox=(20.0, 0.0, 20.0, 5.0), cluster_signature="s",
        label_id="labelI", text="I", source="cad_font",
        vector_signatures=vector_signatures_for([i_vector]), baseline_id=None,
    )
    label_set = LabelSet(pdf_path="x.pdf", entries=[entry], baselines=[])
    templates = build_character_bank(label_set, {0: [i_vector]})
    assert templates == []


def test_build_character_bank_skips_entry_with_unresolved_signature(vector):
    baseline = Baseline(page_index=0, baseline_id="b1", origin=(0.0, 0.0), direction=(1.0, 0.0))
    entry = LabelEntry(
        page_index=0, cluster_bbox=(20.0, 0.0, 20.0, 5.0), cluster_signature="s",
        label_id="labelI", text="I", source="cad_font",
        vector_signatures=["nonexistent-signature"], baseline_id="b1",
    )
    label_set = LabelSet(pdf_path="x.pdf", entries=[entry], baselines=[baseline])
    templates = build_character_bank(label_set, {0: []})
    assert templates == []
