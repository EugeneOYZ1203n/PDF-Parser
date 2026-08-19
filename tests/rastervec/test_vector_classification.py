from __future__ import annotations

import json

import pytest

from rastervec.helpers.clustering import Clustering
from rastervec.models import VectorPath
from rastervec.vector_classification import (
    DEFAULT_PIPELINE,
    PAIRWISE_METRICS,
    SCALAR_METRICS,
    MetricContext,
    StepConfig,
    from_json,
    run_step,
    to_json,
)


class _Meta:
    width = 200
    height = 200


class _Page:
    meta = _Meta()


def _make_path(*, seq=0, kind="re", bbox=(0, 0, 1, 1)) -> VectorPath:
    return VectorPath(
        seq=seq, item_index=0, kind=kind, fill_rule="", points=[(bbox[0], bbox[1]), (bbox[2], bbox[3])],
        bbox=bbox, stroke_color=None, fill_color=None, stroke_opacity=None, fill_opacity=None,
        stroke_width=None, dashes=None, closed=None, layer=None, page_index=0,
    )


def _ctx(**kwargs) -> MetricContext:
    return MetricContext(page=_Page(), item_counts=kwargs.get("item_counts", {}), item_bboxes=kwargs.get("item_bboxes", {}))


# ------------------------------------------------------------------
# Registry completeness
# ------------------------------------------------------------------

def test_scalar_metrics_cover_every_expected_key():
    expected = {
        "bbox_area_fraction", "dimension_fraction", "aspect_ratio", "item_path_count",
        "item_bbox_min_side", "item_bbox_max_side", "layout_panel_singleton", "seq",
    }
    assert set(SCALAR_METRICS.keys()) == expected


def test_pairwise_metrics_cover_every_expected_key():
    assert set(PAIRWISE_METRICS.keys()) == {"spatial_gap", "dimension_similarity", "overlap_tolerance"}


def test_default_pipeline_steps_all_validate():
    for step in DEFAULT_PIPELINE:
        step.validate()


# ------------------------------------------------------------------
# Filter: scope combinations
# ------------------------------------------------------------------

def test_filter_scope_item_drops_flat_across_groups():
    big = _make_path(bbox=(0, 0, 190, 190))  # 90% of 200x200 page
    small = _make_path(bbox=(10, 10, 20, 20))
    step = StepConfig(kind="filter", metric="bbox_area_fraction", condition="<=", threshold=0.2, scope="item")

    kept, dropped = run_step(step, [[big, small]], _ctx(), Clustering())

    kept_paths = [p for g in kept for p in g]
    dropped_paths = [p for g in dropped for p in g]
    assert small in kept_paths
    assert big in dropped_paths


def test_filter_scope_cluster_all_or_nothing():
    big_group = [_make_path(bbox=(0, 0, 190, 190))]
    small_group = [_make_path(bbox=(10, 10, 20, 20))]
    step = StepConfig(kind="filter", metric="bbox_area_fraction", condition="<=", threshold=0.2, scope="cluster")

    kept, dropped = run_step(step, [big_group, small_group], _ctx(), Clustering())

    assert small_group in kept
    assert big_group in dropped


def test_filter_scope_items_in_cluster_preserves_group_structure():
    # A path-level filter that would flatten (scope="item") could still
    # merge this group's survivors with another group's -- "items_in_cluster"
    # must never flatten across groups, only within each group's own list.
    a = _make_path(seq=0, bbox=(0, 0, 190, 190))
    b = _make_path(seq=1, bbox=(10, 10, 20, 20))
    step = StepConfig(kind="filter", metric="bbox_area_fraction", condition="<=", threshold=0.2, scope="items_in_cluster")

    kept, dropped = run_step(step, [[a, b]], _ctx(), Clustering())

    assert kept == [[b]]
    assert dropped == [[a]]


def test_filter_scope_cluster_all_items_requires_every_member_to_pass():
    # Reproduces filter_item_path_count/filter_item_bbox's "every seq in
    # the group must satisfy the bound" semantics.
    passing_group = [_make_path(seq=0, bbox=(0, 0, 5, 5)), _make_path(seq=1, bbox=(0, 0, 5, 5))]
    failing_group = [_make_path(seq=2, bbox=(0, 0, 5, 5)), _make_path(seq=3, bbox=(0, 0, 190, 190))]
    step = StepConfig(kind="filter", metric="bbox_area_fraction", condition="<=", threshold=0.2, scope="cluster_all_items")

    kept, dropped = run_step(step, [passing_group, failing_group], _ctx(), Clustering())

    assert passing_group in kept
    assert failing_group in dropped


# ------------------------------------------------------------------
# Filter: outlier / relative-aggregate metrics
# ------------------------------------------------------------------

def test_relative_aggregate_metric_global_vs_group_population():
    # Three groups: two "normal" small-area groups and one big outlier.
    normal1 = [_make_path(bbox=(0, 0, 10, 10))]
    normal2 = [_make_path(bbox=(0, 0, 10, 10))]
    outlier = [_make_path(bbox=(0, 0, 100, 100))]
    groups = [normal1, normal2, outlier]

    # Drop clusters whose area is > 2x the mean area *of all clusters*.
    step_global = StepConfig(
        kind="filter", metric="bbox_area_fraction", condition="<=", threshold=2.0,
        scope="cluster", aggregate="mean", aggregate_scope="global",
    )
    kept, dropped = run_step(step_global, groups, _ctx(), Clustering())
    assert outlier in dropped
    assert normal1 in kept and normal2 in kept


def test_relative_aggregate_items_in_cluster_group_scope_isolates_outlier_per_group():
    # Group A: one normal-sized item plus one outlier relative to A's own
    # mean; Group B: only normal items. aggregate_scope="group" means the
    # outlier check for A's members never involves B's items.
    a1 = _make_path(seq=0, bbox=(0, 0, 10, 10))
    a2 = _make_path(seq=1, bbox=(0, 0, 40, 40))  # outlier relative to group A's own mean
    b1 = _make_path(seq=2, bbox=(0, 0, 10, 10))
    b2 = _make_path(seq=3, bbox=(0, 0, 12, 12))
    step = StepConfig(
        kind="filter", metric="bbox_area_fraction", condition="<=", threshold=1.5,
        scope="items_in_cluster", aggregate="mean", aggregate_scope="group",
    )

    kept, dropped = run_step(step, [[a1, a2], [b1, b2]], _ctx(), Clustering())

    dropped_paths = [p for g in dropped for p in g]
    kept_paths = [p for g in kept for p in g]
    assert a2 in dropped_paths
    assert a1 in kept_paths and b1 in kept_paths and b2 in kept_paths


# ------------------------------------------------------------------
# Cluster kind
# ------------------------------------------------------------------

def test_cluster_global_spatial_gap_matches_cluster_spatial():
    a = _make_path(bbox=(0, 0, 2, 2))
    b = _make_path(bbox=(20, 0, 22, 2))  # gap = 18px
    step_far = StepConfig(kind="cluster", metric="spatial_gap", method="global", threshold=8.0)
    step_near = StepConfig(kind="cluster", metric="spatial_gap", method="global", threshold=20.0)

    kept_far, _ = run_step(step_far, [[a], [b]], _ctx(), Clustering())
    kept_near, _ = run_step(step_near, [[a], [b]], _ctx(), Clustering())

    assert len(kept_far) == 2
    assert len(kept_near) == 1


def test_cluster_pairwise_within_group_matches_cluster_by_seq():
    a = _make_path(seq=0, bbox=(0, 0, 2, 2))
    b = _make_path(seq=10, bbox=(0, 0, 2, 2))  # seq gap = 10
    step_tight = StepConfig(kind="cluster", metric="seq", method="pairwise", scope="within_group", threshold=3)
    step_loose = StepConfig(kind="cluster", metric="seq", method="pairwise", scope="within_group", threshold=20)

    kept_tight, _ = run_step(step_tight, [[a, b]], _ctx(), Clustering())
    kept_loose, _ = run_step(step_loose, [[a, b]], _ctx(), Clustering())

    assert len(kept_tight) == 2
    assert len(kept_loose) == 1


def test_cluster_pairwise_global_flatten_matches_cluster_by_item_path_count():
    item_counts = {0: 1, 1: 5}
    a = _make_path(seq=0, bbox=(0, 0, 2, 2))
    b = _make_path(seq=1, bbox=(50, 50, 52, 52))
    step = StepConfig(kind="cluster", metric="item_path_count", method="pairwise", scope="global_flatten", threshold=2)

    kept, _ = run_step(step, [[a], [b]], _ctx(item_counts=item_counts), Clustering())

    assert len(kept) == 2  # path counts 1 vs 5, gap=4 > max_gap=2 -> stays split


# ------------------------------------------------------------------
# Group kind
# ------------------------------------------------------------------

def test_group_path_scope_matches_group_overlapping_path():
    a = _make_path(bbox=(0, 0, 10, 10))
    b = _make_path(bbox=(5, 5, 15, 15))  # overlaps a
    c = _make_path(bbox=(0.5, 0.5, 1.5, 1.5))  # fully inside a -- never merges
    step = StepConfig(kind="group", metric="overlap_tolerance", scope="path", threshold=3.0)

    kept, _ = run_step(step, [[a, b, c]], _ctx(), Clustering())

    merged = [g for g in kept if a in g]
    assert len(merged) == 1
    assert sorted(merged[0], key=id) == sorted([a, b], key=id)
    assert any(g == [c] for g in kept)


def test_group_cluster_scope_never_splits_incoming_group():
    g1 = [_make_path(bbox=(0, 0, 5, 5)), _make_path(bbox=(100, 100, 105, 105))]  # far apart internally
    g2 = [_make_path(bbox=(6, 0, 11, 5))]  # close to g1's first member
    step = StepConfig(kind="group", metric="spatial_gap", scope="cluster", threshold=2.0)

    kept, _ = run_step(step, [g1, g2], _ctx(), Clustering())

    # g1 is one atomic unit -- it can only merge with g2 wholesale or not
    # at all, never split apart even though its own members are far apart.
    assert len(kept) == 1
    assert len(kept[0]) == 3


def test_group_dimension_similarity_matches_cluster_groups_by_dimension():
    g1 = [_make_path(bbox=(0, 0, 10, 10))]
    g2 = [_make_path(bbox=(0, 0, 13, 13))]  # 30% bigger
    step_loose = StepConfig(kind="group", metric="dimension_similarity", scope="cluster", threshold=0.35)
    step_tight = StepConfig(kind="group", metric="dimension_similarity", scope="cluster", threshold=0.1)

    kept_loose, _ = run_step(step_loose, [g1, g2], _ctx(), Clustering())
    kept_tight, _ = run_step(step_tight, [g1, g2], _ctx(), Clustering())

    assert len(kept_loose) == 1
    assert len(kept_tight) == 2


def test_group_dimension_similarity_item_bbox_source_matches_cluster_by_item_bbox():
    item_bboxes = {0: (0, 0, 10, 10), 1: (0, 0, 13, 13)}
    a = _make_path(seq=0, bbox=(0, 0, 1, 1))
    b = _make_path(seq=1, bbox=(50, 50, 51, 51))
    step = StepConfig(
        kind="group", metric="dimension_similarity", scope="path", threshold=0.35,
        params={"bbox_source": "item"},
    )

    kept, _ = run_step(step, [[a], [b]], _ctx(item_bboxes=item_bboxes), Clustering())

    assert len(kept) == 1  # merged by their *original item* bbox similarity, despite being far apart


# ------------------------------------------------------------------
# JSON save/load round trip
# ------------------------------------------------------------------

def test_step_config_json_round_trip():
    steps = [
        StepConfig(kind="filter", metric="bbox_area_fraction", condition="within", threshold=[0.0, 0.5], scope="item", label="A"),
        StepConfig(kind="group", metric="dimension_similarity", scope="cluster", threshold=0.2, params={"bbox_source": "item"}),
    ]

    payload = json.dumps(to_json(steps))
    restored = from_json(json.loads(payload))

    assert restored == steps


def test_from_json_rejects_invalid_config():
    with pytest.raises(ValueError):
        from_json([{"kind": "filter", "metric": "not_a_real_metric", "condition": ">", "threshold": 1.0}])


# ------------------------------------------------------------------
# The actual bug-fix regression test: two same-metric step instances at
# different positions must hold independently-effective params.
# ------------------------------------------------------------------

def test_two_same_metric_steps_at_different_positions_hold_independent_thresholds():
    a = _make_path(bbox=(0, 0, 2, 2))
    b = _make_path(bbox=(15, 0, 17, 2))  # gap = 13px
    steps = [
        StepConfig(kind="cluster", metric="spatial_gap", method="global", threshold=5.0, label="tight"),
        StepConfig(kind="cluster", metric="spatial_gap", method="global", threshold=50.0, label="loose"),
    ]
    ctx = _ctx()
    clustering = Clustering()

    groups = [[a], [b]]
    kept_after_step0, _ = run_step(steps[0], groups, ctx, clustering)
    assert len(kept_after_step0) == 2  # threshold 5 < gap 13 -> stays split

    kept_after_step1, _ = run_step(steps[1], kept_after_step0, ctx, clustering)
    assert len(kept_after_step1) == 1  # threshold 50 > gap 13 -> merges

    # Old bug: step_params was keyed by step *name* ("cluster_spatial"), so
    # both occurrences above would have been forced to share one threshold.
    # Here each StepConfig instance carries its own -- confirm they really
    # differ.
    assert steps[0].threshold != steps[1].threshold
