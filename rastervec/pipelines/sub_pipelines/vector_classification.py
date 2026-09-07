"""Vector Classification sub-pipeline: turn raw vector paths into text
candidates + drawing content.

Read `_classify_bucket` to see the fixed 13-step chain as one named call
per step. `classify_vectors` wraps it with the per-`(layer, color)`-bucket
loop, the whole-page similarity grouping, and the drop collection.

New capability = one more named call + `steps.append` in `_classify_bucket`,
or one more line in `classify_vectors`. No registry, no dispatch table.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rastervec.config import (
    DENSITY_DEFAULT_GRID_SIZE,
    DENSITY_MAX_CELL_PX,
    DENSITY_MAX_EMPTY_FRACTION,
    DENSITY_MIN_CELL_PX,
    DUPLICATE_RUN_MIN_LENGTH,
    LOW_VARIETY_MAX_MEMBER_COUNT,
    LOW_VARIETY_MAX_REQUIRED,
    LOW_VARIETY_MIN_MEMBER_COUNT,
    LOW_VARIETY_MIN_REQUIRED,
    MAX_DIMENSION_FRACTION,
    MIN_GROUP_SIZE_PX,
    PATTERN_FRACTION_THRESHOLD,
    PATTERN_MIN_REPEAT_COUNT,
    PATTERN_SPACING_TOLERANCE,
    PERIMETER_MARGIN_FRACTION,
    SEQ_OVERLAP_TOLERANCE_PX,
    SIGNATURE_ROUND_PX,
    SPATIAL_CLUSTER_THRESHOLD,
    SPATIAL_SIZE_TOLERANCE,
)
from rastervec.models import Page, VectorPath, VectorRecord
from rastervec.pipelines.result import ClusteringStageResult
from rastervec.Vector.vector import separate_by_color, separate_by_layer
from rastervec.Vector_Classification import cluster_filters as clf
from rastervec.Vector_Classification import group_filters as grf
from rastervec.Vector_Classification import item_filters as itf
from rastervec.Vector_Classification.classification import (
    CategoryResult,
    StepResult,
    build_vector_records,
    group_similar_clusters,
)


@dataclass
class ClassificationResult:
    text_clusters: list[list[VectorPath]]
    cluster_groups: dict[int, list[list[VectorPath]]]
    clustering: dict
    dropped: list[VectorPath]
    records: list[VectorRecord] = field(default_factory=list)
    similarity_groups: list[list[list[VectorPath]]] = field(default_factory=list)
    cluster_similarity_id: dict[int, int] = field(default_factory=dict)
    paths_by_layer: dict | None = None
    paths_by_layer_color: dict | None = None


def _classify_bucket(paths: list[VectorPath], page: Page) -> list[StepResult]:
    """The fixed classification chain for one (layer, color) bucket -- one
    named step-module call per step, each wrapped into a `StepResult`. The
    previous step's `"kept"` category feeds the next."""
    groups: list[list[VectorPath]] = [[p] for p in paths]
    steps: list[StepResult] = []

    groups, dropped = itf.filter_large_items(groups, page, MAX_DIMENSION_FRACTION)
    steps.append(StepResult("Large items", {
        "kept": CategoryResult(groups, "kept"),
        "dropped_oversized": CategoryResult(dropped, "dropped"),
    }))

    groups, signature_counts = itf.compute_vector_signatures(groups, SIGNATURE_ROUND_PX)
    steps.append(StepResult(
        "Vector signatures", {"kept": CategoryResult(groups, "kept")},
        signature_counts=signature_counts,
    ))

    groups, duplicate_runs = grf.remove_duplicate_runs(
        groups, SIGNATURE_ROUND_PX, DUPLICATE_RUN_MIN_LENGTH
    )
    groups, _ = grf.combine_overlapping_seq(groups, SEQ_OVERLAP_TOLERANCE_PX)
    steps.append(StepResult("Seq dedupe + overlap merge", {
        "kept": CategoryResult(groups, "kept"),
        "duplicate_runs": CategoryResult(duplicate_runs, "dropped"),
    }))

    groups, dropped = grf.filter_tiny_groups(groups, MIN_GROUP_SIZE_PX)
    steps.append(StepResult("Tiny groups", {
        "kept": CategoryResult(groups, "kept"),
        "dropped_tiny": CategoryResult(dropped, "dropped"),
    }))

    groups, dropped = grf.filter_large_groups(groups, page, MAX_DIMENSION_FRACTION)
    steps.append(StepResult("Large groups", {
        "kept": CategoryResult(groups, "kept"),
        "dropped_oversized": CategoryResult(dropped, "dropped"),
    }))

    groups, debug_unconstrained, debug_no_parallel, lineage = clf.cluster_spatial_groups(
        groups, SPATIAL_CLUSTER_THRESHOLD, SPATIAL_SIZE_TOLERANCE,
    )
    steps.append(StepResult("Spatial cluster", {
        "kept": CategoryResult(groups, "kept"),
        "debug_unconstrained": CategoryResult(debug_unconstrained, "info"),
        "debug_no_parallel": CategoryResult(debug_no_parallel, "info"),
    }))

    groups, dropped = clf.filter_mixed_fill_rule_clusters(groups)
    steps.append(StepResult("Mixed fill-rule clusters", {
        "kept": CategoryResult(groups, "kept"),
        "dropped_mixed_fill_rule": CategoryResult(dropped, "dropped"),
    }))

    groups, group_stats = grf.compute_group_stats(groups, SIGNATURE_ROUND_PX)
    steps.append(StepResult(
        "Group stats", {"kept": CategoryResult(groups, "kept")},
        group_stats=group_stats,
    ))

    groups, dropped = clf.filter_perimeter_only_clusters(
        groups, group_stats, PERIMETER_MARGIN_FRACTION
    )
    steps.append(StepResult("Perimeter-only clusters", {
        "kept": CategoryResult(groups, "kept"),
        "dropped_perimeter": CategoryResult(dropped, "dropped"),
    }))

    groups, dropped = clf.filter_density_clusters(
        groups, group_stats, DENSITY_DEFAULT_GRID_SIZE, DENSITY_MIN_CELL_PX,
        DENSITY_MAX_CELL_PX, DENSITY_MAX_EMPTY_FRACTION,
    )
    steps.append(StepResult("Density clusters", {
        "kept": CategoryResult(groups, "kept"),
        "dropped_low_density": CategoryResult(dropped, "dropped"),
    }))

    groups, dropped = clf.filter_constant_spacing_clusters(
        groups, SIGNATURE_ROUND_PX, PATTERN_SPACING_TOLERANCE,
        PATTERN_MIN_REPEAT_COUNT, PATTERN_FRACTION_THRESHOLD,
    )
    steps.append(StepResult("Constant-spacing clusters", {
        "kept": CategoryResult(groups, "kept"),
        "dropped_constant_spacing": CategoryResult(dropped, "dropped"),
    }))

    groups, dropped = clf.filter_low_variety_clusters(
        groups, group_stats,
        LOW_VARIETY_MIN_MEMBER_COUNT, LOW_VARIETY_MIN_REQUIRED,
        LOW_VARIETY_MAX_MEMBER_COUNT, LOW_VARIETY_MAX_REQUIRED,
    )
    cluster_groups = {id(g): lineage.get(id(g), [g]) for g in groups}
    steps.append(StepResult(
        "Low-variety clusters", {
            "kept": CategoryResult(groups, "kept"),
            "dropped_low_variety": CategoryResult(dropped, "dropped"),
        },
        cluster_groups=cluster_groups,
    ))

    return steps


def _iter_buckets(paths_by_layer_color: dict):
    return [
        ((layer, color), paths)
        for layer, color_groups in paths_by_layer_color.items()
        for color, paths in color_groups.items()
    ]


def _collect_dropped(clustering: dict) -> list[VectorPath]:
    out: list[VectorPath] = []
    for stage in clustering.values():
        for step in stage.steps:
            for category in step.categories.values():
                if category.role == "dropped":
                    for group in category.groups:
                        out.extend(group)
    return out


def classify_vectors(
    vector_paths: list[VectorPath], page: Page, *, verbose: bool = False,
) -> ClassificationResult:
    """Separate by (layer, color), run `_classify_bucket` per bucket, gather
    every bucket's surviving "kept" clusters, group them by whole-page
    similarity, and collect every dropped group as drawing content."""
    paths_by_layer = separate_by_layer(vector_paths)
    paths_by_layer_color = {
        layer: separate_by_color(paths) for layer, paths in paths_by_layer.items()
    }

    clustering: dict = {}
    for key, bucket in _iter_buckets(paths_by_layer_color):
        clustering[key] = ClusteringStageResult(steps=_classify_bucket(bucket, page))

    text_clusters: list[list[VectorPath]] = []
    cluster_groups: dict[int, list[list[VectorPath]]] = {}
    records: list[VectorRecord] = []
    for stage in clustering.values():
        if not stage.steps:
            continue
        last = stage.steps[-1]
        # NO copy -- manual_label keys its Ungroup lineage on id(cluster).
        text_clusters.extend(last.categories["kept"].groups)
        if last.cluster_groups:
            cluster_groups.update(last.cluster_groups)
        records.extend(build_vector_records(stage.steps))

    similarity_groups = group_similar_clusters(text_clusters)
    cluster_similarity_id = {
        id(c): gi for gi, group in enumerate(similarity_groups) for c in group
    }

    return ClassificationResult(
        text_clusters=text_clusters,
        cluster_groups=cluster_groups,
        clustering=clustering,
        dropped=_collect_dropped(clustering),
        records=records,
        similarity_groups=similarity_groups,
        cluster_similarity_id=cluster_similarity_id,
        paths_by_layer=paths_by_layer if verbose else None,
        paths_by_layer_color=paths_by_layer_color if verbose else None,
    )
