"""Vector Classification sub-pipeline: turn raw vectors into text
candidates + drawing content.

Read `_classify_bucket` to see the fixed 12-step chain as one named call
per step. `classify_vectors` wraps it with the per-`(layer, color)`-bucket
loop and the drop collection. Whole-page similarity grouping no longer
happens here -- it runs later, on post-Radon `Segment`s (see
`OCR/radon.py` / `pipelines/_steps.py`).

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
from rastervec.models import Page, Vector
from rastervec.pipelines.result import ClusteringStageResult
from rastervec.Vector.vector import separate_by_color, separate_by_layer
from rastervec.Vector_Classification import cluster_filters as clf
from rastervec.Vector_Classification import group_filters as grf
from rastervec.Vector_Classification import item_filters as itf
from rastervec.Vector_Classification.classification import CategoryResult, StepResult


@dataclass
class ClassificationResult:
    """`text_clusters` is tiered: `list[list[list[Vector]]]` -- one entry
    per surviving cluster, each a `list[list[Vector]]` of its member
    groups (see `cluster_filters.cluster_spatial_groups`). `drawing_vectors`
    is flat -- every dropped Vector across every bucket/step, in no
    particular order."""

    text_clusters: list[list[list[Vector]]]
    drawing_vectors: list[Vector]
    clustering: dict
    vectors_by_layer: dict | None = None
    vectors_by_layer_color: dict | None = None


def _classify_bucket(vectors: list[Vector], page: Page) -> list[StepResult]:
    """The fixed classification chain for one (layer, color) bucket -- one
    named step-module call per step, each wrapped into a `StepResult`. The
    previous step's `"kept"` category feeds the next. Steps 1-5 operate on
    `list[list[Vector]]` groups; step 6 onward operate on tiered
    `list[list[list[Vector]]]` clusters."""
    groups: list[list[Vector]] = [[v] for v in vectors]
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

    clusters, debug_unconstrained, debug_no_parallel = clf.cluster_spatial_groups(
        groups, SPATIAL_CLUSTER_THRESHOLD, SPATIAL_SIZE_TOLERANCE,
    )
    steps.append(StepResult("Spatial cluster", {
        "kept": CategoryResult(clusters, "kept"),
        "debug_unconstrained": CategoryResult(debug_unconstrained, "info"),
        "debug_no_parallel": CategoryResult(debug_no_parallel, "info"),
    }))

    clusters, dropped = clf.filter_mixed_fill_rule_clusters(clusters)
    steps.append(StepResult("Mixed fill-rule clusters", {
        "kept": CategoryResult(clusters, "kept"),
        "dropped_mixed_fill_rule": CategoryResult(dropped, "dropped"),
    }))

    clusters, group_stats = grf.compute_group_stats(clusters, SIGNATURE_ROUND_PX)
    steps.append(StepResult(
        "Group stats", {"kept": CategoryResult(clusters, "kept")},
        group_stats=group_stats,
    ))

    clusters, dropped = clf.filter_perimeter_only_clusters(
        clusters, group_stats, PERIMETER_MARGIN_FRACTION
    )
    steps.append(StepResult("Perimeter-only clusters", {
        "kept": CategoryResult(clusters, "kept"),
        "dropped_perimeter": CategoryResult(dropped, "dropped"),
    }))

    clusters, dropped = clf.filter_density_clusters(
        clusters, group_stats, DENSITY_DEFAULT_GRID_SIZE, DENSITY_MIN_CELL_PX,
        DENSITY_MAX_CELL_PX, DENSITY_MAX_EMPTY_FRACTION,
    )
    steps.append(StepResult("Density clusters", {
        "kept": CategoryResult(clusters, "kept"),
        "dropped_low_density": CategoryResult(dropped, "dropped"),
    }))

    clusters, dropped = clf.filter_constant_spacing_clusters(
        clusters, SIGNATURE_ROUND_PX, PATTERN_SPACING_TOLERANCE,
        PATTERN_MIN_REPEAT_COUNT, PATTERN_FRACTION_THRESHOLD,
    )
    steps.append(StepResult("Constant-spacing clusters", {
        "kept": CategoryResult(clusters, "kept"),
        "dropped_constant_spacing": CategoryResult(dropped, "dropped"),
    }))

    clusters, dropped = clf.filter_low_variety_clusters(
        clusters, group_stats,
        LOW_VARIETY_MIN_MEMBER_COUNT, LOW_VARIETY_MIN_REQUIRED,
        LOW_VARIETY_MAX_MEMBER_COUNT, LOW_VARIETY_MAX_REQUIRED,
    )
    steps.append(StepResult("Low-variety clusters", {
        "kept": CategoryResult(clusters, "kept"),
        "dropped_low_variety": CategoryResult(dropped, "dropped"),
    }))

    return steps


def _iter_buckets(vectors_by_layer_color: dict):
    return [
        ((layer, color), vectors)
        for layer, color_groups in vectors_by_layer_color.items()
        for color, vectors in color_groups.items()
    ]


def _collect_dropped(clustering: dict) -> list[Vector]:
    out: list[Vector] = []
    for stage in clustering.values():
        for step in stage.steps:
            for category in step.categories.values():
                if category.role != "dropped":
                    continue
                for entry in category.groups:
                    if entry and isinstance(entry[0], list):
                        out.extend(v for g in entry for v in g)
                    else:
                        out.extend(entry)
    return out


def classify_vectors(
    vectors: list[Vector], page: Page, *, verbose: bool = False,
) -> ClassificationResult:
    """Separate by (layer, color), run `_classify_bucket` per bucket, gather
    every bucket's surviving "kept" clusters (tiered, real nested
    structure -- no lineage side-channel), and collect every dropped
    Vector as drawing content."""
    vectors_by_layer = separate_by_layer(vectors)
    vectors_by_layer_color = {
        layer: separate_by_color(vs) for layer, vs in vectors_by_layer.items()
    }

    clustering: dict = {}
    for key, bucket in _iter_buckets(vectors_by_layer_color):
        clustering[key] = ClusteringStageResult(steps=_classify_bucket(bucket, page))

    text_clusters: list[list[list[Vector]]] = []
    for stage in clustering.values():
        if not stage.steps:
            continue
        last = stage.steps[-1]
        # NO copy -- manual_label keys its Ungroup lineage on id(cluster).
        text_clusters.extend(last.categories["kept"].groups)

    return ClassificationResult(
        text_clusters=text_clusters,
        drawing_vectors=_collect_dropped(clustering),
        clustering=clustering,
        vectors_by_layer=vectors_by_layer if verbose else None,
        vectors_by_layer_color=vectors_by_layer_color if verbose else None,
    )
