"""CollinearVectorClass classification: turn raw vectors into text-candidate
clusters + drawing content.

Per `(layer, color, width)` bucket, `_classify_bucket` runs five named steps:

1. **Collinear drawing** -- group the bucket's straight Vectors by same
   infinite line (`line_geometry.group_collinear`); a group with more than
   `COLLINEAR_DRAWING_MIN_COUNT` members whose length std is below
   `COLLINEAR_DRAWING_MAX_STD_PT` (a long dashed/repeated line) is dropped
   to drawing.
2. **Seq overlap merge** (`group_filters.combine_overlapping_seq`).
3. **Spatial cluster** (`cluster_filters.cluster_spatial_groups`).
4. **Length outliers** -- per cluster, pool the lengths of every straight
   Vector in a parallel group (>= `MIN_PARALLEL_GROUP_SIZE` same-angle
   members); drop the ones more than `LENGTH_OUTLIER_STD` std from the mean.
5. **Crossings** -- per cluster, drop Vectors properly crossed by more than
   `MAX_CROSSING_SEGMENTS` distinct foreign segments.

`classify_vectors` also returns the page's **global potential angles**:
every collinear group's mean angle across every bucket (singletons and
drawing groups included), deduped within `ANGLE_TOL_DEG` -- `parse.py` snaps
Hough readings to these when a quad has no parallel group.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rastervec.commons.models import Page, Vector
from rastervec.P3_Vector_Parsing.CollinearVectorClass import cluster_filters as clf
from rastervec.P3_Vector_Parsing.CollinearVectorClass import group_filters as grf
from rastervec.P3_Vector_Parsing.CollinearVectorClass.classification import CategoryResult, StepResult
from rastervec.P3_Vector_Parsing.CollinearVectorClass.config import (
    ANGLE_TOL_DEG,
    COLLINEAR_DRAWING_MAX_STD_PT,
    COLLINEAR_DRAWING_MIN_COUNT,
    COLLINEAR_OFFSET_TOL_PT,
    CROSS_EPS_PT,
    CURVE_STEPS,
    LENGTH_OUTLIER_STD,
    MAX_CROSSING_SEGMENTS,
    MIN_PARALLEL_GROUP_SIZE,
    SEQ_OVERLAP_TOLERANCE_PX,
    SPATIAL_CLUSTER_THRESHOLD,
    SPATIAL_SIZE_TOLERANCE,
    STRAIGHT_TOL_PT,
)
from rastervec.P3_Vector_Parsing.CollinearVectorClass.layer_color_separation import (
    separate_by_color, separate_by_layer, separate_by_width,
)
from rastervec.P3_Vector_Parsing.CollinearVectorClass.line_geometry import (
    crossing_segment_counts,
    dedupe_angles,
    group_angle,
    group_collinear,
    group_length_std,
    group_parallel,
    split_straight,
)

STEP_LABELS = ("Collinear drawing", "Seq overlap merge", "Spatial cluster", "Length outliers", "Crossings")


@dataclass
class ClusteringStageResult:
    """One (layer, color, width) bucket's result: `steps` is exactly
    `_classify_bucket()`'s return value. `steps[-1].categories["kept"]` is
    the final (tiered) clusters; every `role="dropped"` category is
    drawing content."""

    steps: "list[StepResult]"


@dataclass
class ClassificationResult:
    """`text_clusters` is tiered (`list[list[list[Vector]]]` -- per cluster,
    its member groups). `drawing_vectors` is flat -- every dropped Vector.
    `global_angles` is the page's deduped collinear-group angles (deg,
    [0, 180))."""

    text_clusters: list[list[list[Vector]]]
    drawing_vectors: list[Vector]
    clustering: dict
    global_angles: list[float]
    vectors_by_layer: dict | None = None
    vectors_by_layer_color: dict | None = None
    vectors_by_layer_color_width: dict | None = None


def collinear_drawing(
    vectors: list[Vector],
) -> tuple[list[Vector], list[list[Vector]], list[float]]:
    """Step 1 for one bucket: `(kept, drawing_groups, group_angles)` --
    `kept` in input order, `drawing_groups` the collinear groups judged to
    be drawing, `group_angles` every collinear group's mean angle."""
    straight, _ = split_straight(vectors, STRAIGHT_TOL_PT)
    groups = group_collinear(straight, ANGLE_TOL_DEG, COLLINEAR_OFFSET_TOL_PT)
    drawing_groups = [
        [v for v, _ in g] for g in groups
        if len(g) > COLLINEAR_DRAWING_MIN_COUNT and group_length_std(g) < COLLINEAR_DRAWING_MAX_STD_PT
    ]
    dropped = {id(v) for g in drawing_groups for v in g}
    kept = [v for v in vectors if id(v) not in dropped]
    return kept, drawing_groups, [group_angle(g) for g in groups]


def length_outliers(cluster_vectors: list[Vector]) -> list[Vector]:
    """Step 4 for one cluster: straight Vectors in parallel groups whose
    length is more than `LENGTH_OUTLIER_STD` pooled std from the pooled mean
    (pooled over every parallel group in the cluster). Nothing when fewer
    than two such strokes or a zero std."""
    straight, _ = split_straight(cluster_vectors, STRAIGHT_TOL_PT)
    members = [
        (v, fit) for g in group_parallel(straight, ANGLE_TOL_DEG)
        if len(g) >= MIN_PARALLEL_GROUP_SIZE for v, fit in g
    ]
    if len(members) < 2:
        return []
    lengths = np.asarray([fit.length for _, fit in members])
    mean, std = float(lengths.mean()), float(lengths.std())
    if std == 0.0:
        return []
    return [v for v, fit in members if abs(fit.length - mean) > LENGTH_OUTLIER_STD * std]


def heavily_crossed(cluster_vectors: list[Vector]) -> list[Vector]:
    """Step 5 for one cluster: Vectors crossed by more than
    `MAX_CROSSING_SEGMENTS` distinct foreign segments."""
    counts = crossing_segment_counts(cluster_vectors, eps=CROSS_EPS_PT, curve_steps=CURVE_STEPS)
    return [v for v, n in zip(cluster_vectors, counts) if n > MAX_CROSSING_SEGMENTS]


def _remove_from_clusters(
    clusters: list[list[list[Vector]]], removed_fn,
) -> tuple[list[list[list[Vector]]], list[list[Vector]]]:
    """Apply `removed_fn(flat cluster vectors) -> removed vectors` to every
    cluster, keeping group tiering: `(kept clusters, dropped per cluster)`.
    Emptied groups and clusters disappear."""
    kept_clusters, dropped = [], []
    for cluster in clusters:
        flat = [v for g in cluster for v in g]
        removed = removed_fn(flat)
        if not removed:
            kept_clusters.append(cluster)
            continue
        gone = {id(v) for v in removed}
        dropped.append(removed)
        new_cluster = [[v for v in g if id(v) not in gone] for g in cluster]
        new_cluster = [g for g in new_cluster if g]
        if new_cluster:
            kept_clusters.append(new_cluster)
    return kept_clusters, dropped


def _classify_bucket(vectors: list[Vector]) -> tuple[list[StepResult], list[float]]:
    """The five-step chain for one bucket (see module docstring), plus the
    bucket's collinear-group angles."""
    steps: list[StepResult] = []

    kept, drawing_groups, angles = collinear_drawing(vectors)
    steps.append(StepResult(STEP_LABELS[0], {
        "kept": CategoryResult([[v] for v in kept], "kept"),
        "drawing": CategoryResult(drawing_groups, "dropped"),
    }))

    groups, _ = grf.combine_overlapping_seq([[v] for v in kept], SEQ_OVERLAP_TOLERANCE_PX)
    steps.append(StepResult(STEP_LABELS[1], {"kept": CategoryResult(groups, "kept")}))

    clusters = clf.cluster_spatial_groups(groups, SPATIAL_CLUSTER_THRESHOLD, SPATIAL_SIZE_TOLERANCE)
    steps.append(StepResult(STEP_LABELS[2], {"kept": CategoryResult(clusters, "kept")}))

    clusters, outliers = _remove_from_clusters(clusters, length_outliers)
    steps.append(StepResult(STEP_LABELS[3], {
        "kept": CategoryResult(clusters, "kept"),
        "outliers": CategoryResult(outliers, "dropped"),
    }))

    clusters, crossed = _remove_from_clusters(clusters, heavily_crossed)
    steps.append(StepResult(STEP_LABELS[4], {
        "kept": CategoryResult(clusters, "kept"),
        "crossed": CategoryResult(crossed, "dropped"),
    }))
    return steps, angles


def _iter_buckets(vectors_by_layer_color_width: dict):
    return [
        ((layer, color, width), vectors)
        for layer, color_groups in vectors_by_layer_color_width.items()
        for color, width_groups in color_groups.items()
        for width, vectors in width_groups.items()
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
    """Separate by (layer, color, width), run `_classify_bucket` per bucket,
    gather every bucket's final clusters, every dropped Vector (drawing) and
    the page's deduped global potential angles."""
    vectors_by_layer = separate_by_layer(vectors)
    vectors_by_layer_color = {
        layer: separate_by_color(vs) for layer, vs in vectors_by_layer.items()
    }
    vectors_by_layer_color_width = {
        layer: {color: separate_by_width(vs) for color, vs in color_groups.items()}
        for layer, color_groups in vectors_by_layer_color.items()
    }

    clustering: dict = {}
    all_angles: list[float] = []
    for key, bucket in _iter_buckets(vectors_by_layer_color_width):
        steps, angles = _classify_bucket(bucket)
        clustering[key] = ClusteringStageResult(steps=steps)
        all_angles.extend(angles)

    text_clusters: list[list[list[Vector]]] = []
    for stage in clustering.values():
        text_clusters.extend(stage.steps[-1].categories["kept"].groups)

    return ClassificationResult(
        text_clusters=text_clusters,
        drawing_vectors=_collect_dropped(clustering),
        clustering=clustering,
        global_angles=dedupe_angles(all_angles, ANGLE_TOL_DEG),
        vectors_by_layer=vectors_by_layer if verbose else None,
        vectors_by_layer_color=vectors_by_layer_color if verbose else None,
        vectors_by_layer_color_width=vectors_by_layer_color_width if verbose else None,
    )
