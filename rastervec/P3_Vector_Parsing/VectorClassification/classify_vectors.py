"""Vector Classification sub-pipeline: turn raw vectors into text
candidates + drawing content.

Read `_classify_bucket` to see the reduced 2-step chain (seqno-overlap
merge + spatial clustering) as one named call per step. `classify_vectors`
wraps it with the per-`(layer, color)`-bucket loop and the drop collection.
Whole-page similarity grouping no longer happens here -- it runs later, on
post-Radon `Segment`s (see `OCR/radon.py` / `pipelines/_steps.py`).

New capability = one more named call + `steps.append` in `_classify_bucket`,
or one more line in `classify_vectors`. No registry, no dispatch table.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rastervec.P3_Vector_Parsing.VectorClassification.config import (
    SEQ_OVERLAP_TOLERANCE_PX,
    SPATIAL_CLUSTER_THRESHOLD,
    SPATIAL_SIZE_TOLERANCE,
)
from rastervec.commons.models import Page, Vector
from rastervec.P3_Vector_Parsing.VectorClassification.layer_color_separation import separate_by_color, separate_by_layer
from rastervec.P3_Vector_Parsing.VectorClassification import cluster_filters as clf
from rastervec.P3_Vector_Parsing.VectorClassification import group_filters as grf
from rastervec.P3_Vector_Parsing.VectorClassification.classification import CategoryResult, StepResult


@dataclass
class ClusteringStageResult:
    """One (layer, color) bucket's Vector Classification result: `steps` is
    exactly `_classify_bucket()`'s return value. `steps[-1].categories
    ["kept"]` is the final surviving (tiered) clusters; every
    `role="dropped"` category across every step is drawing content."""

    steps: "list[StepResult]"


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
    """The reduced classification chain for one (layer, color) bucket: only
    the seqno-overlap merge and the constrained spatial clustering step
    remain (every other filter from the original 12-step chain has been
    removed -- see git history for the full chain if it's ever needed
    again). Neither remaining step drops anything, so `drawing_vectors`
    downstream is populated entirely by the later FAST stage, not this
    one."""
    groups: list[list[Vector]] = [[v] for v in vectors]
    steps: list[StepResult] = []

    groups, _ = grf.combine_overlapping_seq(groups, SEQ_OVERLAP_TOLERANCE_PX)
    steps.append(StepResult("Seq overlap merge", {
        "kept": CategoryResult(groups, "kept"),
    }))

    clusters = clf.cluster_spatial_groups(
        groups, SPATIAL_CLUSTER_THRESHOLD, SPATIAL_SIZE_TOLERANCE,
    )
    steps.append(StepResult("Spatial cluster", {
        "kept": CategoryResult(clusters, "kept"),
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
