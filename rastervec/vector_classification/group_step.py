"""Generic Group step: merges nearby/similar existing groups by a pairwise
metric (distance <= threshold). `scope` picks what's compared:

- "path": every individual path across all incoming groups is flattened
  into one pool first and compared by its own bbox (matches the old
  group_overlapping's bbox_scope="path") -- a genuine from-scratch merge,
  not scoped to whatever grouping already existed.
- "cluster": each incoming group is treated as one atomic unit, compared
  by its own aggregate bbox (matches cluster_spatial_union_find and
  group_overlapping's bbox_scope="cluster") -- two incoming groups either
  fully merge or don't, never splitting one apart internally.
"""
from __future__ import annotations

from rastervec.helpers.clustering import Clustering
from rastervec.models import VectorPath
from rastervec.vector_classification.metrics import MetricContext, PAIRWISE_METRICS


def apply_group_step(
    groups: list[list[VectorPath]], step, ctx: MetricContext, clustering: Clustering,
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    metric = PAIRWISE_METRICS[step.metric]
    distance_fn = lambda a, b: metric.compute(a, b, ctx, step.params)  # noqa: E731

    if step.scope == "path":
        flat = [p for g in groups for p in g]
        return clustering.cluster_by_pairwise_distance(
            [flat], distance_fn, threshold=step.threshold,
        ), []

    if step.scope == "cluster":
        super_groups = clustering.cluster_by_pairwise_distance(
            [groups], distance_fn, threshold=step.threshold,
        )
        return [[p for g in sg for p in g] for sg in super_groups], []

    raise ValueError(f"unknown group scope: {step.scope!r}")
