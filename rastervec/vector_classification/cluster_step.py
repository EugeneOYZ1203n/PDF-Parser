"""Generic Cluster step: splits existing groups into smaller ones by a
metric's similarity/closeness. Two methods:

- "global": from-scratch spatial-style clustering -- flattens every
  incoming group into one pool, then forms new clusters by pairwise
  distance <= threshold (single-linkage). `metric="spatial_gap"` reuses
  the grid-bucketed union-find fast path (Clustering.cluster_spatial);
  any other pairwise metric goes through the generic O(k^2)
  cluster_by_pairwise_distance instead.
- "pairwise": splits each incoming group further by gap in a *sorted*
  scalar metric (e.g. seq, item_path_count) -- consecutive items more
  than `threshold` apart in sorted order start a new cluster. `scope`
  picks whether this splits within each incoming group ("within_group",
  matches the old cluster_by_seq) or flattens everything into one pool
  first ("global_flatten", matches the old cluster_by_item_path_count).
"""
from __future__ import annotations

from rastervec.helpers.clustering import Clustering
from rastervec.models import VectorPath
from rastervec.vector_classification.metrics import MetricContext, PAIRWISE_METRICS, SCALAR_METRICS


def apply_cluster_step(
    groups: list[list[VectorPath]], step, ctx: MetricContext, clustering: Clustering,
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    if step.method == "global":
        flat = [p for g in groups for p in g]
        if step.metric == "spatial_gap":
            new_groups = clustering.cluster_spatial(
                flat, get_bbox=lambda p: p.bbox, threshold=step.threshold,
            )
        else:
            metric = PAIRWISE_METRICS[step.metric]
            distance_fn = lambda a, b: metric.compute(a, b, ctx, step.params)  # noqa: E731
            new_groups = clustering.cluster_by_pairwise_distance(
                [flat], distance_fn, threshold=step.threshold,
            )
        return new_groups, []

    if step.method == "pairwise":
        metric = SCALAR_METRICS[step.metric]
        get_value = lambda p: metric.compute(p, ctx, step.params)  # noqa: E731
        if step.scope == "global_flatten":
            flat = [p for g in groups for p in g]
            return clustering.cluster_by_seq([flat], get_seq=get_value, max_gap=step.threshold), []
        return clustering.cluster_by_seq(groups, get_seq=get_value, max_gap=step.threshold), []

    raise ValueError(f"unknown cluster method: {step.method!r}")
