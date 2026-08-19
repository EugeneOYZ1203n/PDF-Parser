"""Dispatches one StepConfig to its kind's engine module."""
from __future__ import annotations

from rastervec.helpers.clustering import Clustering
from rastervec.models import VectorPath
from rastervec.vector_classification.cluster_step import apply_cluster_step
from rastervec.vector_classification.filter_step import apply_filter_step
from rastervec.vector_classification.group_step import apply_group_step
from rastervec.vector_classification.metrics import MetricContext


def run_step(
    step, groups: list[list[VectorPath]], ctx: MetricContext, clustering: Clustering,
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    """Returns (kept_groups, dropped_groups) -- dropped_groups is always
    empty for cluster/group steps (nothing is ever discarded, only
    regrouped); only filter steps can drop."""
    if step.kind == "filter":
        return apply_filter_step(groups, step, ctx)
    if step.kind == "cluster":
        return apply_cluster_step(groups, step, ctx, clustering)
    if step.kind == "group":
        return apply_group_step(groups, step, ctx, clustering)
    raise ValueError(f"unknown step kind: {step.kind!r}")
