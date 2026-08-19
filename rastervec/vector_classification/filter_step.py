"""Generic Filter step: metric + condition, checked per unit -- a unit
failing the condition becomes drawing (dropped), a unit passing stays
unclassified (kept) for later steps. `scope` picks what the unit is:

- "item": every path across all incoming groups, flattened into one pool
  first (matches the old path-level filters, e.g. filter_large_bbox) --
  group structure is not consulted during evaluation, but a group keeps
  whichever of its members survive.
- "cluster": each incoming group is one unit, evaluated by its own
  aggregate bbox (matches the old group-level filters, e.g.
  filter_aspect_ratio) -- all-or-nothing per group.
- "items_in_cluster": per-path, like "item", but group membership is
  preserved during evaluation (no cross-group flattening) -- the scope
  where `aggregate_scope="group"` is meaningful, since each path's
  aggregate is computed from only its own containing group.
- "cluster_all_items": per-path metric, but a group survives only if
  *every* one of its members passes (matches the old
  filter_item_path_count/filter_item_bbox "all seqs in group must pass"
  behavior).
"""
from __future__ import annotations

from rastervec.models import VectorPath
from rastervec.vector_classification.base import _reconcile_groups, _reconcile_items
from rastervec.vector_classification.metrics import MetricContext, check_condition, resolve_value


def apply_filter_step(
    groups: list[list[VectorPath]], step, ctx: MetricContext,
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    all_paths = [p for g in groups for p in g]

    def passes(unit, population) -> bool:
        value = resolve_value(step, unit, population, ctx)
        return check_condition(step.condition, value, step.threshold)

    if step.scope == "item":
        kept_ids = {id(p) for p in all_paths if passes(p, all_paths)}
        return _reconcile_items(groups, kept_ids)

    if step.scope == "cluster":
        kept_groups = [g for g in groups if passes(g, groups)]
        return _reconcile_groups(groups, kept_groups)

    if step.scope == "items_in_cluster":
        kept_groups: list[list[VectorPath]] = []
        dropped_groups: list[list[VectorPath]] = []
        for g in groups:
            population = all_paths if step.aggregate_scope == "global" else g
            keep = [p for p in g if passes(p, population)]
            if keep:
                kept_groups.append(keep)
            keep_ids = {id(p) for p in keep}
            dropped_groups.extend([p] for p in g if id(p) not in keep_ids)
        return kept_groups, dropped_groups

    if step.scope == "cluster_all_items":
        kept_groups = []
        dropped_groups = []
        for g in groups:
            population = all_paths if step.aggregate_scope == "global" else g
            if all(passes(p, population) for p in g):
                kept_groups.append(g)
            else:
                dropped_groups.append(g)
        return kept_groups, dropped_groups

    raise ValueError(f"unknown filter scope: {step.scope!r}")
