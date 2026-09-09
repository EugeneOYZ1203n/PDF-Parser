"""Group-level steps of the Vector Classification pipeline (see
rastervec/Vector_Classification/classification.py for the fixed step
order these are run in). A "group" is the post-seqno-overlap-merge, pre-
spatial-clustering unit -- see Glossary.md for the group/cluster
distinction.

3. `remove_duplicate_runs` + `combine_overlapping_seq` -- sorts by `seqno`,
   drops maximal runs of `DUPLICATE_RUN_MIN_LENGTH`-or-more consecutive
   Vectors sharing the exact same whole-Vector shape signature (e.g. a long
   strip of identical tick marks -- each its own single-item Vector), then
   chain-merges whatever's left into groups by bbox-gap tolerance.
4. `filter_tiny_groups` -- drop a whole group if its aggregate bbox's max
   dimension is under `MIN_GROUP_SIZE_PX` -- a leftover speck too small
   to be a real glyph.
5. `filter_large_groups` -- same dimension check as `filter_large_items`,
   applied to each group's aggregate bbox instead of one Vector's.
8. `compute_group_stats` -- informational, pure pass-through: computes and
   returns per-cluster `GroupStats` (member count, distinct shape-signature
   count, bbox) keyed by `id(cluster)` -- consumed by cluster-level filters.
   Despite the name (kept for continuity with the numbered-step docs), by
   the point this runs in the chain its input is already post-spatial-merge
   clusters, not pre-spatial groups.

All thresholds are passed in by the caller (`classification.py`'s own
constants), so nothing here is hardcoded.
"""
from __future__ import annotations

from dataclasses import dataclass

from rastervec.helpers.geometry import max_dimension, rect_gap, union_bbox
from rastervec.helpers.iterutils import partition
from rastervec.models import Page, Vector
from rastervec.Vector_Classification.item_filters import (
    bbox_of,
    vector_signature,
)


@dataclass
class GroupStats:
    """Per-group data computed by `compute_group_stats`, for downstream
    filter steps and any future consumer that wants per-group shape stats
    without recomputing them from scratch."""

    member_count: int
    unique_signature_count: int
    bbox: tuple[float, float, float, float]
    max_dimension: float


def remove_duplicate_runs(
    groups: list[list[Vector]], round_px: float, min_run_length: int,
) -> tuple[list[list[Vector]], list[list[Vector]]]:
    """Sorts flat by `seqno`, then walks the sequence collecting maximal
    runs of consecutive Vectors sharing the exact same whole-Vector shape
    signature. A run with `min_run_length` or more members is dropped
    whole (as one group per run, for the debug UI); shorter runs are left
    untouched and kept as individual single-Vector groups, ready for
    `combine_overlapping_seq`."""
    flat = sorted((v for g in groups for v in g), key=lambda v: v.seqno)
    if not flat:
        return [], []

    kept: list[Vector] = []
    dropped_runs: list[list[Vector]] = []

    def _flush(run: list[Vector]) -> None:
        if len(run) >= min_run_length:
            dropped_runs.append(run)
        else:
            kept.extend(run)

    run = [flat[0]]
    run_sig = vector_signature(flat[0], round_px)
    for v in flat[1:]:
        sig = vector_signature(v, round_px)
        if sig == run_sig:
            run.append(v)
        else:
            _flush(run)
            run = [v]
            run_sig = sig
    _flush(run)

    return [[v] for v in kept], dropped_runs


def combine_overlapping_seq(
    groups: list[list[Vector]], tolerance: float,
) -> tuple[list[list[Vector]], list[list[Vector]]]:
    """Sorts every incoming Vector by `seqno`, then sweeps in that order:
    a Vector joins the current group if its bbox is within `tolerance` of
    the group's aggregate bbox so far, otherwise it starts a new group.
    Never drops anything -- pure regrouping."""
    flat = sorted((v for g in groups for v in g), key=lambda v: v.seqno)
    if not flat:
        return [], []

    result: list[list[Vector]] = []
    current = [flat[0]]
    current_bbox = flat[0].bbox
    for v in flat[1:]:
        if rect_gap(current_bbox, v.bbox) <= tolerance:
            current.append(v)
            current_bbox = union_bbox([current_bbox, v.bbox])
        else:
            result.append(current)
            current = [v]
            current_bbox = v.bbox
    result.append(current)
    return result, []


def filter_tiny_groups(
    groups: list[list[Vector]], min_size_px: float,
) -> tuple[list[list[Vector]], list[list[Vector]]]:
    """Drops a whole group if its aggregate bbox's max dimension is under
    `min_size_px` -- a leftover speck too small to be a real glyph."""
    return partition(groups, lambda g: max_dimension(bbox_of(g)) >= min_size_px)


def filter_large_groups(
    groups: list[list[Vector]], page: Page, max_dimension_fraction: float,
) -> tuple[list[list[Vector]], list[list[Vector]]]:
    """Same dimension check as `filter_large_items`, applied to each
    group's own aggregate bbox instead of one Vector's -- an all-or-nothing
    drop per group."""
    page_min = min(page.meta.width, page.meta.height)
    threshold = max_dimension_fraction * page_min if page_min > 0 else float("inf")

    return partition(groups, lambda g: max_dimension(bbox_of(g)) <= threshold)


def compute_group_stats(
    clusters: list[list[list[Vector]]],
    round_px: float,
) -> tuple[list[list[list[Vector]]], dict[int, GroupStats]]:
    """Informational pass-through, like `compute_vector_signatures`: never
    drops or regroups anything. By the point this runs in the chain its
    input is already post-spatial-merge clusters (tiered `list[list[
    Vector]]`, one entry per member group), not pre-spatial groups --
    kept for continuity with the numbered-step docs. For every cluster
    (keyed by `id(cluster)` -- stable through this step and every later
    step that keeps the same cluster objects, since filters only ever
    `partition`/append the same list reference), records member Vector
    count, the number of distinct whole-Vector shape signatures among its
    members, and its bbox/max dimension -- consumed by the cluster-level
    filters."""
    stats: dict[int, GroupStats] = {}
    for cluster in clusters:
        flat = [v for g in cluster for v in g]
        sigs = {vector_signature(v, round_px) for v in flat}
        bbox = bbox_of(flat)
        stats[id(cluster)] = GroupStats(
            member_count=len(flat),
            unique_signature_count=len(sigs),
            bbox=bbox,
            max_dimension=max_dimension(bbox),
        )
    return clusters, stats
