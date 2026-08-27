"""Group-level steps of the Vector Classification pipeline (see
rastervec/Vector_Classification/classification.py for the fixed step
order these are run in). A "group" is the post-seqno-overlap-merge, pre-
spatial-clustering unit -- see Glossary.md for the group/cluster
distinction.

3. `remove_duplicate_runs` + `combine_overlapping_seq` -- sorts by `seq`,
   drops maximal runs of `DUPLICATE_RUN_MIN_LENGTH`-or-more consecutive
   items sharing the exact same shape signature (e.g. a long strip of
   identical tick marks), then chain-merges whatever's left into groups by
   bbox-gap tolerance.
4. `filter_tiny_groups` -- drop a whole group if its aggregate bbox's max
   dimension is under `MIN_GROUP_SIZE_PX` -- a leftover speck too small
   to be a real glyph.
5. `filter_large_groups` -- same dimension check as `filter_large_items`,
   applied to each group's aggregate bbox instead of one item's.
8. `compute_group_stats` -- informational, pure pass-through: computes and
   returns per-group `GroupStats` (member count, distinct shape-signature
   count, bbox) keyed by `id(group)` -- consumed by cluster-level filters.

All thresholds are passed in by the caller (`classification.py`'s own
constants), so nothing here is hardcoded.
"""
from __future__ import annotations

from dataclasses import dataclass

from rastervec.helpers.geometry import rect_gap, union_bbox
from rastervec.models import Page, VectorPath
from rastervec.Vector_Classification.items.item_filters import (
    _bbox_of,
    _max_dimension,
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
    groups: list[list[VectorPath]], round_px: float, min_run_length: int,
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    """Sorts flat by `seq`, then walks the sequence collecting maximal runs
    of consecutive items sharing the exact same shape signature. A run
    with `min_run_length` or more members is dropped whole (as one group
    per run, for the debug UI); shorter runs are left untouched and kept
    as individual single-item groups, ready for `combine_overlapping_seq`."""
    flat = sorted((p for g in groups for p in g), key=lambda p: p.seq)
    if not flat:
        return [], []

    kept: list[VectorPath] = []
    dropped_runs: list[list[VectorPath]] = []

    def _flush(run: list[VectorPath]) -> None:
        if len(run) >= min_run_length:
            dropped_runs.append(run)
        else:
            kept.extend(run)

    run = [flat[0]]
    run_sig = vector_signature(flat[0], round_px)
    for path in flat[1:]:
        sig = vector_signature(path, round_px)
        if sig == run_sig:
            run.append(path)
        else:
            _flush(run)
            run = [path]
            run_sig = sig
    _flush(run)

    return [[p] for p in kept], dropped_runs


def combine_overlapping_seq(
    groups: list[list[VectorPath]], tolerance: float,
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    """Sorts every incoming path by `seq`, then sweeps in that order:
    a path joins the current group if its bbox is within `tolerance` of
    the group's aggregate bbox so far, otherwise it starts a new group.
    Never drops anything -- pure regrouping."""
    flat = sorted((p for g in groups for p in g), key=lambda p: p.seq)
    if not flat:
        return [], []

    result: list[list[VectorPath]] = []
    current = [flat[0]]
    current_bbox = flat[0].bbox
    for path in flat[1:]:
        if rect_gap(current_bbox, path.bbox) <= tolerance:
            current.append(path)
            current_bbox = union_bbox([current_bbox, path.bbox])
        else:
            result.append(current)
            current = [path]
            current_bbox = path.bbox
    result.append(current)
    return result, []


def filter_tiny_groups(
    groups: list[list[VectorPath]], min_size_px: float,
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    """Drops a whole group if its aggregate bbox's max dimension is under
    `min_size_px` -- a leftover speck too small to be a real glyph."""
    kept: list[list[VectorPath]] = []
    dropped: list[list[VectorPath]] = []
    for g in groups:
        if _max_dimension(_bbox_of(g)) < min_size_px:
            dropped.append(g)
        else:
            kept.append(g)
    return kept, dropped


def filter_large_groups(
    groups: list[list[VectorPath]], page: Page, max_dimension_fraction: float,
) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
    """Same dimension check as `filter_large_items`, applied to each
    group's own aggregate bbox instead of one path's -- an all-or-nothing
    drop per group."""
    page_min = min(page.meta.width, page.meta.height)
    threshold = max_dimension_fraction * page_min if page_min > 0 else float("inf")

    kept = [g for g in groups if _max_dimension(_bbox_of(g)) <= threshold]
    kept_ids = {id(g) for g in kept}
    dropped = [g for g in groups if id(g) not in kept_ids]
    return kept, dropped


def compute_group_stats(
    groups: list[list[VectorPath]],
    round_px: float,
) -> tuple[list[list[VectorPath]], dict[int, GroupStats]]:
    """Informational pass-through, like `compute_vector_signatures`: never
    drops or regroups anything. For every group (keyed by `id(group)` --
    stable through this step and every later step that keeps the same
    group objects), records member count, the number of distinct shape
    signatures among its members, and its bbox/max dimension -- consumed
    by the cluster-level filters and available to any future downstream
    consumer wanting per-group shape stats without recomputing them."""
    stats: dict[int, GroupStats] = {}
    for g in groups:
        sigs = {vector_signature(p, round_px) for p in g}
        bbox = _bbox_of(g)
        stats[id(g)] = GroupStats(
            member_count=len(g),
            unique_signature_count=len(sigs),
            bbox=bbox,
            max_dimension=_max_dimension(bbox),
        )
    return groups, stats
