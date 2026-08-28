"""Vector Classification: a single fixed, non-configurable 12-step
pipeline that classifies extracted VectorPaths into text candidates vs.
drawing content, run in order by `VectorClassifier.cluster()`. Each step
is implemented as a plain function in this package's items/groups/
clusters submodules (see each submodule's own docstring for its steps'
descriptions):

  items/item_filters.py    -- steps 1-2 (filter_large_items, compute_vector_signatures)
  groups/group_filters.py  -- steps 3-5, 8 (seq dedupe/merge, tiny/large groups, group stats)
  clusters/cluster_filters.py -- steps 6-7, 9-12 (spatial cluster, mixed fill-rule,
                                 perimeter/density/constant-spacing/low-variety)

See Glossary.md for standardized group/cluster/global-group/similarity-group
terminology.

Step 6's spatial merge also tracks lineage: `cluster()`'s final
`StepResult.cluster_groups` (keyed by `id(cluster)`) records which of
step 3's pre-spatial-clustering "groups" each surviving cluster is
composed of.

All thresholds are module-level constants below -- tune the pipeline by
editing them here, not at runtime. Each step's result is wrapped into a
`StepResult` holding one or more named `CategoryResult`s -- exactly one
per step has `role="kept"` and feeds the next step; every other category
is a side-channel for the debug UI (a `role="dropped"` category is folded
into `drawing_vectors` by pipeline.py, same as every other drop). Every
path that survives the whole chain (the last step's `"kept"` category) is
a text candidate handed downstream (pipeline.py's text_candidates/
fast_text_detect/ocr_compare stages) -- there's no separate
drawing-vs-text heuristic; everything any filter step drops along the way
is drawing content, and OCR success/failure itself is the signal for
whether a given cluster was actually text.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from rastervec.helpers.clustering import Clustering
from rastervec.helpers.geometry import is_dashed, union_bbox
from rastervec.logging_setup import get_logger
from rastervec.models import DrawingVector, Page, VectorPath, VectorRecord
from rastervec.Vector_Classification.clusters import cluster_filters as clf
from rastervec.Vector_Classification.groups import group_filters as grf
from rastervec.Vector_Classification.items import item_filters as itf

_LOG = get_logger("classification")

# Classification pipeline thresholds -- edit these to tune classification;
# there is no runtime/UI way to change them.
MAX_DIMENSION_FRACTION = 0.10  # steps 1 & 4: max item/group dimension, as a fraction of the page's smaller side
SIGNATURE_ROUND_PX = 0.5  # step 2: grid size shape signatures are rounded to
DUPLICATE_RUN_MIN_LENGTH = 5  # step 3a: min consecutive identical-signature run length to drop
SEQ_OVERLAP_TOLERANCE_PX = 1.0  # step 3b: bbox gap tolerance when chain-merging by seq order
MIN_GROUP_SIZE_PX = 3.0  # step 3c: drop a group whose bbox's max dimension is under this many px
SPATIAL_CLUSTER_THRESHOLD = 10.0  # step 5: bbox gap tolerance for spatial clustering
SPATIAL_SIZE_TOLERANCE = 0.30  # step 5: max relative difference between a valid pair of parallel sides for two groups to spatially merge
PERIMETER_MARGIN_FRACTION = 0.1  # step 9: drop a cluster if no member reaches past this fraction of its bbox edges
DENSITY_DEFAULT_GRID_SIZE = 4  # step 10: default cells per axis, clamped to keep DENSITY_MIN_CELL_PX <= cell side <= DENSITY_MAX_CELL_PX
DENSITY_MIN_CELL_PX = 5.0  # step 10: each grid cell's side is at least this many px (fewer cells used if needed)
DENSITY_MAX_CELL_PX = 40.0  # step 10: each grid cell's side is at most this many px (more cells used if needed)
DENSITY_MAX_EMPTY_FRACTION = 0.70  # step 10: drop a cluster if more than this fraction of grid cells are untouched
PATTERN_SPACING_TOLERANCE = 0.20  # step 11: max relative deviation between consecutive same-shape gaps
PATTERN_MIN_REPEAT_COUNT = 3  # step 11: min same-signature members needed to judge spacing consistency
PATTERN_FRACTION_THRESHOLD = 0.70  # step 11: drop a cluster if at least this fraction of its members belong to a patterned sub-group
LOW_VARIETY_MIN_MEMBER_COUNT = 5  # step 12: at or under this many members, only LOW_VARIETY_MIN_REQUIRED distinct signatures are required
LOW_VARIETY_MIN_REQUIRED = 1  # step 12: required distinct signature count for clusters at/under LOW_VARIETY_MIN_MEMBER_COUNT members
LOW_VARIETY_MAX_MEMBER_COUNT = 300  # step 12: at or over this many members, LOW_VARIETY_MAX_REQUIRED distinct signatures are required
LOW_VARIETY_MAX_REQUIRED = 10  # step 12: required distinct signature count for clusters at/over LOW_VARIETY_MAX_MEMBER_COUNT members
UNIQUE_CLUSTER_TOLERANCE = 0.04  # pipeline.py's unique_clusters stage: relative-translation tolerance (fraction of each cluster's own bbox max dimension) for two clusters to be judged the same shape

CategoryRole = Literal["kept", "dropped", "info"]


@dataclass
class CategoryResult:
    """One named category within a pipeline step's result -- a list of
    groups plus its role. Exactly one category per step is `role="kept"`
    (feeds the next step); a `role="dropped"` category is folded into
    `drawing_vectors` by pipeline.py's `_run_drawing_vectors`; `role=
    "info"` is never folded anywhere (used only by step 2's pass-through
    counter category)."""

    groups: list[list[VectorPath]]
    role: CategoryRole


@dataclass
class StepResult:
    """One pipeline step's full result: a display label plus every named
    category it produced (`"kept"` always present, plus any number of
    side categories for the debug UI). `signature_counts`, if set (only on
    step 2's result), is the per-`VectorSignature` occurrence count built
    by `compute_vector_signatures`. `group_stats`, if set, is the
    per-group `GroupStats` built by `compute_group_stats`, keyed by
    `id(group)`. `cluster_groups`, if set (only on the final "Low-variety
    clusters" step's result), maps `id(cluster)` (for every cluster in
    that step's own `"kept"` category) to the list of step 3's
    pre-spatial-clustering "groups" that cluster is composed of."""

    label: str
    categories: dict[str, CategoryResult]
    signature_counts: dict[itf.VectorSignature, int] | None = None
    group_stats: dict[int, grf.GroupStats] | None = None
    cluster_groups: dict[int, list[list[VectorPath]]] | None = None


class VectorClassifier:
    """Classifies extracted vector paths into text candidates vs. drawing
    content, and re-aggregates paths back into DrawingVectors."""

    def __init__(self) -> None:
        self._clustering = Clustering()

    def cluster(self, paths: list[VectorPath], page: Page) -> list[StepResult]:
        """Runs the fixed pipeline (see this module's docstring) in order,
        each step's input being the previous step's `"kept"` category.
        Returns one `StepResult` per step; `steps[-1].categories["kept"]`
        is the final surviving groups, handed to text_candidates (see
        pipeline.py's `_run_text_candidates`)."""
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

        groups, debug_unconstrained, debug_no_parallel, lineage = (
            clf.cluster_spatial_groups(
                groups, self._clustering, SPATIAL_CLUSTER_THRESHOLD, SPATIAL_SIZE_TOLERANCE,
            )
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

        groups, dropped = clf.filter_perimeter_only_clusters(groups, group_stats, PERIMETER_MARGIN_FRACTION)
        steps.append(StepResult("Perimeter-only clusters", {
            "kept": CategoryResult(groups, "kept"),
            "dropped_perimeter": CategoryResult(dropped, "dropped"),
        }))

        groups, dropped = clf.filter_density_clusters(
            groups, group_stats, DENSITY_DEFAULT_GRID_SIZE, DENSITY_MIN_CELL_PX, DENSITY_MAX_CELL_PX,
            DENSITY_MAX_EMPTY_FRACTION,
        )
        steps.append(StepResult("Density clusters", {
            "kept": CategoryResult(groups, "kept"),
            "dropped_low_density": CategoryResult(dropped, "dropped"),
        }))

        groups, dropped = clf.filter_constant_spacing_clusters(
            groups, SIGNATURE_ROUND_PX, PATTERN_SPACING_TOLERANCE, PATTERN_MIN_REPEAT_COUNT,
            PATTERN_FRACTION_THRESHOLD,
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

    def group_similar_clusters(
        self, clusters: list[list[VectorPath]],
    ) -> list[list[list[VectorPath]]]:
        """Whole-page similarity grouping of text-candidate clusters -- see
        `clf.group_similar_clusters` and Glossary.md's "similarity group"
        entry. Uses `UNIQUE_CLUSTER_TOLERANCE`."""
        return clf.group_similar_clusters(clusters, UNIQUE_CLUSTER_TOLERANCE)

    def classify(self, paths: list[VectorPath], page: Page) -> list[list[VectorPath]]:
        """Runs cluster() and returns just the final surviving groups -- a
        convenience wrapper for callers that don't need the per-step/
        per-category bookkeeping (pipeline.py's own stage wiring calls
        cluster() directly instead, to keep every step's categories for the
        debug app and drawing_vectors)."""
        steps = self.cluster(paths, page)
        return steps[-1].categories["kept"].groups if steps else []

    @staticmethod
    def _bbox_and_representative(paths: list[VectorPath]) -> dict:
        """Shared by build_vector_records/build_drawing_vectors: a group's
        own union bbox, plus style fields read off its first member as a
        representative value (paths within one group/cluster share the
        same drawing-level style in practice)."""
        first = paths[0]
        return dict(
            bbox=union_bbox([p.bbox for p in paths]),
            stroke_color=first.stroke_color,
            fill_color=first.fill_color,
            stroke_width=first.stroke_width,
            dashed=is_dashed(first.dashes),
            page_index=first.page_index,
        )

    def build_vector_records(self, steps: list[StepResult]) -> list[VectorRecord]:
        """Wires the final step's `cluster_groups` lineage into one
        `VectorRecord` per surviving (role="kept") text-candidate cluster --
        built right here, where the lineage is already known, rather than
        reconstructed after the fact from a flattened list. Drawing-level
        fields VectorPath doesn't carry (even_odd/line_cap/line_join/
        scissor/blendmode/isolated/knockout/opacity) fall back to
        false/0/None -- a cluster can merge paths from several different
        original drawings, so there's no single drawing left to read them
        from; `seqno` uses the cluster's first member's synthetic `seq` as
        a representative value instead."""
        if not steps:
            return []
        last = steps[-1]
        kept = last.categories["kept"].groups
        lineage = last.cluster_groups or {}

        records: list[VectorRecord] = []
        for cluster in kept:
            if not cluster:
                continue
            common = self._bbox_and_representative(cluster)
            first = cluster[0]
            records.append(
                VectorRecord(
                    items=cluster,
                    even_odd=False,
                    line_cap=0,
                    line_join=0,
                    seqno=first.seq,
                    rect=common["bbox"],
                    scissor=None,
                    blendmode=None,
                    isolated=False,
                    knockout=False,
                    opacity=None,
                    groups=lineage.get(id(cluster), [cluster]),
                    role="kept",
                    **common,
                )
            )
        return records

    def build_drawing_vectors(self, paths: list[VectorPath]) -> list[DrawingVector]:
        groups: dict[int, list[VectorPath]] = defaultdict(list)
        for path in paths:
            groups[path.seq].append(path)

        result = []
        for group in groups.values():
            result.append(DrawingVector(paths=group, **self._bbox_and_representative(group)))

        _LOG.debug("build_drawing_vectors: %d path(s) -> %d drawing(s)", len(paths), len(result))
        return result
