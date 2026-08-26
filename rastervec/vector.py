"""Vector stage: extracts and classifies vector drawing paths from a page.

extract_paths/separate_by_layer/separate_by_color/build_drawing_vectors are
implemented, plus classification: a single fixed, non-configurable
pipeline run in order by cluster(), each step a plain function in
rastervec/helpers/vector_classification.py (see that module's docstring
for the full description of each step):

1. filter_large_items                        -- drop oversized items.
2. compute_vector_signatures                  -- informational shape-signature counter.
3. remove_duplicate_runs + combine_overlapping_seq -- sort by seq, drop long
                                                  runs of exact-duplicate shapes, then chain-merge overlaps.
4. filter_tiny_groups                         -- drop groups smaller than MIN_GROUP_SIZE_PX.
5. filter_large_groups                        -- drop oversized groups.
6. cluster_spatial_groups                     -- spatial merge, constrained to groups sharing any
                                                  similar-length, parallel side; also reports two
                                                  debug-only unconstrained variants.
7. filter_mixed_fill_rule_clusters            -- drop clusters mixing fill/stroke paint styles.
8. compute_group_stats                        -- informational per-group stats (member/signature counts).
9. filter_perimeter_only_clusters             -- drop border/ring-only clusters.
10. filter_density_clusters                   -- drop clusters too sparse across their own bbox
                                                  grid (5-40px cells, dropped past 40% empty).
11. filter_constant_spacing_clusters          -- drop clusters where >=70% of members belong to a
                                                  near-perfectly-regular repeated same-shape sub-group.
12. filter_low_variety_clusters               -- drop clusters below a member-count-scaled
                                                  minimum distinct-shape-type count.

See Glossary.md for standardized group/cluster terminology.

Step 6's spatial merge also tracks lineage: `cluster()`'s final
`StepResult.cluster_groups` (keyed by `id(cluster)`) records which of
step 3's pre-spatial-clustering "groups" each surviving cluster is
composed of -- "group" means step 6's own input unit (post
seqno-overlap-merge, pre-spatial-clustering); "cluster" means the final
classification output (post spatial clustering and every filter after
it). Every later step only filters whole clusters (never re-splits one),
so a cluster's member-path identity stays a clean union of some subset of
step 6's input groups all the way to the end.

All thresholds are module-level constants below -- tune the pipeline by
editing them here, not at runtime. There is no caller-configurable step
list any more. Each step's result is wrapped into a `StepResult` holding
one or more named `CategoryResult`s -- exactly one per step has
`role="kept"` and feeds the next step; every other category is a
side-channel for the debug UI (a `role="dropped"` category is folded into
`drawing_vectors` by pipeline.py, same as every other drop). Every path
that survives the whole chain (the last step's `"kept"` category) is a
text candidate handed downstream (pipeline.py's text_candidates/
fast_text_detect/ocr_compare stages) -- there's no separate
drawing-vs-text heuristic; everything any filter step drops along the way
is drawing content, and OCR success/failure itself is the signal for
whether a given cluster was actually text.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import pymupdf as fitz

from rastervec.geometry import round_color
from rastervec.helpers import vector_classification as vc
from rastervec.helpers.clustering import Clustering
from rastervec.logging_setup import get_logger
from rastervec.models import DrawingVector, Page, VectorPath

_LOG = get_logger("vector")

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
    by `compute_vector_signatures`, reused by later steps and the debug
    app's "color by vector type" view. `group_stats`, if set (only on the
    "Group stats" step's result), is the per-group `GroupStats` built by
    `compute_group_stats`, keyed by `id(group)`, reused by
    `filter_low_variety_clusters` and available for any future downstream
    consumer. `cluster_groups`, if set (only on the final "Low-variety
    clusters" step's result), maps `id(cluster)` (for every cluster in
    that step's own `"kept"` category) to the list of step 3's
    pre-spatial-clustering "groups" that cluster is composed of -- see
    this module's docstring for the group/cluster distinction."""

    label: str
    categories: dict[str, CategoryResult]
    signature_counts: dict[vc.VectorSignature, int] | None = None
    group_stats: dict[int, vc.GroupStats] | None = None
    cluster_groups: dict[int, list[list[VectorPath]]] | None = None


def _is_dashed(dashes: str | None) -> bool:
    # PyMuPDF's "dashes" is a PDF dash-array string like "[] 0" (no dash)
    # or "[3 2] 0" (dashed). An empty array means solid.
    if not dashes:
        return False
    return not dashes.strip().startswith("[]")


class Vector:
    """Extracts and classifies vector drawing paths from a page."""

    def __init__(self) -> None:
        self._clustering = Clustering()

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_paths(self, page: Page) -> list[VectorPath]:
        fitz_page = page.fitz_page
        page_index = page.meta.index
        paths: list[VectorPath] = []

        for seq, drawing in enumerate(fitz_page.get_drawings()):
            fill_rule = drawing.get("type", "")
            stroke_color = round_color(drawing.get("color"))
            fill_color = round_color(drawing.get("fill"))
            stroke_opacity = drawing.get("stroke_opacity")
            fill_opacity = drawing.get("fill_opacity")
            stroke_width = drawing.get("width")
            dashes = drawing.get("dashes")
            closed = drawing.get("closePath")
            layer = drawing.get("layer") or None

            common = dict(
                seq=seq,
                fill_rule=fill_rule,
                stroke_color=stroke_color,
                fill_color=fill_color,
                stroke_opacity=stroke_opacity,
                fill_opacity=fill_opacity,
                stroke_width=stroke_width,
                dashes=dashes,
                closed=closed,
                layer=layer,
                page_index=page_index,
            )

            for item_index, item in enumerate(drawing.get("items", [])):
                path = self._extract_item(item, item_index, common)
                if path is not None:
                    paths.append(path)

        _LOG.debug("page %d: extracted %d vector path(s)", page_index, len(paths))
        return paths

    def _extract_item(
        self, item: tuple, item_index: int, common: dict
    ) -> VectorPath | None:
        op = item[0]
        if op == "l":
            return self._extract_line(item, item_index, common)
        if op == "re":
            return self._extract_rect(item, item_index, common)
        if op == "qu":
            return self._extract_quad(item, item_index, common)
        if op == "c":
            return self._extract_curve(item, item_index, common)
        return None

    def _extract_line(self, item: tuple, item_index: int, common: dict) -> VectorPath:
        p1, p2 = fitz.Point(item[1]), fitz.Point(item[2])
        bbox = (
            min(p1.x, p2.x),
            min(p1.y, p2.y),
            max(p1.x, p2.x),
            max(p1.y, p2.y),
        )
        return VectorPath(
            item_index=item_index,
            kind="l",
            points=[(p1.x, p1.y), (p2.x, p2.y)],
            bbox=bbox,
            **common,
        )

    def _extract_rect(self, item: tuple, item_index: int, common: dict) -> VectorPath:
        rect = fitz.Rect(item[1])
        return VectorPath(
            item_index=item_index,
            kind="re",
            points=[(rect.x0, rect.y0), (rect.x1, rect.y1)],
            bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
            **common,
        )

    def _extract_quad(self, item: tuple, item_index: int, common: dict) -> VectorPath:
        quad = fitz.Quad(item[1])
        points = [
            (quad.ul.x, quad.ul.y),
            (quad.ur.x, quad.ur.y),
            (quad.lr.x, quad.lr.y),
            (quad.ll.x, quad.ll.y),
        ]
        rect = quad.rect
        return VectorPath(
            item_index=item_index,
            kind="qu",
            points=points,
            bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
            **common,
        )

    def _extract_curve(self, item: tuple, item_index: int, common: dict) -> VectorPath:
        points = [fitz.Point(p) for p in item[1:5]]
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        return VectorPath(
            item_index=item_index,
            kind="c",
            points=[(p.x, p.y) for p in points],
            bbox=bbox,
            **common,
        )

    # ------------------------------------------------------------------
    # Layer / color separation
    # ------------------------------------------------------------------

    def separate_by_layer(
        self, paths: list[VectorPath]
    ) -> dict[str, list[VectorPath]]:
        groups: dict[str, list[VectorPath]] = defaultdict(list)
        for path in paths:
            groups[path.layer or ""].append(path)
        _LOG.debug("separated %d path(s) into %d layer(s)", len(paths), len(groups))
        return dict(groups)

    def separate_by_color(
        self, paths: list[VectorPath]
    ) -> dict[tuple, list[VectorPath]]:
        groups: dict[tuple, list[VectorPath]] = defaultdict(list)

        for path in paths:
            key = (
                path.stroke_color,
                path.fill_color,
                path.stroke_opacity,
                path.fill_opacity,
            )
            groups[key].append(path)

        _LOG.debug(
            "separated %d path(s) into %d color/opacity groups",
            len(paths),
            len(groups),
        )
        return dict(groups)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def cluster(self, paths: list[VectorPath], page: Page) -> list[StepResult]:
        """Runs the fixed pipeline (see this module's docstring) in order,
        each step's input being the previous step's `"kept"` category.
        Returns one `StepResult` per step, each holding every named
        category that step produced -- `steps[-1].categories["kept"]`
        is the final surviving groups, handed to text_candidates (see
        pipeline.py's `_run_text_candidates`)."""
        groups: list[list[VectorPath]] = [[p] for p in paths]
        steps: list[StepResult] = []

        groups, dropped = vc.filter_large_items(groups, page, MAX_DIMENSION_FRACTION)
        steps.append(StepResult("Large items", {
            "kept": CategoryResult(groups, "kept"),
            "dropped_oversized": CategoryResult(dropped, "dropped"),
        }))

        groups, signature_counts = vc.compute_vector_signatures(groups, SIGNATURE_ROUND_PX)
        steps.append(StepResult(
            "Vector signatures", {"kept": CategoryResult(groups, "kept")},
            signature_counts=signature_counts,
        ))

        groups, duplicate_runs = vc.remove_duplicate_runs(
            groups, SIGNATURE_ROUND_PX, DUPLICATE_RUN_MIN_LENGTH
        )
        groups, _ = vc.combine_overlapping_seq(groups, SEQ_OVERLAP_TOLERANCE_PX)
        steps.append(StepResult("Seq dedupe + overlap merge", {
            "kept": CategoryResult(groups, "kept"),
            "duplicate_runs": CategoryResult(duplicate_runs, "dropped"),
        }))

        groups, dropped = vc.filter_tiny_groups(groups, MIN_GROUP_SIZE_PX)
        steps.append(StepResult("Tiny groups", {
            "kept": CategoryResult(groups, "kept"),
            "dropped_tiny": CategoryResult(dropped, "dropped"),
        }))

        groups, dropped = vc.filter_large_groups(groups, page, MAX_DIMENSION_FRACTION)
        steps.append(StepResult("Large groups", {
            "kept": CategoryResult(groups, "kept"),
            "dropped_oversized": CategoryResult(dropped, "dropped"),
        }))

        groups, debug_unconstrained, debug_no_parallel, lineage = (
            vc.cluster_spatial_groups(
                groups, self._clustering, SPATIAL_CLUSTER_THRESHOLD, SPATIAL_SIZE_TOLERANCE,
            )
        )
        steps.append(StepResult("Spatial cluster", {
            "kept": CategoryResult(groups, "kept"),
            "debug_unconstrained": CategoryResult(debug_unconstrained, "info"),
            "debug_no_parallel": CategoryResult(debug_no_parallel, "info"),
        }))

        groups, dropped = vc.filter_mixed_fill_rule_clusters(groups)
        steps.append(StepResult("Mixed fill-rule clusters", {
            "kept": CategoryResult(groups, "kept"),
            "dropped_mixed_fill_rule": CategoryResult(dropped, "dropped"),
        }))

        groups, group_stats = vc.compute_group_stats(groups, SIGNATURE_ROUND_PX)
        steps.append(StepResult(
            "Group stats", {"kept": CategoryResult(groups, "kept")},
            group_stats=group_stats,
        ))

        groups, dropped = vc.filter_perimeter_only_clusters(groups, group_stats, PERIMETER_MARGIN_FRACTION)
        steps.append(StepResult("Perimeter-only clusters", {
            "kept": CategoryResult(groups, "kept"),
            "dropped_perimeter": CategoryResult(dropped, "dropped"),
        }))

        groups, dropped = vc.filter_density_clusters(
            groups, group_stats, DENSITY_DEFAULT_GRID_SIZE, DENSITY_MIN_CELL_PX, DENSITY_MAX_CELL_PX,
            DENSITY_MAX_EMPTY_FRACTION,
        )
        steps.append(StepResult("Density clusters", {
            "kept": CategoryResult(groups, "kept"),
            "dropped_low_density": CategoryResult(dropped, "dropped"),
        }))

        groups, dropped = vc.filter_constant_spacing_clusters(
            groups, SIGNATURE_ROUND_PX, PATTERN_SPACING_TOLERANCE, PATTERN_MIN_REPEAT_COUNT,
            PATTERN_FRACTION_THRESHOLD,
        )
        steps.append(StepResult("Constant-spacing clusters", {
            "kept": CategoryResult(groups, "kept"),
            "dropped_constant_spacing": CategoryResult(dropped, "dropped"),
        }))

        groups, dropped = vc.filter_low_variety_clusters(
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
        `vc.group_similar_clusters` and Glossary.md's "similarity group"
        entry. Uses `UNIQUE_CLUSTER_TOLERANCE`."""
        return vc.group_similar_clusters(clusters, UNIQUE_CLUSTER_TOLERANCE)

    def classify(self, paths: list[VectorPath], page: Page) -> list[list[VectorPath]]:
        """Runs cluster() and returns just the final surviving groups -- a
        convenience wrapper for callers that don't need the per-step/
        per-category bookkeeping (pipeline.py's own stage wiring calls
        cluster() directly instead, to keep every step's categories for the
        debug app and drawing_vectors)."""
        steps = self.cluster(paths, page)
        return steps[-1].categories["kept"].groups if steps else []

    # ------------------------------------------------------------------
    # Drawing vectors
    # ------------------------------------------------------------------

    def build_drawing_vectors(self, paths: list[VectorPath]) -> list[DrawingVector]:
        groups: dict[int, list[VectorPath]] = defaultdict(list)
        for path in paths:
            groups[path.seq].append(path)

        result = []
        for group in groups.values():
            x0 = min(p.bbox[0] for p in group)
            y0 = min(p.bbox[1] for p in group)
            x1 = max(p.bbox[2] for p in group)
            y1 = max(p.bbox[3] for p in group)
            first = group[0]
            result.append(
                DrawingVector(
                    paths=group,
                    bbox=(x0, y0, x1, y1),
                    stroke_color=first.stroke_color,
                    fill_color=first.fill_color,
                    stroke_width=first.stroke_width,
                    dashed=_is_dashed(first.dashes),
                    page_index=first.page_index,
                )
            )

        _LOG.debug("build_drawing_vectors: %d path(s) -> %d drawing(s)", len(paths), len(result))
        return result
