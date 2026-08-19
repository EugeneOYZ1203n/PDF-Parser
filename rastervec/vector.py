"""Vector stage: extracts and classifies vector drawing paths from a page.

extract_paths/separate_by_layer/separate_by_color/build_drawing_vectors are
implemented, plus the full classification pipeline: a single configurable
chain of up to 8 pipeline steps (4 filters -- filter_layout_panels,
filter_large_bbox, filter_large_group_bbox, filter_aspect_ratio -- plus 7
clustering/grouping operations -- cluster_spatial, cluster_spatial_
union_find, cluster_by_seq, group_overlapping, cluster_groups_by_dimension,
cluster_by_item_path_count, cluster_by_item_bbox -- or "none" to skip an
ordinal position), run in a fully caller-configurable order via cluster().
Every step may repeat any number of times at any position (there's no
uniqueness constraint across the 8 ordinal positions -- e.g. running
cluster_spatial twice, or a filter twice, is valid). PIPELINE_STEPS is the
default order, reproducing the original fixed 5-step pipeline (both filters,
one layer of cluster_spatial, both group filters). Every group that survives
the whole chain is a text candidate handed to OCR (pipeline.py's
ocr_text_clusters stage) -- there's no separate drawing-vs-text heuristic;
everything any filter step drops along the way is drawing content, and OCR
success/failure itself is the signal for whether a given cluster was
actually text. See cluster()'s docstring for the full step semantics.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable

import pymupdf as fitz

from rastervec.geometry import round_color, union_bbox
from rastervec.helpers.clustering import Clustering
from rastervec.logging_setup import get_logger
from rastervec.models import DrawingVector, Page, VectorPath

_LOG = get_logger("vector")


def _is_dashed(dashes: str | None) -> bool:
    # PyMuPDF's "dashes" is a PDF dash-array string like "[] 0" (no dash)
    # or "[3 2] 0" (dashed). An empty array means solid.
    if not dashes:
        return False
    return not dashes.strip().startswith("[]")


class Vector:
    """Extracts and classifies vector drawing paths from a page."""

    # The default order cluster()/classify() use when no explicit order is
    # given: reproduces the original fixed pipeline -- both path-level
    # filters, one layer of cluster_spatial, then both group-level filters
    # -- with the 3 unused ordinal positions "none" (no-ops). Callers (the
    # debug app) can freely reorder, repeat, or substitute any of the other
    # real operations -- filter_layout_panels, filter_large_bbox,
    # cluster_spatial, cluster_spatial_union_find, cluster_by_seq,
    # group_overlapping, cluster_groups_by_dimension,
    # cluster_by_item_path_count, cluster_by_item_bbox,
    # filter_large_group_bbox, filter_aspect_ratio -- into any of the 8
    # ordinal positions via an explicit `order`; a step may repeat.
    PIPELINE_STEPS: tuple[str, ...] = (
        "filter_layout_panels",
        "filter_large_bbox",
        "cluster_spatial",
        "none",
        "none",
        "none",
        "filter_large_group_bbox",
        "filter_aspect_ratio",
    )

    def __init__(
        self,
        *,
        spatial_threshold: float = 8.0,
        seq_max_gap: int = 3,
        large_bbox_area_fraction: float = 0.2,
        max_aspect_ratio: float = 10.0,
        group_dimension_tolerance: float = 0.35,
        item_count_max_gap: int = 2,
        item_bbox_tolerance: float = 0.35,
    ) -> None:
        # spatial_threshold is deliberately "high tolerance" (a loose gap
        # threshold, so nearby-but-not-touching paths still merge) --
        # cluster_by_seq's seq_max_gap is the "lower tolerance" pass that
        # tightens the resulting groups back up by draw order.
        self.spatial_threshold = spatial_threshold
        self.seq_max_gap = seq_max_gap
        self.large_bbox_area_fraction = large_bbox_area_fraction
        self.max_aspect_ratio = max_aspect_ratio
        self.group_dimension_tolerance = group_dimension_tolerance
        self.item_count_max_gap = item_count_max_gap
        self.item_bbox_tolerance = item_bbox_tolerance
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
    # Filtering
    # ------------------------------------------------------------------

    def filter_layout_panels(self, paths: list[VectorPath]) -> list[VectorPath]:
        item_counts = Counter(path.seq for path in paths)
        kept = [
            path
            for path in paths
            if not (item_counts[path.seq] == 1 and path.kind in ("re", "qu"))
        ]
        _LOG.debug(
            "filter_layout_panels: %d -> %d path(s)", len(paths), len(kept)
        )
        return kept

    def filter_large_bbox(
        self, paths: list[VectorPath], page: Page, *, max_area_fraction: float | None = None,
    ) -> list[VectorPath]:
        """Drop paths whose own bbox covers more than `max_area_fraction`
        (default `large_bbox_area_fraction`) of the page -- like
        filter_layout_panels, this catches border/frame/panel geometry
        (just by size instead of item-count), which is real page furniture,
        not drawing content."""
        if max_area_fraction is None:
            max_area_fraction = self.large_bbox_area_fraction
        page_area = page.meta.width * page.meta.height
        if page_area <= 0:
            return list(paths)

        max_area = page_area * max_area_fraction
        kept = []
        for path in paths:
            x0, y0, x1, y1 = path.bbox
            area = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
            if area <= max_area:
                kept.append(path)

        _LOG.debug(
            "filter_large_bbox: %d -> %d path(s) (max_area=%.1f)",
            len(paths), len(kept), max_area,
        )
        return kept

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def cluster_spatial(
        self, paths: list[VectorPath], *, threshold: float | None = None
    ) -> list[list[VectorPath]]:
        """High-tolerance pass: paths within `threshold` (default
        `spatial_threshold`) of each other end up in the same (usually
        loose) cluster."""
        if threshold is None:
            threshold = self.spatial_threshold
        return self._clustering.cluster_spatial(
            paths, get_bbox=lambda p: p.bbox, threshold=threshold
        )

    def cluster_spatial_union_find(
        self, groups: list[list[VectorPath]], *, threshold: float | None = None
    ) -> list[list[VectorPath]]:
        """Same distance rule as cluster_spatial (bbox gap <= `threshold`,
        default `spatial_threshold`, via the same grid-bucketed
        union-find), but applied to the *incoming groups themselves*
        rather than the raw paths: each group is treated as one atomic
        unit (by its aggregate bbox), so groups an earlier operation
        already formed only ever get merged together here, never re-split
        or re-derived from scratch -- unlike cluster_spatial, which always
        re-flattens its input first (see _apply_pipeline_step). Reuses
        Clustering.cluster_spatial at the group level the same way
        cluster_groups_by_dimension reuses Clustering.cluster_by_dimension.
        Its own `threshold` override is independent of cluster_spatial's --
        both default to the same `spatial_threshold` instance attribute,
        but a per-call override here never affects cluster_spatial's."""
        if threshold is None:
            threshold = self.spatial_threshold
        super_groups = self._clustering.cluster_spatial(
            groups, get_bbox=lambda g: union_bbox([p.bbox for p in g]),
            threshold=threshold,
        )
        return [[p for g in super_group for p in g] for super_group in super_groups]

    def cluster_by_seq(
        self, groups: list[list[VectorPath]], *, max_gap: int | None = None
    ) -> list[list[VectorPath]]:
        """Lower-tolerance pass within each spatial cluster: splits it
        further by drawing sequence-number proximity (default
        `seq_max_gap`)."""
        if max_gap is None:
            max_gap = self.seq_max_gap
        return self._clustering.cluster_by_seq(
            groups, get_seq=lambda p: p.seq, max_gap=max_gap
        )

    @staticmethod
    def default_overlap_tolerance(page: Page) -> float:
        """group_overlapping's default tolerance when no explicit override
        is given: `max(0.5% of the page's smaller dimension, 3px)`. A
        `staticmethod` (not derived from a `Vector.__init__` param) since
        it's page-dependent, not a fixed setting -- exposed so callers
        (the debug app) can show/pre-fill this computed default before the
        user overrides it."""
        return max(0.005 * min(page.meta.width, page.meta.height), 3.0)

    def group_overlapping(
        self, groups: list[list[VectorPath]], page: Page, *, tolerance: float | None = None,
        bbox_scope: str = "path",
    ) -> list[list[VectorPath]]:
        """Merge members of `groups` (the current incoming groups) whose
        bboxes overlap OR are within a small gap `tolerance` of each other;
        members where one bbox fully contains/equals another are left
        separate regardless of tolerance. Defaults to
        `default_overlap_tolerance(page)` when not given.

        `bbox_scope` picks what "close or overlapping" is measured between:
        - "path" (default): every individual path across all of `groups` is
          flattened into one pool first and compared by its own bbox, so a
          glyph's separate strokes can merge even if they started out in
          different incoming groups (e.g. the strokes making up one glyph
          or symbol, often a pixel or two apart rather than truly
          touching) -- a genuine from-scratch merge, not scoped to
          whatever grouping already existed.
        - "cluster": each incoming *entry* of `groups` is instead treated
          as one atomic unit, compared by its own aggregate bbox -- two
          incoming groups either fully merge or don't, never splitting one
          apart internally -- mirrors cluster_spatial_union_find's
          group-atomic approach, but using the overlap/tolerance/
          containment rule instead of a flat gap threshold."""
        if tolerance is None:
            tolerance = self.default_overlap_tolerance(page)
        if bbox_scope == "path":
            flat = [p for g in groups for p in g]
            return self._clustering.group_by_overlap(
                [flat], get_bbox=lambda p: p.bbox, tolerance=tolerance
            )
        if bbox_scope == "cluster":
            super_groups = self._clustering.group_by_overlap(
                [groups], get_bbox=lambda g: union_bbox([p.bbox for p in g]),
                tolerance=tolerance,
            )
            return [[p for g in super_group for p in g] for super_group in super_groups]
        raise ValueError(f"unknown bbox_scope: {bbox_scope!r}")

    def filter_large_group_bbox(
        self, groups: list[list[VectorPath]], page: Page, *, max_area_fraction: float | None = None,
    ) -> list[list[VectorPath]]:
        """Drop groups whose overall bbox covers more than
        `max_area_fraction` (default `large_bbox_area_fraction`) of the
        page -- same "real content, not page furniture" rule
        filter_large_bbox applies per-path, applied per-group (a group can
        end up oversized even when no single member path was)."""
        if max_area_fraction is None:
            max_area_fraction = self.large_bbox_area_fraction
        page_area = page.meta.width * page.meta.height
        if page_area <= 0:
            return list(groups)

        max_area = page_area * max_area_fraction
        kept = []
        for group in groups:
            x0, y0, x1, y1 = union_bbox([p.bbox for p in group])
            area = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
            if area <= max_area:
                kept.append(group)

        _LOG.debug(
            "filter_large_group_bbox: %d -> %d group(s) (max_area=%.1f)",
            len(groups), len(kept), max_area,
        )
        return kept

    def filter_aspect_ratio(
        self, groups: list[list[VectorPath]], *, max_aspect_ratio: float | None = None,
    ) -> list[list[VectorPath]]:
        """Drop groups shaped like a long thin line/ruler (bbox aspect
        ratio > `max_aspect_ratio`, default the instance's own
        `max_aspect_ratio`) -- real drawing content, but not a text
        candidate, so it never enters the dimension-similarity pass."""
        if max_aspect_ratio is None:
            max_aspect_ratio = self.max_aspect_ratio
        kept = [g for g in groups if self._aspect_ratio(g) <= max_aspect_ratio]
        _LOG.debug(
            "filter_aspect_ratio: %d -> %d group(s) (max_ratio=%s)",
            len(groups), len(kept), max_aspect_ratio,
        )
        return kept

    def _aspect_ratio(self, group: list[VectorPath]) -> float:
        x0, y0, x1, y1 = union_bbox([p.bbox for p in group])
        w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
        return max(w, h) / min(w, h)

    def cluster_groups_by_dimension(
        self, groups: list[list[VectorPath]], *, tolerance: float | None = None,
    ) -> list[list[VectorPath]]:
        """Merge groups whose overall bbox width/height are similar to
        each other (relative difference <= `tolerance`, default
        `group_dimension_tolerance`, on both axes) -- e.g. the individual
        glyph-groups of one text run. Reuses Clustering.cluster_by_
        dimension's generic pairwise pass, treating each incoming group
        (not each path) as one item to compare; the result is flattened
        back to plain path lists."""
        if tolerance is None:
            tolerance = self.group_dimension_tolerance
        super_groups = self._clustering.cluster_by_dimension(
            [groups],
            get_bbox=lambda g: union_bbox([p.bbox for p in g]),
            tolerance=tolerance,
        )
        return [[p for g in super_group for p in g] for super_group in super_groups]

    @staticmethod
    def _compute_item_stats(
        paths: list[VectorPath],
    ) -> tuple[dict[int, int], dict[int, tuple[float, float, float, float]]]:
        """Per-`seq` (i.e. per original drawing/item) stats, computed once
        from the full path population entering cluster() -- stays stable
        for the whole chain regardless of how paths get regrouped along the
        way, so cluster_by_item_path_count/cluster_by_item_bbox always
        measure a path's *original* item, never whatever cluster it
        currently happens to sit in. Returns (path counts, aggregate
        bboxes), both keyed by seq."""
        by_seq: dict[int, list[VectorPath]] = defaultdict(list)
        for path in paths:
            by_seq[path.seq].append(path)
        counts = {seq: len(group) for seq, group in by_seq.items()}
        bboxes = {seq: union_bbox([p.bbox for p in group]) for seq, group in by_seq.items()}
        return counts, bboxes

    def cluster_by_item_path_count(
        self, groups: list[list[VectorPath]], item_counts: dict[int, int],
        *, max_gap: int | None = None,
    ) -> list[list[VectorPath]]:
        """Cluster by how many paths belong to a path's original item (its
        `seq`) -- e.g. a drawing built from many small strokes vs. one made
        of a single filled shape. `item_counts` is precomputed once per
        cluster() call (see _compute_item_stats), not derived from the
        current grouping. Paths flatten into one pool first (a genuine
        from-scratch pass, like cluster_spatial), then split by gaps in
        sorted item-path-count (default `item_count_max_gap`, an absolute
        integer difference -- small counts are intuitive to compare
        directly rather than relatively)."""
        if max_gap is None:
            max_gap = self.item_count_max_gap
        flat = [p for g in groups for p in g]
        return self._clustering.cluster_by_seq(
            [flat], get_seq=lambda p: item_counts[p.seq], max_gap=max_gap
        )

    def cluster_by_item_bbox(
        self, groups: list[list[VectorPath]], item_bboxes: dict[int, tuple[float, float, float, float]],
        *, tolerance: float | None = None,
    ) -> list[list[VectorPath]]:
        """Cluster by the aggregate bbox width/height of a path's original
        item (its `seq`) -- `item_bboxes` is precomputed once per cluster()
        call (see _compute_item_stats), not the current group's own bbox.
        Paths flatten into one pool first (a genuine from-scratch pass),
        then merge by relative width/height closeness (default
        `item_bbox_tolerance`), same style as cluster_groups_by_dimension."""
        if tolerance is None:
            tolerance = self.item_bbox_tolerance
        flat = [p for g in groups for p in g]
        return self._clustering.cluster_by_dimension(
            [flat], get_bbox=lambda p: item_bboxes[p.seq], tolerance=tolerance
        )

    @staticmethod
    def _filter_step(
        groups: list[list[VectorPath]],
        filter_fn: Callable[[list[VectorPath]], list[VectorPath]],
    ) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
        """Shared reconciliation for a path-level filter (filter_
        layout_panels/filter_large_bbox) running inside the pipeline chain:
        `filter_fn` decides path-by-path over the *current* flattened
        population (so a later position in the chain sees fewer/different
        candidates than an earlier one -- deliberately dynamic, unlike the
        item-size steps' precomputed stats). A group keeps its still-kept
        members (dropping an oversized/singleton-panel member doesn't
        remove its groupmates); each dropped path becomes its own
        singleton group in the drop bucket, matching how a cluster-level
        filter's drops are represented."""
        flat = [p for g in groups for p in g]
        kept_ids = {id(p) for p in filter_fn(flat)}
        kept_groups: list[list[VectorPath]] = []
        dropped_groups: list[list[VectorPath]] = []
        for g in groups:
            keep_members = [p for p in g if id(p) in kept_ids]
            if keep_members:
                kept_groups.append(keep_members)
            dropped_groups.extend([p] for p in g if id(p) not in kept_ids)
        return kept_groups, dropped_groups

    @staticmethod
    def _group_filter_step(
        groups: list[list[VectorPath]], kept_groups: list[list[VectorPath]],
    ) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
        """Shared reconciliation for a group-level filter (filter_
        large_group_bbox/filter_aspect_ratio): `kept_groups` is whatever
        that filter method already decided to keep (as the same list
        objects, so identity comparison recovers what was dropped)."""
        kept_ids = {id(g) for g in kept_groups}
        dropped_groups = [g for g in groups if id(g) not in kept_ids]
        return kept_groups, dropped_groups

    def _apply_pipeline_step(
        self, step: str, groups: list[list[VectorPath]], page: Page,
        params: dict[str, float] | None,
        item_counts: dict[int, int], item_bboxes: dict[int, tuple[float, float, float, float]],
    ) -> tuple[list[list[VectorPath]], list[list[VectorPath]]]:
        """Returns (kept_groups, dropped_groups) -- dropped_groups is
        always empty for a pure clustering/grouping step (nothing is ever
        discarded, only regrouped); only the 4 filter steps can drop."""
        params = params or {}
        if step == "none":
            # Identity: lets a caller (the debug app) skip an ordinal
            # position entirely without special-casing it outside cluster().
            return groups, []
        if step == "filter_layout_panels":
            return self._filter_step(groups, self.filter_layout_panels)
        if step == "filter_large_bbox":
            return self._filter_step(
                groups,
                lambda flat: self.filter_large_bbox(
                    flat, page, max_area_fraction=params.get("max_area_fraction")
                ),
            )
        if step == "filter_large_group_bbox":
            kept = self.filter_large_group_bbox(
                groups, page, max_area_fraction=params.get("max_area_fraction")
            )
            return self._group_filter_step(groups, kept)
        if step == "filter_aspect_ratio":
            kept = self.filter_aspect_ratio(
                groups, max_aspect_ratio=params.get("max_aspect_ratio")
            )
            return self._group_filter_step(groups, kept)
        if step == "cluster_spatial":
            # cluster_spatial is fundamentally a from-scratch spatial pass
            # (it takes a flat path list, not groups), so re-flatten
            # whatever grouping exists so far before re-clustering it --
            # it's never a groups-in/groups-out refinement of its input.
            flat = [p for g in groups for p in g]
            return self.cluster_spatial(flat, threshold=params.get("threshold")), []
        if step == "cluster_spatial_union_find":
            return self.cluster_spatial_union_find(groups, threshold=params.get("threshold")), []
        if step == "cluster_by_seq":
            return self.cluster_by_seq(groups, max_gap=params.get("max_gap")), []
        if step == "group_overlapping":
            return self.group_overlapping(
                groups, page, tolerance=params.get("tolerance"),
                bbox_scope=params.get("bbox_scope", "path"),
            ), []
        if step == "cluster_groups_by_dimension":
            return self.cluster_groups_by_dimension(groups, tolerance=params.get("tolerance")), []
        if step == "cluster_by_item_path_count":
            return self.cluster_by_item_path_count(
                groups, item_counts, max_gap=params.get("max_gap")
            ), []
        if step == "cluster_by_item_bbox":
            return self.cluster_by_item_bbox(
                groups, item_bboxes, tolerance=params.get("tolerance")
            ), []
        raise ValueError(f"unknown pipeline step: {step!r}")

    def cluster(
        self, paths: list[VectorPath], page: Page, order: list[str] | None = None,
        step_params: dict[str, dict[str, float]] | None = None,
    ) -> tuple[list[list[list[VectorPath]]], list[list[list[VectorPath]]]]:
        """Runs up to 8 pipeline steps in `order` (default PIPELINE_STEPS --
        both path-level filters, one layer of cluster_spatial, both
        group-level filters, the other 3 ordinal positions "none"), each
        step's input being the previous step's *kept* output ("none" passes
        its input through unchanged). Any step may repeat any number of
        times at any position -- there's no uniqueness constraint. Returns
        `(kept_snapshots, dropped_snapshots)`: one groups-list snapshot per
        step for each, in the same order as `order`
        (`kept_snapshots[-1]` is the final surviving groups;
        `dropped_snapshots[i]` is only what step `i` itself dropped, not
        cumulative -- callers wanting the running total sum
        `dropped_snapshots[0:i+1]`), so callers (the debug app) can show/
        compare state after each individual step. `step_params`, if given,
        maps a step key (e.g. "cluster_spatial") to a dict of that method's
        own keyword overrides (e.g. {"threshold": 12.0}) -- keyed by step,
        not ordinal position, so a repeated step still shares one set of
        overrides across every position it appears at; a step missing from
        `step_params` (or a param missing from that step's dict) just uses
        that method's own instance-attribute default.

        Item-size steps (cluster_by_item_path_count/cluster_by_item_bbox)
        measure each path's *original* item (see _compute_item_stats),
        computed once here from the full incoming `paths` before any step
        runs -- so their result never depends on what an earlier step in
        `order` already did to the grouping."""
        order = list(order) if order else list(self.PIPELINE_STEPS)
        step_params = step_params or {}
        item_counts, item_bboxes = self._compute_item_stats(paths)
        groups: list[list[VectorPath]] = [[p] for p in paths]
        kept_snapshots: list[list[list[VectorPath]]] = []
        dropped_snapshots: list[list[list[VectorPath]]] = []
        for step in order:
            groups, dropped = self._apply_pipeline_step(
                step, groups, page, step_params.get(step), item_counts, item_bboxes,
            )
            kept_snapshots.append(groups)
            dropped_snapshots.append(dropped)
        return kept_snapshots, dropped_snapshots

    def classify(
        self, paths: list[VectorPath], page: Page, cluster_order: list[str] | None = None,
    ) -> list[list[VectorPath]]:
        """Runs cluster() with its default (or given) order and returns
        just the final surviving groups -- a convenience wrapper for
        callers that don't need the drop bookkeeping (pipeline.py's own
        stage wiring calls cluster() directly instead, to keep every
        step's drops for the debug app and drawing_vectors)."""
        kept_snapshots, _dropped_snapshots = self.cluster(paths, page, cluster_order)
        return kept_snapshots[-1] if kept_snapshots else []

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
