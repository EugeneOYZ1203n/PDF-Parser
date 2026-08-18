"""Vector stage: extracts and classifies vector drawing paths from a page.

extract_paths/separate_by_layer/separate_by_color/build_drawing_vectors are
implemented, plus the full classification pipeline: filter out layout
panels and oversized items; run 4 clustering/grouping operations
(cluster_spatial, cluster_by_seq, group_overlapping,
cluster_groups_by_dimension -- see CLUSTER_STEPS) in a configurable order
via cluster(); filter out oversized groups and extreme-aspect-ratio groups
(lines/rules); then a size/fill heuristic decides drawing vs. text-candidate
per final group. See classify()'s docstring for the full order.
"""
from __future__ import annotations

from collections import Counter, defaultdict

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

    # The 4 clustering/grouping operations `cluster()` can chain in any
    # order -- this is also the default order `classify()` uses.
    CLUSTER_STEPS: tuple[str, ...] = (
        "cluster_spatial",
        "cluster_by_seq",
        "group_overlapping",
        "cluster_groups_by_dimension",
    )

    def __init__(
        self,
        *,
        spatial_threshold: float = 8.0,
        seq_max_gap: int = 3,
        large_bbox_area_fraction: float = 0.2,
        max_aspect_ratio: float = 10.0,
        group_dimension_tolerance: float = 0.35,
        text_max_dim: float = 30.0,
        text_min_fill_fraction: float = 0.5,
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
        self.text_max_dim = text_max_dim
        self.text_min_fill_fraction = text_min_fill_fraction
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
            stroke_width = drawing.get("width")
            dashes = drawing.get("dashes")
            closed = drawing.get("closePath")
            layer = drawing.get("layer") or None

            common = dict(
                seq=seq,
                fill_rule=fill_rule,
                stroke_color=stroke_color,
                fill_color=fill_color,
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
            color = path.stroke_color if path.stroke_color is not None else path.fill_color
            groups[color].append(path)
        _LOG.debug("separated %d path(s) into %d color(s)", len(paths), len(groups))
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
        self, paths: list[VectorPath], page: Page
    ) -> list[VectorPath]:
        """Drop paths whose own bbox covers more than
        `large_bbox_area_fraction` of the page -- like filter_layout_panels,
        this catches border/frame/panel geometry (just by size instead of
        item-count), which is real page furniture, not drawing content."""
        page_area = page.meta.width * page.meta.height
        if page_area <= 0:
            return list(paths)

        max_area = page_area * self.large_bbox_area_fraction
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

    def cluster_spatial(self, paths: list[VectorPath]) -> list[list[VectorPath]]:
        """High-tolerance pass: paths within `spatial_threshold` of each
        other end up in the same (usually loose) cluster."""
        return self._clustering.cluster_spatial(
            paths, get_bbox=lambda p: p.bbox, threshold=self.spatial_threshold
        )

    def cluster_by_seq(self, groups: list[list[VectorPath]]) -> list[list[VectorPath]]:
        """Lower-tolerance pass within each spatial cluster: splits it
        further by drawing sequence-number proximity."""
        return self._clustering.cluster_by_seq(
            groups, get_seq=lambda p: p.seq, max_gap=self.seq_max_gap
        )

    def group_overlapping(
        self, clusters: list[list[VectorPath]], page: Page
    ) -> list[list[VectorPath]]:
        """Within each seq-cluster, merge paths whose bboxes overlap OR are
        within a small gap tolerance of each other (e.g. the strokes making
        up one glyph or symbol, which are often a pixel or two apart rather
        than truly touching); paths where one bbox fully contains/equals
        another are left separate regardless of tolerance. The tolerance is
        `max(0.5% of the page's smaller dimension, 3px)`. Scoping stays
        per-cluster: only paths that already landed in the same incoming
        seq-cluster are ever compared -- Clustering.group_by_overlap applies
        its pairwise merge independently to each incoming group, so a
        larger tolerance never merges paths across different clusters."""
        tolerance = max(0.005 * min(page.meta.width, page.meta.height), 3.0)
        return self._clustering.group_by_overlap(
            clusters, get_bbox=lambda p: p.bbox, tolerance=tolerance
        )

    def filter_large_group_bbox(
        self, groups: list[list[VectorPath]], page: Page
    ) -> list[list[VectorPath]]:
        """Drop groups whose overall bbox covers more than
        `large_bbox_area_fraction` of the page -- same "real content, not
        page furniture" rule filter_large_bbox applies per-path, applied
        per-group after overlapping paths have been merged (a group can end
        up oversized even when no single member path was)."""
        page_area = page.meta.width * page.meta.height
        if page_area <= 0:
            return list(groups)

        max_area = page_area * self.large_bbox_area_fraction
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
        self, groups: list[list[VectorPath]]
    ) -> list[list[VectorPath]]:
        """Drop groups shaped like a long thin line/ruler (bbox aspect
        ratio > max_aspect_ratio) -- real drawing content, but not a text
        candidate, so it never enters the dimension-similarity pass."""
        kept = [g for g in groups if self._aspect_ratio(g) <= self.max_aspect_ratio]
        _LOG.debug(
            "filter_aspect_ratio: %d -> %d group(s) (max_ratio=%s)",
            len(groups), len(kept), self.max_aspect_ratio,
        )
        return kept

    def _aspect_ratio(self, group: list[VectorPath]) -> float:
        x0, y0, x1, y1 = union_bbox([p.bbox for p in group])
        w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
        return max(w, h) / min(w, h)

    def cluster_groups_by_dimension(
        self, groups: list[list[VectorPath]]
    ) -> list[list[VectorPath]]:
        """Merge groups whose overall bbox width/height are similar to
        each other (relative difference <= group_dimension_tolerance on
        both axes) -- e.g. the individual glyph-groups of one text run.
        Reuses Clustering.cluster_by_dimension's generic pairwise pass,
        treating each incoming group (not each path) as one item to
        compare; the result is flattened back to plain path lists."""
        super_groups = self._clustering.cluster_by_dimension(
            [groups],
            get_bbox=lambda g: union_bbox([p.bbox for p in g]),
            tolerance=self.group_dimension_tolerance,
        )
        return [[p for g in super_group for p in g] for super_group in super_groups]

    def _apply_cluster_step(
        self, step: str, groups: list[list[VectorPath]], page: Page
    ) -> list[list[VectorPath]]:
        if step == "none":
            # Identity: lets a caller (the debug app) skip an ordinal
            # position entirely without special-casing it outside cluster().
            return groups
        if step == "cluster_spatial":
            # cluster_spatial is fundamentally a from-scratch spatial pass
            # (it takes a flat path list, not groups), so re-flatten
            # whatever grouping exists so far before re-clustering it --
            # unlike the other three, it's never a groups-in/groups-out
            # refinement of its input.
            flat = [p for g in groups for p in g]
            return self.cluster_spatial(flat)
        if step == "cluster_by_seq":
            return self.cluster_by_seq(groups)
        if step == "group_overlapping":
            return self.group_overlapping(groups, page)
        if step == "cluster_groups_by_dimension":
            return self.cluster_groups_by_dimension(groups)
        raise ValueError(f"unknown cluster step: {step!r}")

    def cluster(
        self, paths: list[VectorPath], page: Page, order: list[str] | None = None,
    ) -> list[list[list[VectorPath]]]:
        """Apply the 4 clustering/grouping operations (CLUSTER_STEPS) in
        `order` (default CLUSTER_STEPS itself), each step's input being the
        previous step's output groups. Returns one groups-list snapshot per
        step, in the same order as `order`, so callers (the debug app) can
        show/compare the clustering state after each individual step."""
        order = list(order) if order else list(self.CLUSTER_STEPS)
        groups: list[list[VectorPath]] = [[p] for p in paths]
        snapshots: list[list[list[VectorPath]]] = []
        for step in order:
            groups = self._apply_cluster_step(step, groups, page)
            snapshots.append(groups)
        return snapshots

    def classify_clusters(
        self, clusters: list[list[VectorPath]]
    ) -> tuple[list[VectorPath], list[list[VectorPath]]]:
        drawing_paths: list[VectorPath] = []
        text_clusters: list[list[VectorPath]] = []

        for cluster in clusters:
            if self._looks_like_text(cluster):
                text_clusters.append(cluster)
            else:
                drawing_paths.extend(cluster)

        _LOG.debug(
            "classify_clusters: %d cluster(s) -> %d drawing path(s), %d text cluster(s)",
            len(clusters),
            len(drawing_paths),
            len(text_clusters),
        )
        return drawing_paths, text_clusters

    def classify(
        self, paths: list[VectorPath], page: Page, cluster_order: list[str] | None = None,
    ) -> tuple[list[VectorPath], list[list[VectorPath]]]:
        """The full pipeline, in order: filter out layout panels and
        oversized items; run the 4 clustering/grouping operations in
        `cluster_order` (default CLUSTER_STEPS -- spatial closeness, then
        seq-number proximity, then overlapping/nearly-touching shapes, then
        dimension similarity); filter out oversized groups and
        extreme-aspect-ratio groups (lines/rules); then classify each final
        group as text or drawing. Paths/groups dropped by any filter along
        the way are drawing content too (this method's `drawing_paths`
        return only reflects the final grouping step -- pipeline.py's
        per-stage wiring is what folds every filter's drops back in for the
        pipeline's actual drawing_vectors output)."""
        kept = self.filter_layout_panels(paths)
        kept = self.filter_large_bbox(kept, page)
        final_clusters = self.cluster(kept, page, cluster_order)[-1]
        size_kept = self.filter_large_group_bbox(final_clusters, page)
        aspect_kept = self.filter_aspect_ratio(size_kept)
        return self.classify_clusters(aspect_kept)

    def _looks_like_text(self, cluster: list[VectorPath]) -> bool:
        if len(cluster) < 2:
            return False

        for path in cluster:
            x0, y0, x1, y1 = path.bbox
            if (x1 - x0) > self.text_max_dim or (y1 - y0) > self.text_max_dim:
                return False

        filled = sum(1 for path in cluster if path.fill_color is not None)
        return (filled / len(cluster)) >= self.text_min_fill_fraction

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
