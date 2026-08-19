"""Vector stage: extracts and classifies vector drawing paths from a page.

extract_paths/separate_by_layer/separate_by_color/build_drawing_vectors are
implemented, plus the classification pipeline: a caller-configurable
`list[StepConfig]` (rastervec/vector_classification/step_config.py) run in
order via cluster(). Each StepConfig is one of 3 kinds -- "filter" (checks
a metric against a condition; units failing become drawing, units passing
stay unclassified), "cluster" (splits existing groups into smaller ones by
a metric), "group" (merges nearby/similar existing groups by a metric) --
driven by a named metric from rastervec/vector_classification/metrics.py's
SCALAR_METRICS/PAIRWISE_METRICS registries. Every StepConfig instance is
self-contained (no shared name-keyed param dict), so two occurrences of the
same metric at different positions in the list are simply two independent
entries, each with its own threshold/params. DEFAULT_PIPELINE is the
default order, reproducing the original fixed 5-step pipeline (both
filters, one spatial clustering pass, both group filters). Every group
that survives the whole chain is a text candidate handed to OCR
(pipeline.py's ocr_text_clusters stage) -- there's no separate
drawing-vs-text heuristic; everything any filter step drops along the way
is drawing content, and OCR success/failure itself is the signal for
whether a given cluster was actually text. See cluster()'s docstring for
the full step semantics.
"""
from __future__ import annotations

from collections import defaultdict

import pymupdf as fitz

from rastervec.geometry import round_color, union_bbox
from rastervec.helpers.clustering import Clustering
from rastervec.logging_setup import get_logger
from rastervec.models import DrawingVector, Page, VectorPath
from rastervec.vector_classification import DEFAULT_PIPELINE, MetricContext, StepConfig, run_step

_LOG = get_logger("vector")


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

    @staticmethod
    def _compute_item_stats(
        paths: list[VectorPath],
    ) -> tuple[dict[int, int], dict[int, tuple[float, float, float, float]]]:
        """Per-`seq` (i.e. per original drawing/item) stats, computed once
        from the full path population entering cluster() -- stays stable
        for the whole chain regardless of how paths get regrouped along the
        way, so item-scoped metrics (item_path_count, item_bbox_*) always
        measure a path's *original* item, never whatever cluster it
        currently happens to sit in. Returns (path counts, aggregate
        bboxes), both keyed by seq."""
        by_seq: dict[int, list[VectorPath]] = defaultdict(list)
        for path in paths:
            by_seq[path.seq].append(path)
        counts = {seq: len(group) for seq, group in by_seq.items()}
        bboxes = {seq: union_bbox([p.bbox for p in group]) for seq, group in by_seq.items()}
        return counts, bboxes

    def cluster(
        self, paths: list[VectorPath], page: Page, steps: list[StepConfig] | None = None,
    ) -> tuple[list[list[list[VectorPath]]], list[list[list[VectorPath]]]]:
        """Runs `steps` (default DEFAULT_PIPELINE -- both filters, one
        spatial clustering pass, both group filters) in order, each step's
        input being the previous step's *kept* output. Every StepConfig
        instance is fully self-contained, so two instances of the same
        kind/metric at different positions never share state. Returns
        `(kept_snapshots, dropped_snapshots)`: one groups-list snapshot per
        step for each, in the same order as `steps` (`kept_snapshots[-1]`
        is the final surviving groups; `dropped_snapshots[i]` is only what
        step `i` itself dropped, not cumulative -- callers wanting the
        running total sum `dropped_snapshots[0:i+1]`), so callers (the
        debug app) can show/compare state after each individual step.

        Item-scoped metrics (item_path_count, item_bbox_min_side/
        item_bbox_max_side) measure each path's *original* item (see
        _compute_item_stats), computed once here from the full incoming
        `paths` before any step runs -- so their result never depends on
        what an earlier step already did to the grouping."""
        steps = list(steps) if steps else list(DEFAULT_PIPELINE)
        for step in steps:
            step.validate()
        item_counts, item_bboxes = self._compute_item_stats(paths)
        ctx = MetricContext(page=page, item_counts=item_counts, item_bboxes=item_bboxes)
        groups: list[list[VectorPath]] = [[p] for p in paths]
        kept_snapshots: list[list[list[VectorPath]]] = []
        dropped_snapshots: list[list[list[VectorPath]]] = []
        for step in steps:
            groups, dropped = run_step(step, groups, ctx, self._clustering)
            kept_snapshots.append(groups)
            dropped_snapshots.append(dropped)
        return kept_snapshots, dropped_snapshots

    def classify(
        self, paths: list[VectorPath], page: Page, steps: list[StepConfig] | None = None,
    ) -> list[list[VectorPath]]:
        """Runs cluster() with its default (or given) steps and returns
        just the final surviving groups -- a convenience wrapper for
        callers that don't need the drop bookkeeping (pipeline.py's own
        stage wiring calls cluster() directly instead, to keep every
        step's drops for the debug app and drawing_vectors)."""
        kept_snapshots, _dropped_snapshots = self.cluster(paths, page, steps)
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
