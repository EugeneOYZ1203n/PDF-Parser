"""Vector stage: extracts and classifies vector drawing paths from a page.

extract_paths/separate_by_layer/separate_by_color/filter_layout_panels/
filter_background_fill/build_drawing_vectors are implemented. classify()
is implemented on top of the Clustering helper (spatial -> dimension ->
seq-number clustering, then a size/fill heuristic decides drawing vs.
text-candidate per cluster).
"""
from __future__ import annotations

from collections import Counter, defaultdict

import pymupdf as fitz

from rastervec.geometry import round_color
from rastervec.helpers.clustering import Clustering
from rastervec.logging_setup import get_logger
from rastervec.models import DrawingVector, Page, VectorPath

_LOG = get_logger("vector")

# Fraction of the page's area a fill color must cover to be treated as
# "the background" -- guards against filtering real content on pages that
# don't actually have one big background fill.
_BACKGROUND_AREA_FRACTION = 0.3


def _is_dashed(dashes: str | None) -> bool:
    # PyMuPDF's "dashes" is a PDF dash-array string like "[] 0" (no dash)
    # or "[3 2] 0" (dashed). An empty array means solid.
    if not dashes:
        return False
    return not dashes.strip().startswith("[]")


class Vector:
    """Extracts and classifies vector drawing paths from a page."""

    def __init__(
        self,
        *,
        spatial_threshold: float = 3.0,
        dimension_tolerance: float = 0.35,
        seq_max_gap: int = 3,
        text_max_dim: float = 30.0,
        text_min_fill_fraction: float = 0.5,
    ) -> None:
        self.spatial_threshold = spatial_threshold
        self.dimension_tolerance = dimension_tolerance
        self.seq_max_gap = seq_max_gap
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

    def filter_background_fill(
        self, paths: list[VectorPath], page: Page
    ) -> list[VectorPath]:
        area_by_color: dict[tuple, float] = defaultdict(float)
        for path in paths:
            if path.fill_color is None:
                continue
            x0, y0, x1, y1 = path.bbox
            area_by_color[path.fill_color] += max(x1 - x0, 0.0) * max(y1 - y0, 0.0)

        if not area_by_color:
            return list(paths)

        background_color = max(area_by_color, key=area_by_color.get)
        page_area = page.meta.width * page.meta.height
        if page_area <= 0 or area_by_color[background_color] < page_area * _BACKGROUND_AREA_FRACTION:
            return list(paths)

        kept = [path for path in paths if path.fill_color != background_color]
        _LOG.debug(
            "filter_background_fill: dropped background color %s, %d -> %d path(s)",
            background_color,
            len(paths),
            len(kept),
        )
        return kept

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def cluster_spatial(self, paths: list[VectorPath]) -> list[list[VectorPath]]:
        return self._clustering.cluster_spatial(
            paths, get_bbox=lambda p: p.bbox, threshold=self.spatial_threshold
        )

    def cluster_by_dimension(
        self, groups: list[list[VectorPath]]
    ) -> list[list[VectorPath]]:
        return self._clustering.cluster_by_dimension(
            groups, get_bbox=lambda p: p.bbox, tolerance=self.dimension_tolerance
        )

    def cluster_by_seq(self, groups: list[list[VectorPath]]) -> list[list[VectorPath]]:
        return self._clustering.cluster_by_seq(
            groups, get_seq=lambda p: p.seq, max_gap=self.seq_max_gap
        )

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
        self, paths: list[VectorPath]
    ) -> tuple[list[VectorPath], list[list[VectorPath]]]:
        spatial = self.cluster_spatial(paths)
        dimension = self.cluster_by_dimension(spatial)
        clusters = self.cluster_by_seq(dimension)
        return self.classify_clusters(clusters)

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
