"""Vector stage: extracts vector drawing paths from a page.

extract_paths (plus layer/color separation, in Layer_Color_Separation/) is
this module's concern. Classification of extracted paths into text
candidates vs. drawing content lives in rastervec/Vector_Classification/
instead -- see that package's classification.py for the fixed 12-step
pipeline and Glossary.md for group/cluster terminology.
"""
from __future__ import annotations

import pymupdf as fitz

from rastervec.helpers.geometry import round_color
from rastervec.logging_setup import get_logger
from rastervec.models import Page, VectorPath, VectorRecord
from rastervec.Vector.Layer_Color_Separation.layer_color_separation import (
    separate_by_color,
    separate_by_layer,
)

_LOG = get_logger("vector")


class Vector:
    """Extracts vector drawing paths from a page, and separates them by
    layer/color (see Layer_Color_Separation/layer_color_separation.py)."""

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

    def extract_records(self, page: Page) -> list[VectorRecord]:
        """Richer, additive counterpart to extract_paths -- one VectorRecord
        per raw get_drawings() drawing, carrying every drawing-level field
        extract_paths's `common` dict drops (even_odd, line_cap, line_join,
        real seqno, rect, scissor, blendmode, isolated, knockout, opacity).
        `groups`/`role` stay None here -- populated later, once this
        drawing's paths have been through Vector_Classification, by whatever
        wires cluster lineage into VectorRecord (see classification.py).
        Never replaces extract_paths/VectorPath for existing consumers."""
        fitz_page = page.fitz_page
        page_index = page.meta.index
        records: list[VectorRecord] = []

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

            items: list[VectorPath] = []
            for item_index, item in enumerate(drawing.get("items", [])):
                path = self._extract_item(item, item_index, common)
                if path is not None:
                    items.append(path)

            rect = drawing.get("rect")
            rect_tuple = (
                (rect.x0, rect.y0, rect.x1, rect.y1)
                if rect is not None
                else self._union_bbox(items)
            )
            scissor = drawing.get("scissor")
            scissor_tuple = (
                (scissor.x0, scissor.y0, scissor.x1, scissor.y1)
                if scissor is not None
                else None
            )

            records.append(
                VectorRecord(
                    items=items,
                    bbox=self._union_bbox(items) or rect_tuple,
                    stroke_color=stroke_color,
                    fill_color=fill_color,
                    stroke_width=stroke_width,
                    dashed=bool(dashes),
                    page_index=page_index,
                    even_odd=bool(drawing.get("even_odd", False)),
                    line_cap=drawing.get("lineCap", 0) or 0,
                    line_join=drawing.get("lineJoin", 0) or 0,
                    seqno=drawing.get("seqno", seq),
                    rect=rect_tuple,
                    scissor=scissor_tuple,
                    blendmode=drawing.get("blendmode"),
                    isolated=bool(drawing.get("isolated", False)),
                    knockout=bool(drawing.get("knockout", False)),
                    opacity=drawing.get("opacity"),
                )
            )

        return records

    def _union_bbox(
        self, items: list[VectorPath]
    ) -> tuple[float, float, float, float] | None:
        if not items:
            return None
        x0 = min(p.bbox[0] for p in items)
        y0 = min(p.bbox[1] for p in items)
        x1 = max(p.bbox[2] for p in items)
        y1 = max(p.bbox[3] for p in items)
        return (x0, y0, x1, y1)

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

    def separate_by_layer(self, paths: list[VectorPath]) -> dict[str, list[VectorPath]]:
        return separate_by_layer(paths)

    def separate_by_color(self, paths: list[VectorPath]) -> dict[tuple, list[VectorPath]]:
        return separate_by_color(paths)
