"""One whole `get_drawings()` drawing -- never decomposed into its own items.

`items` is stored as PyMuPDF's own raw item tuple shape (`(kind,
*geometry)`), fitz objects converted to plain tuples by
`helpers.fitz_geometry.plain_item` at extraction time but the tuple
*structure* otherwise untouched -- a Vector "perfectly mimics"
`get_drawings()`. Any per-item geometry a filter needs (bbox, points) is
computed on demand via `helpers.geometry.item_bbox`/`item_points`; nothing
is precomputed or cached here. `helpers.geometry.transform_vector` builds a
translated+rotated copy without ever touching item structure -- used to
normalize a Segment's Vectors to a canonical frame and to restore them.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Vector:
    type: str  # "f" | "s" | "fs" (get_drawings()['type'])
    items: list[tuple]
    color: tuple[float, ...] | None  # stroke color
    fill: tuple[float, ...] | None
    width: float | None
    dashes: str | None
    closePath: bool | None
    lineCap: int
    lineJoin: int
    even_odd: bool
    stroke_opacity: float | None
    fill_opacity: float | None
    layer: str | None
    rect: tuple[float, float, float, float]  # this drawing's own bbox, page space
    scissor: tuple[float, float, float, float] | None
    seqno: int  # PyMuPDF's own seqno (content-stream draw order)
    blendmode: str | None
    isolated: bool
    knockout: bool
    opacity: float | None
    page_index: int  # pipeline-added, not from PyMuPDF

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.rect
