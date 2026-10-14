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

`from_pymupdf`/`to_pymupdf` are the fitz boundary: the only two places a
`Vector` ever touches a live `fitz`/dict object. Everything else in this
module (and everywhere `Vector` flows through the pipeline) stays
fitz-free plain data.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from raster_parser.commons.helpers.fitz_geometry import fitz_item, plain_item
from raster_parser.commons.helpers.geometry import round_color



def _to_tuples(obj):
    """Lists (from JSON) back to tuples, recursively."""
    if isinstance(obj, list):
        return tuple(_to_tuples(x) for x in obj)
    return obj


@dataclass
class Vector:
    type: str  # "f" | "s" | "fs" (get_drawings()['type'])
    items: list[tuple]
    color: tuple[float, ...] | None  # stroke color
    fill: tuple[float, ...] | None
    width: float | None
    dashes: str | None
    closePath: bool | None
    lineCap: tuple[int, int, int]  # PyMuPDF's (start, dash, end) caps; an int is widened to all three
    lineJoin: float  # PyMuPDF's raw value (0 miter / 1 round / 2 bevel)
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
    # Exact shape of every enclosing *non-rectangular* clip, outermost first:
    # `((items, even_odd), ...)`, `items` plain item tuples like `self.items`.
    # Rectangular clips are already exact in `scissor` and not repeated here.
    # Pipeline-added at extraction (`vector_extract`), not from a path dict.
    clips: tuple = ()
    # `get_drawings(extended=True)` nesting depth of this path inside clips/groups.
    level: int = 0

    def __post_init__(self) -> None:
        cap = self.lineCap
        if cap is None:
            cap = 0
        if isinstance(cap, (int, float)):
            cap = (int(cap),) * 3
        else:
            cap = tuple(int(c) for c in cap) or (0, 0, 0)
            cap = (cap + (cap[-1],) * 3)[:3]
        self.lineCap = cap

    @property
    def line_cap(self) -> int:
        """The start cap as one int -- what `Shape.finish(lineCap=...)` takes."""
        return self.lineCap[0]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.rect

    def to_dict(self) -> dict:
        """JSON-safe dict of every field (tuples become lists); `from_dict` restores it."""
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Vector":
        """Inverse of `to_dict`: tuples restored (items, colours, rects, clips); unknown keys ignored."""
        d = dict(d)
        for key in ("color", "fill", "rect", "scissor", "lineCap"):
            if d.get(key) is not None:
                d[key] = tuple(d[key])
        if d.get("items") is not None:
            d["items"] = [_to_tuples(item) for item in d["items"]]
        d["clips"] = tuple(
            (tuple(_to_tuples(item) for item in items), bool(even_odd))
            for items, even_odd in d.get("clips") or ()
        )
        names = {f.name for f in dataclasses.fields(Vector)}
        return Vector(**{k: v for k, v in d.items() if k in names})

    @staticmethod
    def from_pymupdf(drawing: dict, *, page_index: int, seq: int) -> "Vector":
        """Build a `Vector` from one raw `get_drawings()` entry. `seq` is
        used as `seqno` only when the drawing itself carries none (older
        PyMuPDF versions always do; kept as a defensive fallback)."""
        rect = drawing.get("rect")
        scissor = drawing.get("scissor")
        return Vector(
            type=drawing.get("type", ""),
            items=[plain_item(item) for item in drawing.get("items", [])],
            color=round_color(drawing.get("color")),
            fill=round_color(drawing.get("fill")),
            width=drawing.get("width"),
            dashes=drawing.get("dashes"),
            closePath=drawing.get("closePath"),
            lineCap=drawing.get("lineCap") or 0,
            lineJoin=float(drawing.get("lineJoin") or 0),
            even_odd=bool(drawing.get("even_odd", False)),
            stroke_opacity=drawing.get("stroke_opacity"),
            fill_opacity=drawing.get("fill_opacity"),
            layer=drawing.get("layer") or None,
            rect=(rect.x0, rect.y0, rect.x1, rect.y1) if rect is not None else (0.0, 0.0, 0.0, 0.0),
            scissor=(scissor.x0, scissor.y0, scissor.x1, scissor.y1) if scissor is not None else None,
            seqno=drawing.get("seqno", seq),
            blendmode=drawing.get("blendmode"),
            isolated=bool(drawing.get("isolated", False)),
            knockout=bool(drawing.get("knockout", False)),
            opacity=drawing.get("opacity"),
            page_index=page_index,
            level=int(drawing.get("level") or 0),
        )

    def to_pymupdf(self) -> dict:
        """The inverse of `from_pymupdf`: rebuild a `get_drawings()`-shaped
        dict, with real `fitz.Point`/`Rect`/`Quad` objects in `items`/
        `rect`/`scissor`. Note: a real `get_drawings()` dict from a given
        PyMuPDF version may not carry every key this method sets (e.g.
        `scissor`/`blendmode`/`isolated`/`knockout`/`opacity` are absent
        entirely on PyMuPDF versions that don't support transparency
        groups) -- compare with `.get(key)` against the fields you actually
        care about, not a blind `==` against a real dict, when round-
        tripping against live `get_drawings()` output."""
        import pymupdf as fitz

        return {
            "type": self.type,
            "items": [fitz_item(item) for item in self.items],
            "color": self.color,
            "fill": self.fill,
            "width": self.width,
            "dashes": self.dashes,
            "closePath": self.closePath,
            "lineCap": self.lineCap,
            "lineJoin": self.lineJoin,
            "even_odd": self.even_odd,
            "stroke_opacity": self.stroke_opacity,
            "fill_opacity": self.fill_opacity,
            "layer": self.layer,
            "rect": fitz.Rect(*self.rect),
            "scissor": fitz.Rect(*self.scissor) if self.scissor is not None else None,
            "seqno": self.seqno,
            "blendmode": self.blendmode,
            "isolated": self.isolated,
            "knockout": self.knockout,
            "opacity": self.opacity,
            "level": self.level,
        }
