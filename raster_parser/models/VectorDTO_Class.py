from __future__ import annotations

from typing import List, Any

import pymupdf
from pydantic import BaseModel, Field

from raster_parser.models import Vector
from raster_parser.serialisation.serialization import drawing_items_to_dicts, vector_items_to_dict




class VectorDTO(BaseModel):
    type: str
    x0: float
    y0: float
    x1: float
    y1: float
    items: List[dict[str, Any]] = Field(default_factory=list)
    fill: tuple[float, float, float] | None = None
    color: tuple[float, float, float] | None = None
    width: float = 0.0
    even_odd: bool = False
    close_path: bool = False
    dashes: str = ""
    fill_opacity: float = 1.0
    stroke_opacity: float = 1.0
    seqno: int = 0
    layer: str | None = None
    line_cap: tuple[int, int, int] | None = None
    line_join: float | None = None
    level: int = 0
    scissor: tuple[float, float, float, float] | None = None
    blendmode: str | None = None
    isolated: bool | None = None
    knockout: bool | None = None
    opacity: float | None = None

    @classmethod
    def from_vector(cls, vector: Vector) -> VectorDTO:
        fill = vector.fill
        color = vector.color
        return cls(
            type=vector.type,
            x0=float(vector.rect[0]),
            y0=float(vector.rect[1]),
            x1=float(vector.rect[2]),
            y1=float(vector.rect[3]),
            items=drawing_items_to_dicts(vector.items),
            fill=tuple(float(c) for c in fill) if fill is not None else None,
            color=tuple(float(c) for c in color) if color is not None else None,
            width=vector.width,
            even_odd=vector.even_odd,
            close_path=vector.close_path,
            dashes=vector.dashes,
            fill_opacity=vector.fill_opacity,
            stroke_opacity=vector.stroke_opacity,
            seqno=vector.seqno,
            layer=vector.layer,
            line_cap=vector.line_cap,
            line_join=vector.line_join,
            level=vector.level,
            scissor=tuple(float(x) for x in vector.scissor)
            if vector.scissor is not None
            else None,
            blendmode=vector.blendmode,
            isolated=vector.isolated,
            knockout=vector.knockout,
            opacity=vector.opacity,
        )

    def to_pymupdf_dict(self):
        out = {
            "type": self.type,
            "rect": pymupdf.Rect(self.x0, self.y0, self.x1, self.y1),
            "items": vector_items_to_dict(self.items),
            "fill": self.fill,
            "color": self.color,
            "width": self.width,
            "even_odd": self.even_odd,
            "close_path": self.close_path,
            "dashes": self.dashes,
            "fill_opacity": self.fill_opacity,
            "stroke_opacity": self.stroke_opacity,
            "opacity": self.opacity,
            "seqno": self.seqno,
            "layer": self.layer,
            "line_cap": self.line_cap,
            "line_join": self.line_join,
            "level": self.level,
            "scissor": self.scissor,
            "blendmode": self.blendmode,
            "isolated": self.isolated,
            "knockout": self.knockout,
        }
        return out