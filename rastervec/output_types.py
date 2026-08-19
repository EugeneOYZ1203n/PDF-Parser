"""Standardized output DTOs: pydantic models mirroring what a single
PyMuPDF `get_text("words")` word / `get_drawings()` drawing look like,
built from rastervec's own `models.py` dataclasses (`TextWord`,
`DrawingVector`) rather than raw PyMuPDF output -- these are the
serialization/export shape for a page's final native text + vector output
(e.g. for evaluation.py's eventual serialization boundary), not a
replacement for the dataclasses used mid-pipeline (clustering, rendering,
the debug app all keep using TextWord/VectorPath/DrawingVector directly).
"""
from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, Field

from rastervec.models import DrawingVector, TextWord, VectorPath


class TextDTO(BaseModel):
    """Mirrors a pymupdf `get_text("words")` word's shape (x0/y0/x1/y1/
    word), extended with the extra real fields `TextWord` actually carries
    (`angle`/`font`/`font_size`/`seq`) that a raw pymupdf word tuple
    doesn't have. `block_no`/`line_no`/`word_no` are pymupdf-word-shape
    fields `TextWord` has no equivalent for (Native.extract_text only
    assigns a flat `seq`, not block/line/word grouping) -- they default to
    0 rather than being fabricated. `rotate` is `angle` rounded to the
    nearest quarter-turn, for callers wanting pymupdf's own quantized
    convention; `angle` carries the exact value."""

    x0: float
    y0: float
    x1: float
    y1: float
    word: str
    block_no: int = 0
    line_no: int = 0
    word_no: int = 0
    rotate: int = 0
    glyph_orientation: str = "horizontal"
    angle: float = 0.0
    font: str = ""
    font_size: float = 0.0
    seq: int = 0

    @classmethod
    def from_text_word(cls, word: TextWord) -> "TextDTO":
        x0, y0, x1, y1 = word.bbox
        return cls(
            x0=x0, y0=y0, x1=x1, y1=y1, word=word.text,
            rotate=round(word.angle / 90.0) % 4 * 90,
            glyph_orientation="horizontal" if abs(word.angle % 180) < 45 else "vertical",
            angle=word.angle, font=word.font, font_size=word.font_size, seq=word.seq,
        )


def _vector_path_to_item_dict(path: VectorPath) -> dict[str, Any]:
    """Mirrors one raw pymupdf get_drawings() item tuple's shape (e.g.
    ("l", p1, p2) / ("re", rect) / ("qu", quad) / ("c", p1,p2,p3,p4)) as a
    labeled dict, sourced from a VectorPath instead of re-deriving
    fitz.Point/Rect/Quad objects."""
    return {
        "kind": path.kind,
        "item_index": path.item_index,
        "points": path.points,
        "bbox": path.bbox,
        "fill_rule": path.fill_rule,
        "closed": path.closed,
    }


class VectorDTO(BaseModel):
    """Mirrors a pymupdf `get_drawings()` drawing's shape (an aggregate
    bbox/style plus an `items` list of its primitive geometry), sourced
    from a `DrawingVector` -- fields `DrawingVector` doesn't carry
    (`even_odd`/`fill_opacity`/`stroke_opacity`/`line_cap`/`line_join`/
    `scissor`/`blendmode`/`isolated`/`knockout`/`opacity`) are dropped
    rather than fabricated; per-path fields like opacity/closed are
    available per-item in `items` instead."""

    x0: float
    y0: float
    x1: float
    y1: float
    items: List[dict[str, Any]] = Field(default_factory=list)
    fill: tuple[float, ...] | None = None
    color: tuple[float, ...] | None = None
    width: float = 0.0
    dashes: str = ""
    seqno: int = 0
    layer: str | None = None
    page_index: int = 0

    @classmethod
    def from_drawing_vector(cls, dv: DrawingVector) -> "VectorDTO":
        x0, y0, x1, y1 = dv.bbox
        first = dv.paths[0] if dv.paths else None
        return cls(
            x0=x0, y0=y0, x1=x1, y1=y1,
            items=[_vector_path_to_item_dict(p) for p in dv.paths],
            fill=dv.fill_color, color=dv.stroke_color, width=dv.stroke_width or 0.0,
            dashes="[3 2] 0" if dv.dashed else "[] 0",
            seqno=first.seq if first is not None else 0,
            layer=first.layer if first is not None else None,
            page_index=dv.page_index,
        )


class NativePDFElements(BaseModel):
    """Serializable native text + vector-drawing output from one PDF page."""

    words: List[TextDTO]
    vectors: List[VectorDTO]

    @classmethod
    def from_extract(
        cls, words: list[TextWord], drawings: list[DrawingVector],
    ) -> "NativePDFElements":
        return cls(
            words=[TextDTO.from_text_word(w) for w in words],
            vectors=[VectorDTO.from_drawing_vector(dv) for dv in drawings],
        )
