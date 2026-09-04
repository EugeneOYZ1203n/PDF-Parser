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

from pydantic import BaseModel, Field, PrivateAttr

from rastervec.models import DrawingVector, TextWord, VectorPath, VectorRecord


class TextDTO(BaseModel):
    """Mirrors a pymupdf `get_text("words")` word's shape (x0/y0/x1/y1/
    word/block_no/line_no/word_no), extended with the extra real fields
    `TextWord` carries (`angle`/`font`/`font_size`/`seq`) that a raw pymupdf
    word tuple doesn't. `rotate` is `angle` rounded to the nearest
    quarter-turn, for callers wanting pymupdf's own quantized convention;
    `angle` carries the exact value. `get_text_object()` returns the source
    `TextWord`."""

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

    _source: TextWord = PrivateAttr()

    @classmethod
    def from_text_word(cls, word: TextWord) -> "TextDTO":
        x0, y0, x1, y1 = word.bbox
        dto = cls(
            x0=x0, y0=y0, x1=x1, y1=y1, word=word.text,
            block_no=word.block_no, line_no=word.line_no, word_no=word.word_no,
            rotate=round(word.angle / 90.0) % 4 * 90,
            glyph_orientation="horizontal" if abs(word.angle % 180) < 45 else "vertical",
            angle=word.angle, font=word.font, font_size=word.font_size, seq=word.seq,
        )
        dto._source = word
        return dto

    def get_text_object(self) -> TextWord:
        return self._source


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

    _source: VectorRecord = PrivateAttr()

    @classmethod
    def from_drawing_vector(
        cls, dv: DrawingVector, record: VectorRecord | None = None,
    ) -> "VectorDTO":
        """`record`, if given, is the richer VectorRecord this drawing was
        built alongside -- stashed on `_source` for get_vector_object(). If
        omitted (e.g. from_extract's own call site, which only has
        DrawingVectors), a VectorRecord is synthesized from `dv` with the
        drawing-level fields DrawingVector doesn't carry defaulted to
        false/0/None, so get_vector_object() always returns something
        valid."""
        x0, y0, x1, y1 = dv.bbox
        first = dv.paths[0] if dv.paths else None
        dto = cls(
            x0=x0, y0=y0, x1=x1, y1=y1,
            items=[_vector_path_to_item_dict(p) for p in dv.paths],
            fill=dv.fill_color, color=dv.stroke_color, width=dv.stroke_width or 0.0,
            dashes="[3 2] 0" if dv.dashed else "[] 0",
            seqno=first.seq if first is not None else 0,
            layer=first.layer if first is not None else None,
            page_index=dv.page_index,
        )
        dto._source = record if record is not None else VectorRecord(
            items=dv.paths, bbox=dv.bbox, stroke_color=dv.stroke_color,
            fill_color=dv.fill_color, stroke_width=dv.stroke_width, dashed=dv.dashed,
            page_index=dv.page_index, even_odd=False, line_cap=0, line_join=0,
            seqno=first.seq if first is not None else 0, rect=dv.bbox, scissor=None,
            blendmode=None, isolated=False, knockout=False, opacity=None,
        )
        return dto

    def get_vector_object(self) -> VectorRecord:
        return self._source


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
