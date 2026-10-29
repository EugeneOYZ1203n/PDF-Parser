"""Reverse JSON-safe DTOs and drawing items back to pipeline-native types."""

from __future__ import annotations

import re
from typing import Any

import pymupdf as fitz

from raster_parser.models import Text
from raster_parser.models import Vector
from raster_parser.models import GlyphOrientation
from raster_parser.models import TextDTO, NativePDFElements, VectorDTO

_POINT_RE = re.compile(r"Point\(([^,]+),\s*([^)]+)\)")

_ORIENTATION_FROM_DTO = {
    "horizontal": GlyphOrientation.HORIZONTAL,
    "vertical": GlyphOrientation.VERTICAL,
    "other": GlyphOrientation.OTHER,
}


def _quad_from_string(value: str) -> fitz.Quad:
    points = _POINT_RE.findall(value)
    if len(points) != 4:
        raise ValueError(f"unsupported quad string: {value!r}")
    corners = [fitz.Point(float(x), float(y)) for x, y in points]
    return fitz.Quad(corners[0], corners[1], corners[2], corners[3])


def _coords_to_fitz(value: list | tuple | str) -> fitz.Point | fitz.Rect | fitz.Quad:
    if isinstance(value, str):
        if value.startswith("Quad("):
            return _quad_from_string(value)
        raise ValueError(f"unsupported coordinate string: {value!r}")
    if len(value) == 2:
        return fitz.Point(float(value[0]), float(value[1]))
    if len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
        return fitz.Rect(float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    if len(value) == 4 and all(isinstance(v, (list, tuple)) for v in value):
        points = [fitz.Point(float(p[0]), float(p[1])) for p in value]
        return fitz.Quad(points[0], points[1], points[2], points[3])
    raise ValueError(f"unsupported coordinate list: {value!r}")


def drawing_item_from_dict(item: dict[str, Any]) -> tuple:
    """Rebuild one ``get_drawings()`` path segment from a serialized dict."""
    op = item["op"]
    args = item.get("args", [])
    if op == "re":
        return ("re", _coords_to_fitz(args[0]))
    if op == "l":
        return ("l", _coords_to_fitz(args[0]), _coords_to_fitz(args[1]))
    if op == "c":
        if len(args) != 4:
            raise ValueError(f"bezier expects 4 points, got {args!r}")
        return (
            "c",
            _coords_to_fitz(args[0]),
            _coords_to_fitz(args[1]),
            _coords_to_fitz(args[2]),
            _coords_to_fitz(args[3]),
        )
    if op == "qu":
        if len(args) == 1:
            return ("qu", _coords_to_fitz(args[0]))
        if len(args) == 4:
            return ("qu", _coords_to_fitz(args))
        raise ValueError(f"quad expects 1 or 4 args, got {args!r}")
    raise ValueError(f"unsupported drawing op {op!r}")


def drawing_items_from_dicts(items: list[dict[str, Any]]) -> list[tuple]:
    return [drawing_item_from_dict(item) for item in items]


def deserialize_text(dto: TextDTO) -> Text:
    """Convert one :class:`TextDTO` to pipeline :class:`Text`."""
    orientation = _ORIENTATION_FROM_DTO.get(
        dto.glyph_orientation.lower(),
        GlyphOrientation.HORIZONTAL,
    )
    return Text(
        dto.x0,
        dto.y0,
        dto.x1,
        dto.y1,
        dto.word,
        dto.block_no,
        dto.line_no,
        dto.word_no,
        rotate=dto.rotate,
        glyph_orientation=orientation,
    )


def deserialize_vector(dto: VectorDTO) -> Vector:
    """Convert one :class:`VectorDTO` to pipeline :class:`Vector`."""
    return Vector(
        items=drawing_items_from_dicts(dto.items),
        type=dto.type,
        rect=(dto.x0, dto.y0, dto.x1, dto.y1),
        fill=dto.fill,
        color=dto.color,
        width=dto.width,
        even_odd=dto.even_odd,
        close_path=dto.close_path,
        dashes=dto.dashes,
        fill_opacity=dto.fill_opacity,
        stroke_opacity=dto.stroke_opacity,
        seqno=dto.seqno,
        layer=dto.layer,
        line_cap=dto.line_cap,
        line_join=dto.line_join,
        level=dto.level,
        scissor=dto.scissor,
        blendmode=dto.blendmode,
        isolated=dto.isolated,
        knockout=dto.knockout,
        opacity=dto.opacity,
    )


def deserialize(
    words: list[TextDTO],
    vectors: list[VectorDTO],
) -> tuple[list[Text], list[Vector]]:
    """Convert DTO lists to pipeline-native :class:`Text` and :class:`Vector` lists."""
    return (
        [deserialize_text(dto) for dto in words],
        [deserialize_vector(dto) for dto in vectors],
    )


def deserialize_output(output: NativePDFElements) -> tuple[list[Text], list[Vector]]:
    """Convert a full :class:`NativePDFElements` to native word and vector lists."""
    return deserialize(output.words, output.vectors)
