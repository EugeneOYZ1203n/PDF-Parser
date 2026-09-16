"""Ported from archive/raster_parser/rendering/pdf_render/reconstruct.py.
`Vector.type`/`.fill`/`.color`/`.rect`/`.items` map onto commons.models.Vector
unchanged, so this is a near-verbatim port, just retargeted at that type and
this folder's own config constants.
"""
from __future__ import annotations

from rastervec.commons.models import Vector
from rastervec.P3_Vector_Parsing.LegacyRecreation.config import (
    BOX_MIN_SIDE_PX,
    BOX_RULE_ASPECT_MIN,
    BOX_SQUARE_ASPECT_MAX,
    WHITE_FILL,
)

_TEXT_COLORS = (
    (0.0, 0.0, 0.0),
    (1.0, 1.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def filter_out_white_vectors(vectors: list[Vector]) -> list[Vector]:
    return [v for v in vectors if v.fill != WHITE_FILL]


def filter_fill_vectors(vectors: list[Vector]) -> list[Vector]:
    return [v for v in vectors if v.type == "f"]


def filter_fs_vectors(vectors: list[Vector]) -> list[Vector]:
    return [v for v in vectors if v.type == "fs"]


def is_box_or_rule(vec: Vector) -> bool:
    x0, y0, x1, y1 = vec.rect
    w, h = x1 - x0, y1 - y0
    if w == 0 or h == 0:
        return False
    if w < BOX_MIN_SIDE_PX and h < BOX_MIN_SIDE_PX:
        return False
    ratio = max(w, h) / min(w, h)
    is_rule = ratio > BOX_RULE_ASPECT_MIN
    is_square_ish = ratio < BOX_SQUARE_ASPECT_MAX
    is_not_text_fill = vec.fill is not None and vec.fill not in _TEXT_COLORS
    is_not_text_color = vec.color is not None and vec.color not in _TEXT_COLORS
    return is_rule or is_square_ish or is_not_text_fill or is_not_text_color


def _is_four_line_box(vec: Vector) -> bool:
    return len(vec.items) == 4 and all(seg[0] == "l" for seg in vec.items)


def _is_re_box(vec: Vector) -> bool:
    return len(vec.items) == 1 and vec.items[0][0] == "re"


def _is_quad_box(vec: Vector) -> bool:
    return len(vec.items) == 1 and vec.items[0][0] == "qu"


def is_box_v1(vec: Vector) -> bool:
    if vec.fill is None:
        return False
    if _is_re_box(vec) or _is_quad_box(vec):
        return True
    return _is_four_line_box(vec) and is_box_or_rule(vec)


def filter_out_boxes(vectors: list[Vector]) -> list[Vector]:
    return [v for v in vectors if not is_box_v1(v)]


def filter_text_vectors(
    vectors: list[Vector], *, include_fs: bool = False, only_fs: bool = False,
) -> list[Vector]:
    """Type-2 candidate vectors: filled glyph shapes, not page background,
    not a box/panel/rule."""
    if only_fs and include_fs:
        raise ValueError("filter_text_vectors: only_fs and include_fs are mutually exclusive")
    white_filtered = filter_out_white_vectors(vectors)
    if only_fs:
        fills = filter_fs_vectors(white_filtered)
    elif include_fs:
        fills = [v for v in white_filtered if v.type in ("f", "fs")]
    else:
        fills = filter_fill_vectors(white_filtered)
    return filter_out_boxes(fills)
