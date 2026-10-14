"""Step 1 -- fill-glyph candidates: filled (`"f"`) vectors that aren't white and aren't layout
boxes or rules. Everything else is drawing.

Copied from `rendering/pdf_render/reconstruct.py` (`filter_text_vectors` and the filters it
calls), keeping only the default path: the `include_fs` / `only_fs` options were never used.
"""
from __future__ import annotations

from raster_parser.commons.models import Vector

WHITE_FILL = (1.0, 1.0, 1.0)


def filter_out_white_vectors(vectors: list[Vector]) -> list[Vector]:
    return [v for v in vectors if v.fill != WHITE_FILL]


def filter_fill_vectors(vectors: list[Vector]) -> list[Vector]:
    return [v for v in vectors if v.type == "f"]


def is_box_or_rule(vec: Vector) -> bool:
    x0, y0, x1, y1 = vec.rect
    w = x1 - x0
    h = y1 - y0
    if w == 0 or h == 0:
        return False
    if w < 10 and h < 10:
        return False
    ratio = max(w, h) / min(w, h)
    is_rule = ratio > 10
    is_square_ish = ratio < 3
    is_not_text_fill = vec.fill is not None and vec.fill not in (
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    is_not_text_color = vec.color is not None and vec.color not in (
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
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
    # MuPDF rectangle ops are layout/panel fills, never Type-2 glyph ink.
    if _is_re_box(vec) or _is_quad_box(vec):
        return True
    return _is_four_line_box(vec) and is_box_or_rule(vec)


def filter_out_boxes(vectors: list[Vector]) -> list[Vector]:
    return [v for v in vectors if not is_box_v1(v)]


def filter_text_vectors(vectors: list[Vector]) -> list[Vector]:
    """The fill-glyph candidates among `vectors`."""
    white_filtered = filter_out_white_vectors(vectors)
    fills = filter_fill_vectors(white_filtered)
    fills = filter_out_boxes(fills)
    return fills
