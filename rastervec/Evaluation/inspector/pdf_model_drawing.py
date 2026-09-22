"""`extract_annot_items`, `extract_drawing_items`, and `collect_drawing_colors`
-- annotations and vector-drawing extraction/color-discovery."""
from __future__ import annotations

import pymupdf as fitz

from rastervec.Evaluation.inspector.layers import OverlayItem, rgb_to_hex
from rastervec.Evaluation.inspector.pdf_model_core import (
    _line_length,
    _point_angle,
    _quad_angle,
    _quad_metadata,
    _rect_metadata,
    _round_color,
)


def extract_annot_items(
    page: "fitz.Page",
) -> list[OverlayItem]:

    items = []

    for annot in page.annots() or []:

        annot_type = (
            annot.type[1]
            if annot.type
            else ""
        )

        rect = fitz.Rect(
            annot.rect
        )

        metadata = {
            "type": annot_type,
            "xref": getattr(
                annot,
                "xref",
                None,
            ),
            **_rect_metadata(rect),
        }

        for attr in (
            "author",
            "contents",
            "subject",
            "opacity",
            "creationDate",
            "modDate",
        ):
            try:
                value = getattr(
                    annot,
                    attr,
                )

                if value is not None:
                    metadata[attr] = value

            except Exception:
                pass

        items.append(
            OverlayItem(
                bbox=rect,
                kind="annot",
                shape="rect",
                attrs={
                    "type": annot_type,
                },
                metadata=metadata,
                label=annot_type,
            )
        )

    return items


def _path_length(
    points: list[fitz.Point],
) -> float:

    if len(points) < 2:
        return 0.0

    total = 0.0

    for a, b in zip(
        points,
        points[1:],
    ):
        total += _line_length(
            a,
            b,
        )

    return total


def _drawing_common_metadata(
    drawing: dict,
    seqno: int,
) -> dict:

    stroke = _round_color(
        drawing.get("color")
    )

    fill = _round_color(
        drawing.get("fill")
    )

    metadata = {
        "seqno": seqno,

        "path_type": drawing.get(
            "type",
            "",
        ),

        "stroke_color": stroke,
        "fill_color": fill,

        "stroke_width": drawing.get(
            "width",
            None,
        ),

        "stroke_opacity": drawing.get(
            "stroke_opacity",
            None,
        ),

        "fill_opacity": drawing.get(
            "fill_opacity",
            None,
        ),

        "line_cap": drawing.get(
            "lineCap",
            None,
        ),

        "line_join": drawing.get(
            "lineJoin",
            None,
        ),

        "close_path": drawing.get(
            "closePath",
            None,
        ),

        "even_odd": drawing.get(
            "even_odd",
            None,
        ),

        "dashes": drawing.get(
            "dashes",
            None,
        ),

        "layer": drawing.get(
            "layer",
            None,
        ),
    }

    rect = drawing.get(
        "rect"
    )

    if rect is not None:
        rect = fitz.Rect(rect)
        metadata.update(
            {
                "drawing_bbox": (
                    round(rect.x0, 2),
                    round(rect.y0, 2),
                    round(rect.x1, 2),
                    round(rect.y1, 2),
                ),
                "drawing_width": round(
                    rect.width,
                    3,
                ),
                "drawing_height": round(
                    rect.height,
                    3,
                ),
                "drawing_area": round(
                    rect.get_area(),
                    3,
                ),
            }
        )

    return metadata


def extract_drawing_items(
    page: "fitz.Page",
) -> list[OverlayItem]:

    items = []

    for seqno, drawing in enumerate(
        page.get_drawings()
    ):

        path_type = drawing.get(
            "type",
            "",
        )

        stroke_color = _round_color(
            drawing.get("color")
        )

        fill_color = _round_color(
            drawing.get("fill")
        )

        common_attrs = {
            "path_type": path_type,
            "stroke_color": stroke_color,
            "fill_color": fill_color,
            "seqno": seqno,
        }

        common_metadata = (
            _drawing_common_metadata(
                drawing,
                seqno,
            )
        )

        operations = drawing.get(
            "items",
            [],
        )

        operation_count = len(
            operations
        )


        for item_index, it in enumerate(
            operations
        ):

            op = it[0]

            base_metadata = {
                **common_metadata,
                "item_index": item_index,
                "operation_count": operation_count,
                "operation": op,
            }


            if op == "l":

                p1 = fitz.Point(
                    it[1]
                )

                p2 = fitz.Point(
                    it[2]
                )

                angle = _point_angle(
                    p1,
                    p2,
                )

                length = _line_length(
                    p1,
                    p2,
                )

                bbox = fitz.Rect(
                    min(p1.x, p2.x),
                    min(p1.y, p2.y),
                    max(p1.x, p2.x),
                    max(p1.y, p2.y),
                )

                metadata = {
                    **base_metadata,
                    "angle": round(
                        angle,
                        3,
                    ),
                    "length": round(
                        length,
                        3,
                    ),
                    "start": (
                        round(p1.x, 2),
                        round(p1.y, 2),
                    ),
                    "end": (
                        round(p2.x, 2),
                        round(p2.y, 2),
                    ),
                    **_rect_metadata(bbox),
                }

                items.append(
                    OverlayItem(
                        bbox=bbox,
                        kind="l",
                        shape="line",
                        points=[
                            p1,
                            p2,
                        ],
                        attrs={
                            **common_attrs,
                            "kind": "l",
                            "angle": angle,
                            "length": length,
                            "item_index": item_index,
                        },
                        metadata=metadata,
                    )
                )


            elif op == "re":

                rect = fitz.Rect(
                    it[1]
                )

                metadata = {
                    **base_metadata,
                    **_rect_metadata(rect),
                }

                items.append(
                    OverlayItem(
                        bbox=rect,
                        kind="re",
                        shape="rect",
                        attrs={
                            **common_attrs,
                            "kind": "re",
                            "item_index": item_index,
                        },
                        metadata=metadata,
                    )
                )


            elif op == "qu":

                quad = fitz.Quad(
                    it[1]
                )

                metadata = {
                    **base_metadata,
                    **_quad_metadata(
                        quad
                    ),
                    "corners": [
                        (
                            round(quad.ul.x, 2),
                            round(quad.ul.y, 2),
                        ),
                        (
                            round(quad.ur.x, 2),
                            round(quad.ur.y, 2),
                        ),
                        (
                            round(quad.lr.x, 2),
                            round(quad.lr.y, 2),
                        ),
                        (
                            round(quad.ll.x, 2),
                            round(quad.ll.y, 2),
                        ),
                    ],
                }

                items.append(
                    OverlayItem(
                        bbox=quad.rect,
                        quad=quad,
                        kind="qu",
                        shape="quad",
                        attrs={
                            **common_attrs,
                            "kind": "qu",
                            "angle": _quad_angle(
                                quad
                            ),
                            "item_index": item_index,
                        },
                        metadata=metadata,
                    )
                )


            elif op == "c":

                p1 = fitz.Point(
                    it[1]
                )
                p2 = fitz.Point(
                    it[2]
                )
                p3 = fitz.Point(
                    it[3]
                )
                p4 = fitz.Point(
                    it[4]
                )

                points = [
                    p1,
                    p2,
                    p3,
                    p4,
                ]

                xs = [
                    p.x
                    for p in points
                ]

                ys = [
                    p.y
                    for p in points
                ]

                bbox = fitz.Rect(
                    min(xs),
                    min(ys),
                    max(xs),
                    max(ys),
                )

                metadata = {
                    **base_metadata,
                    "control_points": [
                        (
                            round(
                                p.x,
                                2,
                            ),
                            round(
                                p.y,
                                2,
                            ),
                        )
                        for p in points
                    ],
                    "control_polygon_length": round(
                        _path_length(
                            points
                        ),
                        3,
                    ),
                    **_rect_metadata(bbox),
                }

                items.append(
                    OverlayItem(
                        bbox=bbox,
                        kind="c",
                        shape="polygon",
                        points=points,
                        attrs={
                            **common_attrs,
                            "kind": "c",
                            "item_index": item_index,
                        },
                        metadata=metadata,
                    )
                )

    return items


def collect_drawing_colors(
    page: "fitz.Page",
) -> tuple[
    dict[str, str],
    dict[str, str],
]:

    strokes = {}
    fills = {}

    saw_no_stroke = False
    saw_no_fill = False

    for drawing in page.get_drawings():

        stroke = _round_color(
            drawing.get("color")
        )

        fill = _round_color(
            drawing.get("fill")
        )

        if stroke:
            strokes.setdefault(
                stroke,
                rgb_to_hex(stroke),
            )
        else:
            saw_no_stroke = True

        if fill:
            fills.setdefault(
                fill,
                rgb_to_hex(fill),
            )
        else:
            saw_no_fill = True

    if saw_no_stroke:
        strokes.setdefault(
            "__none__",
            "(none)",
        )

    if saw_no_fill:
        fills.setdefault(
            "__none__",
            "(none)",
        )

    return strokes, fills
