from __future__ import annotations

from itertools import groupby
from math import degrees, atan2, hypot, sqrt

import pymupdf as fitz

from rastervec.Evaluation.inspector.layers import OverlayItem, rgb_to_hex


class PdfDocument:
    """Thin wrapper around a PyMuPDF document."""

    def __init__(self, path: str):
        self.path = path
        self.doc = fitz.open(path)

    @property
    def page_count(self) -> int:
        return self.doc.page_count

    def get_page(self, n: int) -> "fitz.Page":
        return self.doc[n]

    def render_pixmap(
        self,
        n: int,
        zoom: float,
    ) -> "fitz.Pixmap":
        """
        Render a page.

        get_pixmap() respects the page's /Rotate attribute.
        """

        page = self.get_page(n)

        matrix = fitz.Matrix(
            zoom,
            zoom,
        )

        return page.get_pixmap(
            matrix=matrix,
        )

    def close(self) -> None:
        self.doc.close()


def _round_color(color):
    if not color:
        return None

    return tuple(
        round(c, 3)
        for c in color
    )


def _point_angle(
    p1: fitz.Point,
    p2: fitz.Point,
) -> float:
    dx = p2.x - p1.x
    dy = p2.y - p1.y

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0

    return degrees(
        atan2(dy, dx)
    )


def _line_length(
    p1: fitz.Point,
    p2: fitz.Point,
) -> float:
    return hypot(
        p2.x - p1.x,
        p2.y - p1.y,
    )


def _quad_angle(
    quad: fitz.Quad,
) -> float:
    return _point_angle(
        quad.ul,
        quad.ur,
    )


def _quad_metadata(
    quad: fitz.Quad,
) -> dict:
    width = _line_length(
        quad.ul,
        quad.ur,
    )

    height = _line_length(
        quad.ul,
        quad.ll,
    )

    return {
        "angle": round(
            _quad_angle(quad),
            3,
        ),
        "width": round(
            width,
            3,
        ),
        "height": round(
            height,
            3,
        ),
    }


def _rect_metadata(
    rect: fitz.Rect,
) -> dict:
    return {
        "x": round(rect.x0, 3),
        "y": round(rect.y0, 3),
        "width": round(rect.width, 3),
        "height": round(rect.height, 3),
        "area": round(rect.get_area(), 3),
        "aspect_ratio": round(
            rect.width / rect.height,
            4,
        )
        if abs(rect.height) > 1e-9
        else None,
    }


def _format_matrix(matrix) -> tuple:
    """
    Convert a PyMuPDF Matrix to a compact tuple.

    Matrix values are:

        a b c d e f
    """

    return (
        round(matrix.a, 5),
        round(matrix.b, 5),
        round(matrix.c, 5),
        round(matrix.d, 5),
        round(matrix.e, 5),
        round(matrix.f, 5),
    )


def _matrix_rotation(matrix) -> float:
    """
    Estimate visual rotation from the matrix's x-axis.
    """

    return degrees(
        atan2(
            matrix.b,
            matrix.a,
        )
    )


def _matrix_scale(matrix) -> tuple[float, float]:
    """
    Estimate X/Y scale from a transformation matrix.
    """

    sx = sqrt(
        matrix.a * matrix.a
        + matrix.b * matrix.b
    )

    sy = sqrt(
        matrix.c * matrix.c
        + matrix.d * matrix.d
    )

    return (
        sx,
        sy,
    )


def extract_text_items(
    page: "fitz.Page",
) -> list[OverlayItem]:
    """
    Extract text words while retaining span orientation metadata.
    """

    words = page.get_text(
        "words",
        sort=False,
    )

    spans = []

    text_dict = page.get_text(
        "dict",
        sort=False,
    )

    for block in text_dict.get(
        "blocks",
        [],
    ):
        if block.get("type") != 0:
            continue

        for line in block.get(
            "lines",
            [],
        ):
            line_dir = line.get(
                "dir",
                (1.0, 0.0),
            )

            for span in line.get(
                "spans",
                [],
            ):
                text = span.get(
                    "text",
                    "",
                )

                if not text.strip():
                    continue

                bbox = fitz.Rect(
                    span.get(
                        "bbox",
                        (0, 0, 0, 0),
                    )
                )

                spans.append(
                    {
                        "bbox": bbox,
                        "text": text,
                        "font": span.get(
                            "font",
                            "",
                        ),
                        "size": span.get(
                            "size",
                            0,
                        ),
                        "flags": span.get(
                            "flags",
                            0,
                        ),
                        "color": span.get(
                            "color",
                            None,
                        ),
                        "origin": span.get(
                            "origin",
                            None,
                        ),
                        "dir": line_dir,
                        "ascender": span.get(
                            "ascender",
                            None,
                        ),
                        "descender": span.get(
                            "descender",
                            None,
                        ),
                    }
                )

    words = sorted(
        words,
        key=lambda w: (
            round(w[1]),
            w[0],
        ),
    )

    items = []

    for _, group in groupby(
        words,
        key=lambda w: round(w[1]),
    ):
        for word in sorted(
            group,
            key=lambda w: w[0],
        ):
            x0, y0, x1, y1, text = (
                word[0],
                word[1],
                word[2],
                word[3],
                word[4],
            )

            bbox = fitz.Rect(
                x0,
                y0,
                x1,
                y1,
            )

            span = _find_best_span(
                bbox,
                spans,
            )

            if span is None:
                items.append(
                    OverlayItem(
                        bbox=bbox,
                        kind="word",
                        shape="rect",
                        label=text,
                        metadata={
                            "text": text,
                            "angle": 0.0,
                            "orientation_source": "fallback",
                        },
                    )
                )

                continue

            dx, dy = span["dir"]

            angle = degrees(
                atan2(
                    dy,
                    dx,
                )
            )

            quad = _make_oriented_quad(
                bbox,
                dx,
                dy,
            )

            metadata = {
                "text": text,
                "angle": round(angle, 3),
                "direction": (
                    round(dx, 5),
                    round(dy, 5),
                ),
                "font": span["font"],
                "font_size": round(
                    span["size"],
                    3,
                ),
                "flags": span["flags"],
                "origin": span["origin"],
                "color": span["color"],
                "ascender": span["ascender"],
                "descender": span["descender"],
                "orientation_source": "text-span",
                **_rect_metadata(bbox),
            }

            items.append(
                OverlayItem(
                    bbox=bbox,
                    quad=quad,
                    angle=angle,
                    kind="word",
                    shape="quad",
                    label=text,
                    attrs={
                        "angle": angle,
                        "font": span["font"],
                        "font_size": span["size"],
                    },
                    metadata=metadata,
                )
            )

    return items


def _find_best_span(
    bbox: fitz.Rect,
    spans: list[dict],
) -> dict | None:

    if not spans:
        return None

    best = None
    best_score = -1.0

    word_area = max(
        bbox.get_area(),
        1e-9,
    )

    for span in spans:

        span_bbox = span["bbox"]

        intersection = bbox & span_bbox

        if intersection.is_empty:
            continue

        score = (
            intersection.get_area()
            / word_area
        )

        if score > best_score:
            best_score = score
            best = span

    return best


def _make_oriented_quad(
    bbox: fitz.Rect,
    dx: float,
    dy: float,
) -> fitz.Quad:
    """Build a quad around bbox, oriented along the text direction (dx, dy).

    bbox.width/height are the *axis-aligned* extents, which only match the
    text's along-direction/normal-direction extents when the text is
    horizontal. For rotated text they can be swapped or otherwise wrong, so
    the bbox corners are projected onto the (dx, dy) / normal axes instead
    to recover the correct along/normal extents regardless of orientation.
    """

    length = hypot(
        dx,
        dy,
    )

    if length < 1e-9:
        dx = 1.0
        dy = 0.0
        length = 1.0

    dx /= length
    dy /= length

    nx = -dy
    ny = dx

    corners = [
        (bbox.x0, bbox.y0),
        (bbox.x1, bbox.y0),
        (bbox.x1, bbox.y1),
        (bbox.x0, bbox.y1),
    ]

    along_vals = [x * dx + y * dy for x, y in corners]
    normal_vals = [x * nx + y * ny for x, y in corners]

    along_min, along_max = min(along_vals), max(along_vals)
    normal_min, normal_max = min(normal_vals), max(normal_vals)

    half_along = (along_max - along_min) / 2.0
    half_normal = (normal_max - normal_min) / 2.0

    along_center = (along_max + along_min) / 2.0
    normal_center = (normal_max + normal_min) / 2.0

    cx = along_center * dx + normal_center * nx
    cy = along_center * dy + normal_center * ny

    def point(
        along: float,
        normal: float,
    ) -> fitz.Point:

        return fitz.Point(
            cx
            + dx * along
            + nx * normal,
            cy
            + dy * along
            + ny * normal,
        )

    ul = point(
        -half_along,
        -half_normal,
    )

    ur = point(
        half_along,
        -half_normal,
    )

    ll = point(
        -half_along,
        half_normal,
    )

    lr = point(
        half_along,
        half_normal,
    )

    return fitz.Quad(
        ul,
        ur,
        ll,
        lr,
    )


def extract_image_items(
    page: "fitz.Page",
) -> list[OverlayItem]:
    """
    Extract actual image placements.

    get_image_info(xrefs=True) is preferable to get_images() here
    because it provides placement information, transformation data,
    resolution and mask information.
    """

    items = []

    try:
        image_infos = page.get_image_info(
            xrefs=True
        )
    except Exception:
        image_infos = []

    for index, info in enumerate(
        image_infos
    ):

        bbox = fitz.Rect(
            info.get(
                "bbox",
                (0, 0, 0, 0),
            )
        )

        width_px = info.get(
            "width",
            0,
        )

        height_px = info.get(
            "height",
            0,
        )

        xref = info.get(
            "xref",
            0,
        )

        smask = info.get(
            "smask",
            0,
        )

        transform = info.get(
            "transform",
            None,
        )

        metadata = {
            "image_index": index,

            "xref": xref,

            "width_px": width_px,
            "height_px": height_px,

            "display_width": round(
                bbox.width,
                3,
            ),

            "display_height": round(
                bbox.height,
                3,
            ),

            "display_area": round(
                bbox.get_area(),
                3,
            ),

            "pixel_count": (
                width_px * height_px
                if width_px and height_px
                else None
            ),

            "aspect_ratio": round(
                width_px / height_px,
                4,
            )
            if height_px
            else None,

            "colorspace": info.get(
                "cs-name",
                info.get(
                    "colorspace",
                    None,
                ),
            ),

            "bpc": info.get(
                "bpc",
                None,
            ),

            "xres": info.get(
                "xres",
                None,
            ),

            "yres": info.get(
                "yres",
                None,
            ),

            "has_mask": info.get(
                "has-mask",
                False,
            ),

            "smask": smask or None,

            "digest": info.get(
                "digest",
                None,
            ),
        }


        if transform is not None:

            # get_image_info() returns "transform" as a plain 6-tuple,
            # not a fitz.Matrix, so it must be wrapped before use.
            transform = fitz.Matrix(*transform)

            metadata[
                "transform"
            ] = _format_matrix(
                transform
            )

            metadata[
                "rotation"
            ] = round(
                _matrix_rotation(
                    transform
                ),
                3,
            )

            sx, sy = _matrix_scale(
                transform
            )

            metadata[
                "scale"
            ] = (
                round(sx, 5),
                round(sy, 5),
            )

        # --------------------------------------------------------------
        # Physical display DPI
        #
        # PDF points are 1/72 inch.
        # --------------------------------------------------------------

        if bbox.width > 0 and width_px:
            dpi_x = (
                width_px
                / bbox.width
                * 72.0
            )
        else:
            dpi_x = None

        if bbox.height > 0 and height_px:
            dpi_y = (
                height_px
                / bbox.height
                * 72.0
            )
        else:
            dpi_y = None

        metadata[
            "effective_dpi"
        ] = (
            round(dpi_x, 2)
            if dpi_x is not None
            else None,
            round(dpi_y, 2)
            if dpi_y is not None
            else None,
        )


        if xref:
            try:
                extracted = page.parent.extract_image(
                    xref
                )

                if extracted:
                    metadata[
                        "extension"
                    ] = extracted.get(
                        "ext"
                    )

                    metadata[
                        "compressed_size"
                    ] = extracted.get(
                        "size"
                    )

                    metadata[
                        "compressed_size_kb"
                    ] = round(
                        extracted.get(
                            "size",
                            0,
                        )
                        / 1024,
                        2,
                    )

                    metadata[
                        "image_colorspace"
                    ] = extracted.get(
                        "colorspace"
                    )

                    metadata[
                        "image_bpc"
                    ] = extracted.get(
                        "bpc"
                    )

            except Exception:
                pass

        attrs = {
            "xref": xref,
            "colorspace": metadata.get(
                "colorspace"
            ),
            "has_mask": metadata.get(
                "has_mask"
            ),
        }

        items.append(
            OverlayItem(
                bbox=bbox,
                kind="image",
                shape="rect",
                attrs=attrs,
                metadata=metadata,
                label=f"xref={xref}",
            )
        )

    return items


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