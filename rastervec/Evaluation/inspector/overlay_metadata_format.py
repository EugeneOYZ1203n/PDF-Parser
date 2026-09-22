"""Pure formatting of an `OverlayItem`'s metadata dict into the multi-line
text `PageView`'s hover tooltip displays -- no Tk/canvas coupling. Lifted
out of `overlay_canvas.py::PageView` (where these were `_format_metadata`/
`_pretty_key`/`_format_value`, the latter two already `@staticmethod`s)."""
from __future__ import annotations

from rastervec.Evaluation.inspector.layer_types import OverlayItem

_PREFERRED_KEY_ORDER = [


    "text",
    "image_index",
    "xref",
    "type",
    "operation",


    "x",
    "y",
    "width",
    "height",
    "drawing_width",
    "drawing_height",
    "display_width",
    "display_height",

    "area",
    "display_area",
    "drawing_area",

    "aspect_ratio",

    "bbox",
    "drawing_bbox",


    "angle",
    "rotation",
    "direction",

    "transform",
    "scale",


    "origin",
    "font",
    "font_size",
    "flags",
    "color",
    "ascender",
    "descender",


    "width_px",
    "height_px",
    "pixel_count",

    "colorspace",
    "image_colorspace",

    "bpc",
    "image_bpc",

    "xres",
    "yres",
    "effective_dpi",

    "has_mask",
    "smask",

    "extension",
    "compressed_size",
    "compressed_size_kb",

    "digest",


    "start",
    "end",
    "length",

    "path_type",
    "stroke_color",
    "fill_color",

    "stroke_width",
    "stroke_opacity",
    "fill_opacity",

    "line_cap",
    "line_join",

    "dashes",
    "close_path",
    "even_odd",

    "layer",

    "operation_count",
    "item_index",
    "seqno",

    "corners",
    "control_points",
    "control_polygon_length",


    "author",
    "contents",
    "subject",
    "opacity",
    "creationDate",
    "modDate",
]


def format_metadata(
    item: OverlayItem,
    layer_key: str,
) -> str:

    metadata = item.get_metadata()

    lines = []

    title = layer_key.upper()

    if item.kind:
        title += f" — {item.kind}"

    lines.append(title)
    lines.append("")

    displayed = set()

    for key in _PREFERRED_KEY_ORDER:

        if key not in metadata:
            continue

        value = metadata[key]

        if value is None:
            continue

        lines.append(
            f"{pretty_key(key)}: "
            f"{format_value(value)}"
        )

        displayed.add(key)

    for key, value in metadata.items():

        if key in displayed:
            continue

        if value is None:
            continue

        lines.append(
            f"{pretty_key(key)}: "
            f"{format_value(value)}"
        )

    return "\n".join(lines)


def pretty_key(
    key: str,
) -> str:

    return key.replace(
        "_",
        " ",
    ).title()


def format_value(value):

    if value is None:
        return ""

    if isinstance(value, float):
        return f"{value:.3f}"

    if isinstance(value, bool):
        return "Yes" if value else "No"

    if isinstance(value, tuple):

        return "(" + ", ".join(
            format_value(v)
            for v in value
        ) + ")"

    if isinstance(value, list):

        if not value:
            return "[]"

        return "[" + ", ".join(
            format_value(v)
            for v in value
        ) + "]"

    if isinstance(value, int):

        return str(value)

    return str(value)
