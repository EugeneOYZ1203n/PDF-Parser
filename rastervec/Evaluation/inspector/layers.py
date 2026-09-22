"""PDF layer and overlay data model.

This module defines the common representation used by the PDF inspector.

Coordinate convention
---------------------
All geometry stored in OverlayItem is in PDF page coordinates.

Rendering code is responsible for applying:

    page coordinates
        -> page rotation
        -> zoom
        -> canvas coordinates

Geometry
--------
OverlayItem supports three types of geometry:

    Rect:
        Axis-aligned bounding box.

    Quad:
        Rotated rectangle, useful for text.

    Points:
        Arbitrary polygon / line geometry.

The bbox is always retained even when a quad or arbitrary geometry is
available. This makes filtering, spatial indexing, and hit testing easier.

**This is the extensibility core** (`filter_items`/`summarize_selection`/
`build_layers` -- see `CLAUDE.md`'s inspector-architecture section): the
data model itself (`OverlayItem`/`SubFilterSpec`/`LayerSpec`) lives in
`layer_types.py`, and the color utilities (`rgb_to_hex`/seqno-rainbow) live
in `layer_colors.py` -- both re-exported here since several other inspector
modules import them by this module's own name.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

import pymupdf as fitz

from rastervec.Evaluation.inspector.layer_types import (  # noqa: F401
    LayerSpec,
    OverlayItem,
    SubFilterSpec,
    _point_to_tuple,
    _quad_to_tuple,
    _rect_to_tuple,
)
from rastervec.Evaluation.inspector.layer_colors import (  # noqa: F401
    rgb_to_hex,
    seqno_rainbow_color_map,
    seqno_rainbow_colorer,
)


NONE_OPTION = (
    "__none__",
    "(none)",
)


ITEM_KIND_OPTIONS = [
    ("l", "Line (l)"),
    ("re", "Rectangle (re)"),
    ("qu", "Quad (qu)"),
    ("c", "Curve (c)"),
]


PATH_TYPE_OPTIONS = [
    ("f", "Fill only (f)"),
    ("s", "Stroke only (s)"),
    ("fs", "Fill + Stroke (fs)"),
]


DRAWING_CLOSE_OPTIONS = [
    ("True", "Closed"),
    ("False", "Open"),
]


IMAGE_MASK_OPTIONS = [
    ("True", "Has mask"),
    ("False", "No mask"),
]


item_kind_filter = SubFilterSpec(
    key="item_kind",
    label="Item type",
    attr_getter=lambda item: item.attrs.get(
        "kind"
    ),
    static_options=ITEM_KIND_OPTIONS,
)


path_type_filter = SubFilterSpec(
    key="path_type",
    label="Path type",
    attr_getter=lambda item: item.attrs.get(
        "path_type"
    ),
    static_options=PATH_TYPE_OPTIONS,
)


stroke_color_filter = SubFilterSpec(
    key="stroke_color",
    label="Stroke color",
    attr_getter=lambda item: (
        item.attrs.get("stroke_color")
        or NONE_OPTION[0]
    ),
    dynamic=True,
    render_as="swatch",
)


fill_color_filter = SubFilterSpec(
    key="fill_color",
    label="Fill color",
    attr_getter=lambda item: (
        item.attrs.get("fill_color")
        or NONE_OPTION[0]
    ),
    dynamic=True,
    render_as="swatch",
)


drawing_close_filter = SubFilterSpec(
    key="close_path",
    label="Path closure",
    attr_getter=lambda item: str(
        item.metadata.get(
            "close_path"
        )
    ),
    static_options=DRAWING_CLOSE_OPTIONS,
)


image_mask_filter = SubFilterSpec(
    key="has_mask",
    label="Image mask",
    attr_getter=lambda item: str(
        item.attrs.get(
            "has_mask"
        )
    ),
    static_options=IMAGE_MASK_OPTIONS,
)


def filter_items(
    items: list[OverlayItem],
    active_filters: dict[
        str,
        set[str],
    ],
) -> list[OverlayItem]:
    """Filter OverlayItems according to active filters.

    Semantics:

        Across different subfilters:
            AND

        Within one subfilter:
            OR

        Empty checked set:
            no restriction
    """

    result: list[OverlayItem] = []

    for item in items:
        keep = True

        for key, checked in active_filters.items():

            if not checked:
                continue

            getter = _GETTERS.get(key)

            if getter is None:
                continue

            value = getter(item)

            if value not in checked:
                keep = False
                break

        if keep:
            result.append(item)

    return result


def summarize_selection(
    items: list[OverlayItem],
    region: "fitz.Rect",
) -> dict[str, Any]:
    """Vector/item/type counts for `items` intersecting `region`.

    "Items" are individual OverlayItems (one per draw operation);
    "vectors" are distinct source paths, identified by the `seqno` attr
    shared by every item belonging to the same `get_drawings()` path.
    Intersection is bbox-only (matches the existing rubber-band select in
    scripts/label/vector_label.py).
    """

    hits = [
        item
        for item in items
        if item.bbox.intersects(region)
    ]

    vector_ids = {
        item.attrs["seqno"]
        for item in hits
        if "seqno" in item.attrs
    }

    type_counts = Counter(
        item.attrs.get("kind", "")
        for item in hits
    )

    return {
        "vector_count": len(vector_ids),
        "item_count": len(hits),
        "type_counts": type_counts,
    }


_GETTERS: dict[
    str,
    Callable[
        [OverlayItem],
        Any,
    ],
] = {
    "item_kind": item_kind_filter.attr_getter,
    "path_type": path_type_filter.attr_getter,
    "stroke_color": stroke_color_filter.attr_getter,
    "fill_color": fill_color_filter.attr_getter,
    "close_path": drawing_close_filter.attr_getter,
    "has_mask": image_mask_filter.attr_getter,
}


def build_layers(pdf_model) -> list[LayerSpec]:

    return [


        LayerSpec(
            key="text",
            label="Text",
            color="#2563eb",
            extractor=pdf_model.extract_text_items,
            enabled_default=True,
        ),


        LayerSpec(
            key="images",
            label="Images",
            color="#16a34a",
            extractor=pdf_model.extract_image_items,
            subfilters=[
                image_mask_filter,
            ],
        ),


        LayerSpec(
            key="annotations",
            label="Annotations",
            color="#dc2626",
            extractor=pdf_model.extract_annot_items,
        ),


        LayerSpec(
            key="drawings",
            label="Drawings",
            color="#9333ea",
            extractor=pdf_model.extract_drawing_items,
            subfilters=[
                item_kind_filter,
                path_type_filter,
                stroke_color_filter,
                fill_color_filter,
                drawing_close_filter,
            ],
        ),
    ]
