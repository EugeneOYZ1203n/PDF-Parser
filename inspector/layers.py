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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pymupdf as fitz


@dataclass
class OverlayItem:
    """A single inspectable object extracted from a PDF page.

    All coordinates are stored in PDF page-space coordinates.

    Parameters
    ----------
    bbox:
        Axis-aligned bounding box of the object.

    quad:
        Optional rotated rectangle. This is preferred for text because
        it preserves the actual text orientation.

    kind:
        Semantic object type, e.g. "text", "image", "line", "curve".

    shape:
        How the object should be rendered:

            "rect"
            "quad"
            "polygon"
            "line"

    points:
        Arbitrary page-space points for polygons and lines.

    attrs:
        Machine-readable attributes used by filters.

    metadata:
        Human-readable / diagnostic information shown on hover.

    label:
        Optional short display label.
    """

    bbox: fitz.Rect

    quad: fitz.Quad | None = None

    angle: float | None = None

    kind: str = ""

    shape: str = "rect"

    points: list[fitz.Point] | None = None

    attrs: dict[str, Any] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    label: str = ""

    def __post_init__(self) -> None:
        """Normalize geometry after construction."""

        if not isinstance(self.bbox, fitz.Rect):
            self.bbox = fitz.Rect(self.bbox)

        if self.points is not None:
            self.points = [
                point
                if isinstance(point, fitz.Point)
                else fitz.Point(point)
                for point in self.points
            ]

        if self.quad is not None and not isinstance(
            self.quad,
            fitz.Quad,
        ):
            self.quad = fitz.Quad(self.quad)


    def get_geometry_points(self) -> list[fitz.Point]:
        """Return the object's visual geometry as points.

        Priority:

            quad -> points -> bbox

        This is useful for generic rendering and hit testing.
        """

        if self.quad is not None:
            return [
                self.quad.ul,
                self.quad.ur,
                self.quad.lr,
                self.quad.ll,
            ]

        if self.points:
            return list(self.points)

        return [
            self.bbox.tl,
            self.bbox.tr,
            self.bbox.br,
            self.bbox.bl,
        ]

    def transformed(
        self,
        matrix: fitz.Matrix,
    ) -> "OverlayItem":
        """Return a copy transformed by a PyMuPDF matrix.

        This is useful when debugging coordinate transformations.

        Normally the renderer should transform the geometry directly
        instead of creating transformed OverlayItems.
        """

        bbox = self.bbox * matrix

        quad = None

        if self.quad is not None:
            quad = self.quad * matrix

        points = None

        if self.points is not None:
            points = [
                point * matrix
                for point in self.points
            ]

        return OverlayItem(
            bbox=bbox,
            quad=quad,
            angle=self.angle,
            kind=self.kind,
            shape=self.shape,
            points=points,
            attrs=dict(self.attrs),
            metadata=dict(self.metadata),
            label=self.label,
        )


    def get_metadata(self) -> dict[str, Any]:
        """Return metadata suitable for display in the inspector."""

        result = dict(self.metadata)

        # Generic information should always be available.
        result.setdefault(
            "kind",
            self.kind,
        )

        result.setdefault(
            "shape",
            self.shape,
        )

        result.setdefault(
            "bbox",
            _rect_to_tuple(self.bbox),
        )

        # If this is rotated geometry, expose the four corners.
        if self.quad is not None:
            result.setdefault(
                "quad",
                _quad_to_tuple(self.quad),
            )

        if self.angle is not None:
            result.setdefault(
                "angle",
                self.angle,
            )

        if self.points is not None:
            result.setdefault(
                "points",
                [
                    _point_to_tuple(point)
                    for point in self.points
                ],
            )

        if self.label:
            result.setdefault(
                "label",
                self.label,
            )

        return result


def _point_to_tuple(
    point: fitz.Point,
) -> tuple[float, float]:
    """Convert a point to a compact serializable tuple."""

    return (
        round(point.x, 2),
        round(point.y, 2),
    )


def _rect_to_tuple(
    rect: fitz.Rect,
) -> tuple[float, float, float, float]:
    """Convert a rectangle to a compact serializable tuple."""

    return (
        round(rect.x0, 2),
        round(rect.y0, 2),
        round(rect.x1, 2),
        round(rect.y1, 2),
    )


def _quad_to_tuple(
    quad: fitz.Quad,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    """Convert a quad to four corner coordinates.

    Order:

        upper-left
        upper-right
        lower-right
        lower-left
    """

    return (
        _point_to_tuple(quad.ul),
        _point_to_tuple(quad.ur),
        _point_to_tuple(quad.lr),
        _point_to_tuple(quad.ll),
    )


@dataclass
class SubFilterSpec:
    """Definition of a filter belonging to a layer."""

    key: str

    label: str

    attr_getter: Callable[
        [OverlayItem],
        Any,
    ]

    static_options: list[
        tuple[str, str]
    ] | None = None

    dynamic: bool = False

    # "checkbox" | "swatch"
    render_as: str = "checkbox"


@dataclass
class LayerSpec:
    """Definition of one inspectable PDF layer."""

    key: str

    label: str

    color: str

    extractor: Callable[
        [fitz.Page],
        list[OverlayItem],
    ]

    subfilters: list[
        SubFilterSpec
    ] = field(
        default_factory=list
    )

    enabled_default: bool = False


def rgb_to_hex(color) -> str:
    """Convert an RGB tuple in [0, 1] to a Tkinter hex colour."""

    if not color:
        return "#000000"

    return "#%02x%02x%02x" % tuple(
        round(
            max(
                0.0,
                min(1.0, c),
            )
            * 255
        )
        for c in color
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