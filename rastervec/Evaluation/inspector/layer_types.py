"""The inspector's data model: `OverlayItem` (one inspectable object
extracted from a PDF page) plus the declarative `SubFilterSpec`/`LayerSpec`
types `layers.py::build_layers` wires extractors to. See `layers.py`'s own
module docstring for the coordinate convention and geometry priority these
types share.
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
