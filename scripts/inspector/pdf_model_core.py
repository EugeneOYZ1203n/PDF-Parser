"""`PdfDocument` (the thin fitz-document wrapper) plus the geometry/matrix/
color helpers shared by every extractor in `pdf_model_text.py`,
`pdf_model_image.py`, and `pdf_model_drawing.py`."""
from __future__ import annotations

from math import degrees, atan2, hypot, sqrt

import pymupdf as fitz


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
