"""`PdfDocument` (the thin pymupdf-document wrapper) plus the geometry/matrix/
color helpers shared by every extractor in `pdf_model_text.py`,
`pdf_model_image.py`, and `pdf_model_drawing.py` (re-exported from
`commons.helpers.pymupdf_geometry` / `geometry` under their old private names)."""

from __future__ import annotations

import pymupdf
from raster_parser.commons.helpers.pymupdf_geometry import (  # noqa: F401
    format_matrix as _format_matrix,
    line_length as _line_length,
    matrix_rotation as _matrix_rotation,
    matrix_scale as _matrix_scale,
    point_angle as _point_angle,
    quad_angle as _quad_angle,
    quad_metadata as _quad_metadata,
    rect_metadata as _rect_metadata,
)
from raster_parser.commons.helpers.geometry import round_color as _round_color  # noqa: F401


class PdfDocument:
    """Thin wrapper around a PyMuPDF document."""

    def __init__(self, path: str):
        self.path = path
        self.doc = pymupdf.open(path)

    @property
    def page_count(self) -> int:
        return self.doc.page_count

    def get_page(self, n: int) -> "pymupdf.Page":
        return self.doc[n]

    def render_pixmap(
        self,
        n: int,
        zoom: float,
    ) -> "pymupdf.Pixmap":
        """
        Render a page.

        get_pixmap() respects the page's /Rotate attribute.
        """

        page = self.get_page(n)

        matrix = pymupdf.Matrix(
            zoom,
            zoom,
        )

        return page.get_pixmap(
            matrix=matrix,
        )

    def close(self) -> None:
        self.doc.close()
