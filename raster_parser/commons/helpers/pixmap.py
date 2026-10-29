"""PyMuPDF pixmap helpers: a pixmap's samples as a numpy array, and the render zoom for a DPI."""

from __future__ import annotations

import numpy as np
import pymupdf

from raster_parser.commons.helpers.geometry import PDF_POINTS_PER_INCH


def pixmap_to_array(pix: pymupdf.Pixmap) -> np.ndarray:
    """`pix`'s samples as a read-only `(height, width, n)` uint8 array (no copy)."""
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)


def zoom_matrix(dpi: float) -> pymupdf.Matrix:
    """The matrix that renders a page at `dpi` (PDF points are 1/72 inch)."""
    return pymupdf.Matrix(dpi / PDF_POINTS_PER_INCH, dpi / PDF_POINTS_PER_INCH)
