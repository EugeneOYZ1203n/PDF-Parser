"""`commons.helpers.pixmap`: pixmap -> numpy and the render zoom for a DPI."""

from __future__ import annotations

import numpy as np
import pymupdf
import pytest

from raster_parser.commons.helpers.pixmap import pixmap_to_array, zoom_matrix


@pytest.mark.parametrize(
    "colorspace, alpha, n",
    [
        (pymupdf.csRGB, False, 3),
        (pymupdf.csRGB, True, 4),
        (pymupdf.csGRAY, False, 1),
    ],
)
def test_pixmap_to_array_is_height_width_channels(colorspace, alpha, n):
    pix = pymupdf.Pixmap(colorspace, pymupdf.IRect(0, 0, 5, 3), alpha)
    pix.set_pixel(4, 2, (10,) * pix.n)
    arr = pixmap_to_array(pix)
    assert arr.dtype == np.uint8 and arr.shape == (3, 5, n)
    np.testing.assert_array_equal(
        arr, np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    )
    assert arr[2, 4, 0] == 10


@pytest.mark.parametrize("dpi", [72, 144, 300, 96.5])
def test_zoom_matrix_scales_points_to_pixels(dpi):
    assert tuple(zoom_matrix(dpi)) == tuple(pymupdf.Matrix(dpi / 72.0, dpi / 72.0))
