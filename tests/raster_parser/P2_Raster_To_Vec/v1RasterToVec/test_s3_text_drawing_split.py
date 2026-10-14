"""v1RasterToVec step 3: text/drawing split (copied from tests/raster_parser/test_text_drawing_parser.py)."""

from __future__ import annotations

import numpy as np

from raster_parser.P2_Raster_To_Vec.v1RasterToVec.s3_text_drawing_split import line_mask


def test_line_mask_runs_on_blank_and_line_image():
    blank = np.zeros((64, 64), dtype=np.uint8)
    out = line_mask(blank, line_min=10, grow=1, threshold=20, max_gap=5)
    assert out.shape == blank.shape
    assert out.dtype == np.uint8

    img = np.zeros((64, 64), dtype=np.uint8)
    img[30, 5:55] = 255
    out = line_mask(img, line_min=10, grow=1, threshold=20, max_gap=5)
    assert out.shape == img.shape
    assert int(out.max()) == 255
