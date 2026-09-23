from __future__ import annotations

import numpy as np

from rastervec.P3_Vector_Parsing.VectorClassification.paddle_engine import (
    _CROP_BORDER_PX,
    _CROP_EXPAND_FRACTION,
    _rotate_crop,
)


def test_rotate_crop_expands_quad_and_adds_white_border():
    img = np.zeros((80, 150, 3), dtype=np.uint8)
    # width=100, height=40 -- chosen so *1.05 lands on an exact integer,
    # keeping the shape assertion unambiguous.
    quad = np.array([(10.0, 10.0), (110.0, 10.0), (110.0, 50.0), (10.0, 50.0)])

    crop = _rotate_crop(img, quad)

    expanded_w = int(round(100 * (1.0 + _CROP_EXPAND_FRACTION)))
    expanded_h = int(round(40 * (1.0 + _CROP_EXPAND_FRACTION)))
    assert crop.shape[:2] == (expanded_h + 2 * _CROP_BORDER_PX, expanded_w + 2 * _CROP_BORDER_PX)

    b = _CROP_BORDER_PX
    assert (crop[:b, :, :] == 255).all()
    assert (crop[-b:, :, :] == 255).all()
    assert (crop[:, :b, :] == 255).all()
    assert (crop[:, -b:, :] == 255).all()
