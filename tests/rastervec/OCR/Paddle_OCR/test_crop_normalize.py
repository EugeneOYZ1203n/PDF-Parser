from __future__ import annotations

import numpy as np
from PIL import Image

from rastervec.OCR.Paddle_OCR.crop_normalize import normalize_line_crop


def _crop(h: int, w: int) -> Image.Image:
    a = np.full((h, w), 255, np.uint8)
    a[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 0
    return Image.fromarray(a)


def test_normalize_line_crop_fixed_height_and_mode():
    out = normalize_line_crop(_crop(20, 80))
    assert out.height == 48
    assert 0 < out.width <= 1024
    assert out.mode == "L"


def test_normalize_line_crop_clamps_very_wide_crop():
    out = normalize_line_crop(_crop(10, 6000))
    assert out.width == 1024
    assert out.height == 48


def test_normalize_line_crop_zero_area_returned_unchanged():
    empty = Image.new("L", (0, 5))
    assert normalize_line_crop(empty) is empty
