from __future__ import annotations

import cv2
import numpy as np

from raster_parser.commons.raster_text.enhance import clahe_lab


def test_clahe_lab_raises_contrast_and_keeps_hue():
    img = np.full((200, 200, 3), 200, np.uint8)  # gray paper
    img[90:110, 20:180] = (170, 120, 120)  # faint, low-contrast reddish line
    enhanced, lab = clahe_lab(img)

    assert enhanced.shape == img.shape and enhanced.dtype == np.uint8
    before = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    contrast_before = int(before[0, 0, 0]) - int(before[100, 100, 0])
    contrast_after = int(lab[0, 0, 0]) - int(lab[100, 100, 0])
    assert contrast_after > contrast_before
    # a/b (hue, chroma) untouched
    assert (lab[:, :, 1:] == before[:, :, 1:]).all()
    r, g, b = (int(c) for c in enhanced[100, 100])
    assert r > g and r > b
