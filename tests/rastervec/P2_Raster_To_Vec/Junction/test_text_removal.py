from __future__ import annotations

import numpy as np

from rastervec.P2_Raster_To_Vec.Junction.color_separation import separate_colors
from rastervec.P2_Raster_To_Vec.Junction.text_ocr import OcrHit
from rastervec.P2_Raster_To_Vec.Junction.text_removal import erase_text, ink_label


def _scene():
    img = np.full((120, 120, 3), 255, dtype=np.uint8)
    img[40:52, 20:60] = 0                  # "text" ink: 480 px black
    img[:, 36:41] = (220, 20, 20)          # red line crossing it, 5 px wide
    quad = np.array([[18, 38], [62, 38], [62, 54], [18, 54]], dtype=float)
    return img, quad


def test_ink_label_is_dominant_non_background_layer():
    img, quad = _scene()
    layers = separate_colors(img)
    assert ink_label(layers.labels, quad, layers.background) == layers.labels[45, 25]


def test_erase_removes_text_ink_but_keeps_crossing_line():
    img, quad = _scene()
    layers = separate_colors(img)
    red = layers.labels[5, 38]
    black = int(layers.labels[45, 25])
    inks = erase_text(img, layers, [OcrHit(quad=quad, padded_box=(0, 0, 120, 120), text="AB")])

    assert inks == [black]
    region = img[40:52, 20:60]
    assert not (region.max(axis=2) < 60).any()          # no black left
    assert (img[40:52, 38] == (220, 20, 20)).all()      # line's core survives
    assert layers.labels[45, 25] == layers.background   # labels follow the erase
    assert layers.labels[45, 38] == red


def test_blank_hit_is_not_erased():
    img, quad = _scene()
    layers = separate_colors(img)
    before = img.copy()
    inks = erase_text(img, layers, [OcrHit(quad=quad, padded_box=(0, 0, 120, 120), text="")])
    assert inks == [None]
    assert (img == before).all()
