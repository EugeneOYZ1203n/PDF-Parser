from __future__ import annotations

import numpy as np

from rastervec.P2_Raster_To_Vec.RasterText.color_layers import PAPER, split_layers
from rastervec.P2_Raster_To_Vec.RasterText.enhance import clahe_lab
from rastervec.P2_Raster_To_Vec.RasterText.text_ocr import OcrHit
from rastervec.P2_Raster_To_Vec.RasterText.text_removal import erase_text, ink_label


def _scene():
    img = np.full((120, 120, 3), 255, dtype=np.uint8)
    img[40:52, 20:60] = 0                  # "text" ink: 480 px black
    img[:, 36:41] = (220, 20, 20)          # red line crossing it, 5 px wide
    quad = np.array([[18, 38], [62, 38], [62, 54], [18, 54]], dtype=float)
    enhanced, lab = clahe_lab(img)
    return enhanced, split_layers(lab, img), quad


def test_ink_label_is_dominant_ink_layer():
    _img, layers, quad = _scene()
    assert ink_label(layers.labels, quad) == layers.labels[45, 25]


def test_erase_removes_text_ink_but_keeps_crossing_line():
    img, layers, quad = _scene()
    red = int(layers.labels[5, 38])
    black = int(layers.labels[45, 25])
    assert red != black and PAPER not in (red, black)
    line_before = img[40:52, 38].copy()
    inks, erased = erase_text(img, layers, [OcrHit(quad=quad, padded_box=(0, 0, 120, 120), text="AB")])

    assert inks == [black]
    assert erased[45, 25] and not erased[45, 38]
    assert not (layers.labels == black).any()            # every text pixel is paper now
    assert layers.labels[45, 25] == PAPER
    assert layers.labels[45, 38] == red                   # line's core survives
    assert (img[40:52, 38] == line_before).all()
    assert img[45, 25].min() > 200                        # painted paper colour


def test_blank_hit_is_not_erased():
    img, layers, quad = _scene()
    before = img.copy()
    labels_before = layers.labels.copy()
    inks, erased = erase_text(img, layers, [OcrHit(quad=quad, padded_box=(0, 0, 120, 120), text="")])
    assert inks == [None]
    assert not erased.any()
    assert (img == before).all() and (layers.labels == labels_before).all()
