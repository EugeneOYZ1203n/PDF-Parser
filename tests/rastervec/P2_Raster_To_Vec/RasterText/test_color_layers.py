from __future__ import annotations

import cv2
import numpy as np

from rastervec.P2_Raster_To_Vec.RasterText.color_layers import (
    PAPER, _smooth_circular, hue_peaks, recolor, split_layers,
)
from rastervec.P2_Raster_To_Vec.RasterText.enhance import clahe_lab


def _paper(h=300, w=400, seed=0) -> np.ndarray:
    img = np.full((h, w, 3), 248, np.uint8)
    img += np.random.default_rng(seed).integers(0, 5, img.shape, dtype=np.uint8)
    return img


def _split(img):
    _enhanced, lab = clahe_lab(img)
    return split_layers(lab, img)


def _layer_at(layers, y, x) -> int:
    return int(layers.labels[y, x])


def test_black_red_and_pale_yellow_become_three_layers():
    img = _paper()
    cv2.line(img, (10, 50), (390, 50), (0, 0, 0), 3)
    cv2.line(img, (10, 150), (390, 150), (220, 30, 30), 3)
    cv2.line(img, (10, 250), (390, 250), (250, 235, 70), 3)  # bright in L, kept by chroma
    layers = _split(img)

    black, red, yellow = (_layer_at(layers, y, 200) for y in (50, 150, 250))
    assert PAPER not in (black, red, yellow)
    assert len({black, red, yellow}) == 3
    assert layers.n_layers == 3
    assert layers.kinds[black] == "achromatic"
    assert layers.kinds[red] == layers.kinds[yellow] == "hue"
    assert tuple(layers.colors_rgb[red]) == (220, 30, 30)   # median of the *original* pixels
    assert _layer_at(layers, 100, 200) == PAPER


def test_hue_straddling_a_bin_edge_is_not_split():
    img = _paper()
    # two shades of one red, 4 deg of LAB hue apart -> one smoothed peak
    cv2.line(img, (10, 100), (390, 100), (220, 30, 30), 3)
    cv2.line(img, (10, 200), (390, 200), (220, 30, 45), 3)
    layers = _split(img)
    assert _layer_at(layers, 100, 200) == _layer_at(layers, 200, 200)
    assert layers.n_layers == 1


def test_small_layer_is_folded_into_its_nearest():
    img = _paper()
    cv2.line(img, (10, 100), (390, 100), (0, 0, 0), 3)
    img[200:205, 200:205] = (40, 80, 230)  # 25 px of blue: below LAYER_MIN_PIXELS
    layers = _split(img)
    assert layers.n_layers == 1
    assert _layer_at(layers, 202, 202) == _layer_at(layers, 100, 200)


def test_blank_paper_has_no_ink():
    layers = _split(_paper(seed=3))
    assert layers.n_layers == 0
    assert (layers.labels == PAPER).all()
    assert recolor(layers)[..., 3].max() == 0


def test_hue_peaks_wrap_around_the_circle():
    hist = np.zeros(360)
    hist[358] = 10.0
    hist[2] = 10.0  # one red cluster split by the 0/360 seam
    smooth = _smooth_circular(hist, 6.0)
    peaks = hue_peaks(smooth)
    assert len(peaks) == 1 and (peaks[0] >= 355 or peaks[0] <= 4)
