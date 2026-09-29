from __future__ import annotations

import numpy as np

from rastervec.P2_Raster_To_Vec.Junction.color_separation import cone_coords, separate_colors


def _canvas(h: int = 120, w: int = 160) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _layer_color(layers, y: int, x: int) -> tuple[int, ...]:
    return tuple(int(c) for c in layers.centroids_rgb[layers.labels[y, x]])


def test_black_red_white_give_three_layers_white_background():
    img = _canvas()
    img[20:25, 10:150] = (0, 0, 0)       # 700 px black line
    img[30:110, 70:75] = (230, 20, 20)   # 400 px red line
    layers = separate_colors(img)

    assert layers.n_layers == 3
    assert _layer_color(layers, 0, 0)[0] > 240  # background is white
    assert layers.labels[0, 0] == layers.background
    black, red = layers.labels[22, 50], layers.labels[60, 72]
    assert len({int(black), int(red), layers.background}) == 3
    assert set(layers.ink_layers()) == {int(black), int(red)}


def test_hue_wraps_around_zero():
    img = _canvas()
    img[10:30, 10:40] = (255, 9, 0)      # H ~ 2 deg
    img[60:80, 10:40] = (255, 0, 9)      # H ~ 358 deg
    layers = separate_colors(img)
    assert layers.labels[20, 20] == layers.labels[70, 20]
    assert layers.n_layers == 2


def test_cone_collapses_dark_hues():
    # Near-black scan noise with opposite (meaningless) hues sits almost on
    # top of itself -- chroma S*V is tiny, so hue barely moves the point.
    pts = cone_coords(np.array([[22.0, 16.0, 16.0], [16.0, 16.0, 22.0]]))
    assert np.linalg.norm(pts[0] - pts[1]) < 0.05


def test_sparse_noise_goes_to_nearest_layer():
    img = _canvas()
    img[20:25, 10:150] = (0, 0, 0)
    img[100, 5:25] = (60, 60, 60)        # 20 px of dark gray -- DBSCAN noise
    layers = separate_colors(img)
    assert layers.labels[100, 10] == layers.labels[22, 50]


def test_blurred_ramp_still_separates_ink_from_paper():
    # A dense, continuous gray ramp between black and white would let DBSCAN
    # chain ink and paper together; the V split keeps them apart.
    img = _canvas(200, 256)
    ramp = np.linspace(0, 255, 256).astype(np.uint8)
    img[:100] = ramp[None, :, None]
    img[150:170, 20:200] = 0
    layers = separate_colors(img)
    assert layers.labels[160, 100] != layers.labels[180, 5]
