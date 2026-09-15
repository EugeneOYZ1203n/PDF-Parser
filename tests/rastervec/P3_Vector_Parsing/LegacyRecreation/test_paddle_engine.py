from __future__ import annotations

import numpy as np

from rastervec.P3_Vector_Parsing.LegacyRecreation.paddle_engine import (
    dpi_for_cluster,
    pad_image,
)


def test_dpi_for_cluster_bumps_small_cluster_up(vector):
    v = vector(bbox=(0.0, 0.0, 1.0, 1.0))
    assert dpi_for_cluster([v], dpi=72, padding=0.0) > 72


def test_dpi_for_cluster_never_reduces_dpi(vector):
    v = vector(bbox=(0.0, 0.0, 1000.0, 1000.0))
    assert dpi_for_cluster([v], dpi=300, padding=0.0) == 300


def test_dpi_for_cluster_accounts_for_padding(vector):
    v = vector(bbox=(0.0, 0.0, 50.0, 50.0))
    assert dpi_for_cluster([v], dpi=72, padding=0.0) >= dpi_for_cluster(
        [v], dpi=72, padding=50.0
    )


def test_pad_image_adds_border_sized_from_larger_dimension():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    padded, (pad_x, pad_y) = pad_image(img, fraction=0.1)
    assert pad_x == pad_y == 2
    assert padded.shape == (14, 24, 3)
    assert (padded[0, 0] == 255).all()


def test_pad_image_empty_input_returns_unchanged():
    img = np.zeros((0, 0, 3), dtype=np.uint8)
    padded, offset = pad_image(img)
    assert padded is img
    assert offset == (0, 0)
