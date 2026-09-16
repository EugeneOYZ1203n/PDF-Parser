from __future__ import annotations

import math

import numpy as np
import pytest

from rastervec.P3_Vector_Parsing.LegacyRecreation.config import OCR_LANG, OCR_VERSION
from rastervec.P3_Vector_Parsing.LegacyRecreation.paddle_engine import (
    PaddleDetectBackend,
    _normalize_rotation,
    _quad_rotation_deg,
    _rotate_crop,
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


def test_normalize_rotation_wraps_to_plus_minus_90():
    assert _normalize_rotation(0.0) == pytest.approx(0.0)
    # Exact multiples of 90 sit on the wrap boundary -- +/-90 are the same
    # line orientation (mod 180), and the formula consistently resolves the
    # boundary to -90.
    assert _normalize_rotation(90.0) == pytest.approx(-90.0)
    assert _normalize_rotation(91.0) == pytest.approx(-89.0)
    assert _normalize_rotation(-91.0) == pytest.approx(89.0)
    assert _normalize_rotation(180.0) == pytest.approx(0.0)


def test_quad_rotation_deg_axis_aligned_quad_is_zero():
    quad = np.array([(0.0, 0.0), (10.0, 0.0), (10.0, 2.0), (0.0, 2.0)])
    assert _quad_rotation_deg(quad) == pytest.approx(0.0, abs=1e-6)


def test_quad_rotation_deg_picks_longer_edge_for_vertical_text():
    # Taller than wide -- the left edge (0->3) is the dominant (longer) one,
    # already vertical, so the "bring to horizontal" rotation is +/-90.
    quad = np.array([(0.0, 0.0), (2.0, 0.0), (2.0, 10.0), (0.0, 10.0)])
    assert abs(_quad_rotation_deg(quad)) == pytest.approx(90.0, abs=1e-6)


def test_quad_rotation_deg_tilted_quad():
    # Top edge tilted 10 degrees below horizontal in image (y-down) coords.
    theta = math.radians(-10.0)
    dx, dy = 10.0 * math.cos(theta), 10.0 * math.sin(theta)
    quad = np.array([(0.0, 0.0), (dx, dy), (dx, dy + 2.0), (0.0, 2.0)])
    # Rotating by +10 (CCW) should bring this edge to horizontal.
    assert _quad_rotation_deg(quad) == pytest.approx(10.0, abs=1e-3)


def test_rotate_crop_axis_aligned_quad_is_a_plain_crop():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[5:10, 2:12] = 255
    quad = np.array([(2.0, 5.0), (12.0, 5.0), (12.0, 10.0), (2.0, 10.0)])

    crop = _rotate_crop(img, quad)

    assert crop.shape[:2] == (5, 10)
    assert crop.mean() > 200  # mostly the white region we carved out


def test_paddle_detect_backend_key_uses_config_defaults():
    assert PaddleDetectBackend().key == (OCR_VERSION, OCR_LANG)
