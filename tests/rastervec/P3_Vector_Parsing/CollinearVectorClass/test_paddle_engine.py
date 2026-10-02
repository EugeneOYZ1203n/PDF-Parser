from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from rastervec.P3_Vector_Parsing.CollinearVectorClass import paddle_engine as pe


def _line_image(angle_deg: float, size: int = 200) -> np.ndarray:
    """White BGR image with one thick black line through the centre whose
    image-space (y down) direction angle is `angle_deg`."""
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    c = size / 2
    dx, dy = math.cos(math.radians(angle_deg)) * 70, math.sin(math.radians(angle_deg)) * 70
    cv2.line(img, (int(c - dx), int(c - dy)), (int(c + dx), int(c + dy)), (0, 0, 0), 3)
    return img


@pytest.mark.parametrize("angle", [0.0, 30.0, 90.0, 135.0])
def test_hough_angle_is_y_down_direction_mod_180(angle):
    got, mask = pe.hough_angle_deg(_line_image(angle))
    assert mask.any()
    assert min(abs(got - angle) % 180, 180 - abs(got - angle) % 180) <= 2.5  # 1-deg theta grid, thick line


def test_hough_angle_none_on_blank_crop():
    got, _ = pe.hough_angle_deg(np.full((20, 20, 3), 255, dtype=np.uint8))
    assert got is None


@pytest.mark.parametrize("angle", [30.0, -60.0, 89.0])
def test_rotate_image_levels_a_line_at_that_angle(angle):
    rotated, _m = pe.rotate_image(_line_image(angle), angle)
    got, _ = pe.hough_angle_deg(rotated)
    assert min(got, 180 - got) <= 2.5  # horizontal


def test_rotate_image_zero_is_identity():
    img = _line_image(10.0)
    out, m = pe.rotate_image(img, 0.0)
    assert out is img
    assert np.allclose(pe.unrotate_points(np.array([[3.0, 4.0]]), m), [[3.0, 4.0]])


def test_unrotate_points_inverts_rotation():
    img = np.full((50, 120, 3), 255, dtype=np.uint8)
    _out, m = pe.rotate_image(img, 37.0)
    pts = np.array([[0.0, 0.0], [119.0, 0.0], [60.0, 25.0]])
    forward = pts @ m[:, :2].T + m[:, 2]
    assert np.allclose(pe.unrotate_points(forward, m), pts)


def test_normalize_rotation_range():
    assert pe.normalize_rotation(90.0) == -90.0
    assert pe.normalize_rotation(179.0) == -1.0
    assert pe.normalize_rotation(-91.0) == 89.0
