from __future__ import annotations

import numpy as np
import pytest
from skimage.draw import line
from skimage.transform import rotate as sk_rotate

from rastervec.P3_Vector_Parsing.VectorClassification.paddle_engine import (
    _CROP_BORDER_PX,
    _axis_aligned_crop,
    _circular_avg_mod90,
    _combined_rotation_deg,
    _hough_angle_deg,
    _hough_ink_mask,
    _to_signed_small_angle,
    hough_deskew,
)


def test_axis_aligned_crop_expands_quad_and_adds_white_border():
    img = np.zeros((80, 150, 3), dtype=np.uint8)
    quad = np.array([(10.0, 10.0), (110.0, 10.0), (110.0, 50.0), (10.0, 50.0)])

    crop = _axis_aligned_crop(img, quad)

    # Bbox of the quad expanded _CROP_EXPAND_FRACTION about its own centroid
    # (60, 30): x in [7.5, 112.5], y in [9.0, 51.0] -- floor/ceil'd to pixel
    # bounds, unlike the old perspective-warp crop's single-dimension round.
    b = _CROP_BORDER_PX
    assert crop.shape[:2] == (51 - 9 + 2 * b, 113 - 7 + 2 * b)
    assert (crop[:b, :, :] == 255).all()
    assert (crop[-b:, :, :] == 255).all()
    assert (crop[:, :b, :] == 255).all()
    assert (crop[:, -b:, :] == 255).all()


def test_axis_aligned_crop_clamps_to_image_bounds():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    quad = np.array([(-5.0, -5.0), (25.0, -5.0), (25.0, 25.0), (-5.0, 25.0)])
    crop = _axis_aligned_crop(img, quad)
    assert crop.shape[0] > 0 and crop.shape[1] > 0  # no crash on an out-of-bounds quad


def test_hough_ink_mask_flags_dark_pixels_and_thickens_them():
    crop = np.full((10, 10, 3), 255, dtype=np.uint8)
    crop[5, 5] = (0, 0, 0)  # a single dark pixel
    mask = _hough_ink_mask(crop)
    assert mask[5, 5]
    assert mask.sum() > 1  # dilation thickened the single ink pixel


def test_circular_avg_mod90_matches_worked_examples():
    # Values near the 0/90 wraparound boundary average toward 0, not 45.
    assert _circular_avg_mod90(89.0, 1.0) == pytest.approx(0.0, abs=1e-6)
    assert _circular_avg_mod90(45.0, 43.0) == pytest.approx(44.0, abs=1e-6)
    assert _circular_avg_mod90(3.0, 4.0) == pytest.approx(3.5, abs=1e-6)


def test_to_signed_small_angle_wraps_near_90_to_negative():
    assert _to_signed_small_angle(88.0) == pytest.approx(-2.0)
    assert _to_signed_small_angle(2.0) == pytest.approx(2.0)
    assert _to_signed_small_angle(45.0) == pytest.approx(45.0)
    assert _to_signed_small_angle(46.0) == pytest.approx(-44.0)


def test_combined_rotation_deg_falls_back_to_quad_when_hough_is_none():
    assert _combined_rotation_deg(6.0, None) == pytest.approx(10.0)  # 6 snaps to 10


def test_combined_rotation_deg_snaps_to_a_multiple_of_10():
    combined = _combined_rotation_deg(6.0, 4.0)
    assert combined % 10.0 == pytest.approx(0.0)


def _draw_line_mask(shape: "tuple[int, int]", angle_deg: float) -> np.ndarray:
    """A synthetic boolean mask with a single straight line through the
    center, tilted by `angle_deg` using the SAME sign convention
    `_quad_rotation_deg` uses (the rotation that would bring the line to
    horizontal) -- built by drawing a horizontal line then rotating the
    mask itself by `-angle_deg` (skimage.transform.rotate's own
    counter-clockwise-positive convention), so a correct `_hough_angle_deg`
    implementation should recover ~`angle_deg` back out."""
    h, w = shape
    mask = np.zeros(shape, dtype=np.float64)
    rr, cc = line(h // 2, 5, h // 2, w - 5)
    mask[rr, cc] = 1.0
    rotated = sk_rotate(mask, -angle_deg, resize=False, order=1, preserve_range=True)
    return rotated > 0.5


def test_hough_angle_deg_recovers_known_tilt_sign_matches_quad_rotation_deg():
    """End-to-end sign check: a mask tilted by a known angle (using
    `_quad_rotation_deg`'s own rotation convention) should come back out of
    `_hough_angle_deg` close to that same signed angle -- this is what
    actually matters for `_combined_rotation_deg`'s circular average to be
    meaningful, more than either function's angle matching an abstract
    convention in isolation."""
    for angle in (-20.0, -5.0, 5.0, 20.0):
        mask = _draw_line_mask((101, 101), angle)
        measured = _hough_angle_deg(mask)
        assert measured is not None
        assert abs(measured - angle) < 2.0, f"angle={angle} measured={measured}"


def test_hough_angle_deg_none_for_empty_mask():
    assert _hough_angle_deg(np.zeros((20, 20), dtype=bool)) is None


def test_hough_deskew_reads_a_tilted_line_close_to_its_known_angle():
    """A small white image with a single line tilted 8 degrees, cropped by
    a perfectly axis-aligned quad (quad_angle_deg=0) -- so the whole
    correction should come from Hough's own reading of the tilted content,
    landing close to the known 8-degree tilt and snapping to a multiple of
    10."""
    size = 120
    base = np.ones((size, size), dtype=np.float64)
    rr, cc = line(size // 2, 10, size // 2, size - 10)
    base[rr, cc] = 0.0
    tilted = sk_rotate(base, -8.0, resize=False, cval=1.0, order=1, preserve_range=True)
    bgr = np.stack([tilted * 255.0] * 3, axis=-1).astype(np.uint8)

    quad = np.array([(0.0, 0.0), (size - 1.0, 0.0), (size - 1.0, size - 1.0), (0.0, size - 1.0)])
    crop, debug = hough_deskew(bgr, quad)

    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert debug.quad_angle_deg == pytest.approx(0.0)
    assert debug.hough_angle_deg is not None
    assert abs(debug.hough_angle_deg - 8.0) <= 2.5
    assert debug.combined_angle_deg % 10.0 == pytest.approx(0.0)


def test_hough_deskew_falls_back_cleanly_on_blank_crop():
    """A blank (all-white) crop has no ink for Hough to find -- hough_deskew
    should fall back to the quad's own angle rather than error."""
    bgr = np.full((40, 40, 3), 255, dtype=np.uint8)
    quad = np.array([(0.0, 0.0), (39.0, 0.0), (39.0, 39.0), (0.0, 39.0)])
    crop, debug = hough_deskew(bgr, quad)
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert debug.hough_angle_deg is None
    assert debug.combined_angle_deg == pytest.approx(0.0)
