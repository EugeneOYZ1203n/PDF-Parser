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
    _grayscale,
    _hough_angle_deg,
    _hough_ink_mask,
    _minarea_angle_deg,
    _minarea_ink_mask,
    _mod90_circular_distance,
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
    mask = _hough_ink_mask(_grayscale(crop))
    assert mask[5, 5]
    assert mask.sum() > 1  # dilation thickened the single ink pixel


def test_minarea_ink_mask_flags_dark_pixels_without_dilating():
    crop = np.full((10, 10, 3), 255, dtype=np.uint8)
    crop[5, 5] = (0, 0, 0)  # a single dark pixel
    mask = _minarea_ink_mask(_grayscale(crop))
    assert mask[5, 5]
    assert mask.sum() == 1  # no dilation, unlike _hough_ink_mask


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


def test_combined_rotation_deg_falls_back_to_zero_when_both_are_none():
    assert _combined_rotation_deg(None, None) == pytest.approx(0.0)


def test_combined_rotation_deg_defaults_to_zero_when_minarea_is_none():
    # A single uncorroborated reading is no longer trusted alone -- only
    # having Hough (no minAreaRect to agree with) now defaults to 0, not the
    # old single-reading fallback.
    assert _combined_rotation_deg(6.0, None) == pytest.approx(0.0)


def test_combined_rotation_deg_defaults_to_zero_when_hough_is_none():
    assert _combined_rotation_deg(None, 6.0) == pytest.approx(0.0)


def test_combined_rotation_deg_defaults_to_zero_when_readings_disagree():
    # 6.0 and 4.0 (mod 90) are 2.0 deg apart -- beyond
    # ROTATION_AGREEMENT_TOLERANCE_DEG (1.0), so the correction is not
    # trusted even though both readings are present.
    assert _combined_rotation_deg(6.0, 4.0) == pytest.approx(0.0)


def test_combined_rotation_deg_agrees_at_the_tolerance_boundary():
    # Exactly 1.0 deg apart (mod 90) is still "agrees" -- the tolerance
    # check is inclusive. Picked so the agreeing average (6.0) snaps to a
    # distinctly non-zero 10.0, unlike the disagreement-default of 0.0, so
    # this actually distinguishes "treated as agreeing" from "defaulted".
    assert _combined_rotation_deg(5.5, 6.5) == pytest.approx(10.0)


def test_combined_rotation_deg_defaults_to_zero_just_beyond_tolerance():
    assert _combined_rotation_deg(5.5, 6.7) == pytest.approx(0.0)


def test_combined_rotation_deg_snaps_to_a_multiple_of_10_when_readings_agree():
    # 6.0 and 6.5 (mod 90) are 0.5 deg apart -- within tolerance.
    combined = _combined_rotation_deg(6.0, 6.5)
    assert combined % 10.0 == pytest.approx(0.0)


def test_mod90_circular_distance_wraps_around_the_0_90_boundary():
    assert _mod90_circular_distance(89.0, 1.0) == pytest.approx(2.0)
    assert _mod90_circular_distance(1.0, 3.0) == pytest.approx(2.0)
    assert _mod90_circular_distance(10.0, 10.0) == pytest.approx(0.0)


def _draw_line_mask(shape: "tuple[int, int]", angle_deg: float) -> np.ndarray:
    """A synthetic boolean mask with a single straight line through the
    center, tilted by `angle_deg` using the convention "the rotation that
    would bring the line to horizontal" -- built by drawing a horizontal
    line then rotating the mask itself by `-angle_deg`
    (`skimage.transform.rotate`'s own counter-clockwise-positive
    convention), so a correct `_hough_angle_deg` implementation should
    recover ~`angle_deg` back out."""
    h, w = shape
    mask = np.zeros(shape, dtype=np.float64)
    rr, cc = line(h // 2, 5, h // 2, w - 5)
    mask[rr, cc] = 1.0
    rotated = sk_rotate(mask, -angle_deg, resize=False, order=1, preserve_range=True)
    return rotated > 0.5


def test_hough_angle_deg_recovers_known_tilt_sign():
    """End-to-end sign check: a mask tilted by a known angle should come
    back out of `_hough_angle_deg` close to that same signed angle -- this
    is what actually matters for `_combined_rotation_deg`'s circular average
    to be meaningful, more than the function's angle matching an abstract
    convention in isolation."""
    for angle in (-20.0, -5.0, 5.0, 20.0):
        mask = _draw_line_mask((101, 101), angle)
        measured = _hough_angle_deg(mask)
        assert measured is not None
        assert abs(measured - angle) < 2.0, f"angle={angle} measured={measured}"


def test_hough_angle_deg_none_for_empty_mask():
    assert _hough_angle_deg(np.zeros((20, 20), dtype=bool)) is None


def _draw_block_mask(shape: "tuple[int, int]", angle_deg: float) -> np.ndarray:
    """A synthetic boolean mask with a filled horizontal bar through the
    center (closer to a real ink-block shape than `_draw_line_mask`'s
    1px line), tilted the same way -- used to verify `_minarea_angle_deg`'s
    `cv2.minAreaRect` reading lands in the same mod-90 convention
    `_hough_angle_deg` uses (verified empirically: for a filled block,
    `cv2.minAreaRect`'s own returned angle already equals `angle_deg % 90`
    to within a small fraction of a degree, no sign conversion needed)."""
    h, w = shape
    mask = np.zeros(shape, dtype=np.float64)
    mask[h // 2 - 3 : h // 2 + 3, 10 : w - 10] = 1.0
    rotated = sk_rotate(mask, -angle_deg, resize=False, order=1, preserve_range=True)
    return rotated > 0.5


def test_minarea_angle_deg_recovers_known_tilt_mod_90():
    # Angles kept within (0, 45) so `angle % 90` needs no wraparound to
    # compare directly against cv2.minAreaRect's own [0, 90) range.
    for angle in (5.0, 10.0, 20.0, 40.0):
        mask = _draw_block_mask((101, 101), angle)
        measured = _minarea_angle_deg(mask)
        assert measured is not None
        assert abs(measured - angle) < 2.0, f"angle={angle} measured={measured}"


def test_minarea_angle_deg_none_for_empty_mask():
    assert _minarea_angle_deg(np.zeros((20, 20), dtype=bool)) is None


def test_hough_deskew_defaults_to_zero_when_estimators_disagree_beyond_tolerance():
    """A small white image with a single line tilted 8 degrees, cropped by
    a perfectly axis-aligned quad. Hough's own reading is quantized to
    whole-degree bins (`_hough_angle_deg`'s `np.linspace(..., 180, ...)` --
    1-degree steps) while minAreaRect's is continuous, so the two commonly
    differ by more than `ROTATION_AGREEMENT_TOLERANCE_DEG` (1.0) even for a
    perfectly clean synthetic tilt like this one -- the correction correctly
    defaults to 0 rather than trusting either single reading alone (see
    `_combined_rotation_deg`). Each individual reading is still close to the
    known 8-degree tilt; see
    `test_hough_deskew_applies_correction_when_estimators_agree` for a case
    where the two do agree and a correction is actually applied."""
    size = 120
    base = np.ones((size, size), dtype=np.float64)
    rr, cc = line(size // 2, 10, size // 2, size - 10)
    base[rr, cc] = 0.0
    tilted = sk_rotate(base, -8.0, resize=False, cval=1.0, order=1, preserve_range=True)
    bgr = np.stack([tilted * 255.0] * 3, axis=-1).astype(np.uint8)

    quad = np.array([(0.0, 0.0), (size - 1.0, 0.0), (size - 1.0, size - 1.0), (0.0, size - 1.0)])
    crop, debug = hough_deskew(bgr, quad)

    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert debug.hough_angle_deg is not None
    assert abs(debug.hough_angle_deg - 8.0) <= 2.5
    assert debug.minarea_angle_deg is not None
    assert abs(debug.minarea_angle_deg - 8.0) <= 2.5
    assert debug.combined_angle_deg == pytest.approx(0.0)


def test_hough_deskew_applies_correction_when_estimators_agree():
    """A thicker, filled tilted block (closer to real ink than a 1px line)
    at 20 degrees: both Hough and minAreaRect land close enough together
    (within `ROTATION_AGREEMENT_TOLERANCE_DEG`) that the correction is
    actually applied, recovering the known tilt exactly."""
    size = 120
    base = np.ones((size, size), dtype=np.float64)
    base[size // 2 - 3 : size // 2 + 3, 10 : size - 10] = 0.0
    tilted = sk_rotate(base, -20.0, resize=False, cval=1.0, order=1, preserve_range=True)
    bgr = np.stack([tilted * 255.0] * 3, axis=-1).astype(np.uint8)

    quad = np.array([(0.0, 0.0), (size - 1.0, 0.0), (size - 1.0, size - 1.0), (0.0, size - 1.0)])
    crop, debug = hough_deskew(bgr, quad)

    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert debug.combined_angle_deg == pytest.approx(20.0)


def test_hough_deskew_skips_estimation_when_rotation_not_allowed():
    """`allow_rotation=False` (the caller's own gate, see `parse.py::
    _quad_allows_rotation`) should skip the grayscale/ink-mask/Hough/
    minAreaRect estimation entirely -- both angles stay `None`, the
    correction is `0.0`, and the crop comes back unrotated, even for
    content (this same tilted line) that would otherwise read a real
    tilt."""
    size = 120
    base = np.ones((size, size), dtype=np.float64)
    rr, cc = line(size // 2, 10, size // 2, size - 10)
    base[rr, cc] = 0.0
    tilted = sk_rotate(base, -8.0, resize=False, cval=1.0, order=1, preserve_range=True)
    bgr = np.stack([tilted * 255.0] * 3, axis=-1).astype(np.uint8)

    quad = np.array([(0.0, 0.0), (size - 1.0, 0.0), (size - 1.0, size - 1.0), (0.0, size - 1.0)])
    crop, debug = hough_deskew(bgr, quad, keep_debug=True, allow_rotation=False)

    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert debug.hough_angle_deg is None
    assert debug.minarea_angle_deg is None
    assert debug.combined_angle_deg == pytest.approx(0.0)
    # base_crop is still retained (it's just the axis-aligned crop, always
    # built regardless of whether rotation was attempted) -- only the two
    # ink masks are skipped, since they were never computed to begin with.
    assert debug.base_crop is not None
    assert debug.dilated_ink_mask is None
    assert debug.minarea_mask is None


def test_hough_deskew_falls_back_cleanly_on_blank_crop():
    """A blank (all-white) crop has no ink for Hough or minAreaRect to find
    -- hough_deskew should fall back to a 0-degree correction rather than
    error (there is no other angle source left once both are unavailable)."""
    bgr = np.full((40, 40, 3), 255, dtype=np.uint8)
    quad = np.array([(0.0, 0.0), (39.0, 0.0), (39.0, 39.0), (0.0, 39.0)])
    crop, debug = hough_deskew(bgr, quad)
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    assert debug.hough_angle_deg is None
    assert debug.minarea_angle_deg is None
    assert debug.combined_angle_deg == pytest.approx(0.0)


def test_hough_deskew_omits_debug_arrays_by_default():
    """`base_crop`/`dilated_ink_mask`/`minarea_mask` are debug-only (read
    back only by `scripts/debug_image_savers.py`) -- `hough_deskew` should
    not retain them unless a caller passes `keep_debug=True`, since a page-
    wide caller (`parse.py`) would otherwise hold one extra full-size array
    per detected quad for no reason on every non-debug run."""
    bgr = np.full((40, 40, 3), 255, dtype=np.uint8)
    quad = np.array([(0.0, 0.0), (39.0, 0.0), (39.0, 39.0), (0.0, 39.0)])

    _crop, debug_default = hough_deskew(bgr, quad)
    assert debug_default.base_crop is None
    assert debug_default.dilated_ink_mask is None
    assert debug_default.minarea_mask is None

    _crop, debug_kept = hough_deskew(bgr, quad, keep_debug=True)
    assert debug_kept.base_crop is not None
    assert debug_kept.dilated_ink_mask is not None
    assert debug_kept.minarea_mask is not None
