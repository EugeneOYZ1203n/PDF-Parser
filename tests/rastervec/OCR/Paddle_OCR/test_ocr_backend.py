from __future__ import annotations

import pytest

from rastervec.OCR.Paddle_OCR.ocr_backend import (
    PaddleOcrBackend,
    _undo_doc_rotation,
    _undo_doc_rotation_point,
)


def test_page_rotation_combines_doc_angle_and_textline_flip():
    backend = PaddleOcrBackend()
    assert backend._page_rotation({}) == 0
    assert backend._page_rotation({"doc_preprocessor_res": {"angle": 90}}) == 90
    assert backend._page_rotation({"textline_orientation_angles": [1, 1, 0]}) == 180
    assert backend._page_rotation({"textline_orientation_angles": [1, 0, 0]}) == 0
    assert backend._page_rotation(
        {"doc_preprocessor_res": {"angle": 90}, "textline_orientation_angles": [1, 1]}
    ) == 270


def test_undo_doc_rotation_zero_angle_is_identity():
    points = [(1.0, 2.0), (3.0, 4.0)]
    assert _undo_doc_rotation(points, 0, (100.0, 50.0)) == points


@pytest.mark.parametrize("angle", [90, 180, 270])
def test_undo_doc_rotation_round_trips_corners(angle):
    # Paddle's doc-preprocessor rotates the WxH original into a
    # doc-corrected image (swapped to HxW for 90/270) via
    # cv2.getRotationMatrix2D before detection ever runs; _undo_doc_rotation
    # must invert that exactly. Verify by rotating the four corners of a
    # WxH image forward (via the same matrix rastervec's docstring derives)
    # and checking the inverse recovers the originals.
    w, h = 40.0, 20.0
    corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]

    if angle == 90:
        rotated = [(y, w - x) for x, y in corners]
    elif angle == 180:
        rotated = [(w - x, h - y) for x, y in corners]
    else:  # 270
        rotated = [(h - y, x) for x, y in corners]

    recovered = _undo_doc_rotation(rotated, angle, (w, h))
    for (ox, oy), (rx, ry) in zip(corners, recovered):
        assert rx == pytest.approx(ox)
        assert ry == pytest.approx(oy)


def test_undo_doc_rotation_point_unknown_angle_is_identity():
    assert _undo_doc_rotation_point(5.0, 6.0, 45, 100.0, 50.0) == (5.0, 6.0)
