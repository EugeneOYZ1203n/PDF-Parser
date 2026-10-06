from __future__ import annotations

import cv2
import numpy as np
import pytest

from rastervec.P3_Vector_Parsing.LatestVectorClassification import paddle_engine as pe

_BAR = np.array([[100.0, 190.0], [300.0, 190.0], [300.0, 210.0], [100.0, 210.0]])


def _rotated_bar(angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """A 400x400 white BGR image with a solid 200x20 black bar whose
    reading direction (left -> right in the unrotated image) runs at
    `angle_deg` (image y-down), with a white notch near its *end* so the
    reading direction is recoverable from the crop. Returns `(image,
    quad)`, `quad` the bar's 4 corners after rotation."""
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (100, 190), (300, 210), (0, 0, 0), -1)
    cv2.rectangle(img, (270, 194), (290, 206), (255, 255, 255), -1)  # notch at the end
    # cv2's positive angle is visually counter-clockwise; a y-down direction
    # angle of +a is visually clockwise, hence -a.
    m = cv2.getRotationMatrix2D((200.0, 200.0), -angle_deg, 1.0)
    out = cv2.warpAffine(img, m, (400, 400), borderValue=(255, 255, 255))
    quad = _BAR @ m[:, :2].T + m[:, 2]
    return out, quad


def _start_vs_end_ink(crop: np.ndarray) -> tuple[float, float]:
    ink = crop.min(axis=2) < 128
    w = ink.shape[1]
    return float(ink[:, : w // 5].mean()), float(ink[:, -(w // 5):].mean())


@pytest.mark.parametrize("angle", [0.0, 30.0, -45.0, 60.0])
def test_quad_long_edge_angle_follows_the_bar(angle):
    _img, quad = _rotated_bar(angle)
    assert pe.quad_long_edge_angle(quad) == pytest.approx(pe.normalize_rotation(angle), abs=1e-6)


def test_quad_long_edge_angle_is_order_independent():
    _img, quad = _rotated_bar(30.0)
    assert pe.quad_long_edge_angle(np.roll(quad, 1, axis=0)) == pytest.approx(30.0, abs=1e-6)


@pytest.mark.parametrize("angle", [30.0, 120.0])
def test_upright_crop_rotates_the_quad_horizontal(angle):
    img, quad = _rotated_bar(angle)
    crop = pe.upright_crop(img, quad, pe.quad_long_edge_angle(quad))
    h, w = crop.shape[:2]
    assert w > 5 * h  # long side horizontal: ~210 x ~21 plus the border
    ink = crop.min(axis=2) < 128
    rows = np.nonzero(ink.any(axis=1))[0]
    cols = np.nonzero(ink.any(axis=0))[0]
    assert rows.size and cols.size
    # The bar fills its crop horizontally and stays a thin band vertically --
    # a rotation, not a re-boxed envelope (which would be far taller).
    assert (cols[-1] - cols[0]) > 180 and (rows[-1] - rows[0]) < 30
    start, end = _start_vs_end_ink(crop)
    if angle == 30.0:
        assert start > end  # reads left -> right: the notch is at the right
    else:
        # 120 deg normalises to -60: the crop is upright but upside down --
        # the 0/180 classifier's job, not the rotation's.
        assert end > start


def test_upright_crop_zero_angle_matches_axis_aligned_region():
    img, quad = _rotated_bar(0.0)
    crop = pe.upright_crop(img, quad, 0.0)
    assert crop.shape[0] < 40 and crop.shape[1] > 200


def test_quad_region_is_white_outside_the_image():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    quad = np.array([[15.0, 15.0], [25.0, 15.0], [25.0, 19.0], [15.0, 19.0]])
    region, centre = pe.quad_region(img, quad)
    assert region.shape[0] == region.shape[1]
    assert (region[-1, -1] == 255).all()  # beyond the image -> white
    assert centre[0] == pytest.approx(region.shape[1] // 2 + 0.0, abs=1.0)


def test_reorder_quad_reading_puts_reading_edge_first():
    quad = ((10.0, 5.0), (0.0, 5.0), (0.0, 0.0), (10.0, 0.0))
    assert pe.reorder_quad_reading(quad, 0.0) == ((0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0))
    assert pe.reorder_quad_reading(quad, 180.0) == ((10.0, 5.0), (0.0, 5.0), (0.0, 0.0), (10.0, 0.0))
    # Vertical text reading downwards (90 deg, y-down): p0 -> p1 runs down.
    vq = pe.reorder_quad_reading(((0.0, 0.0), (5.0, 0.0), (5.0, 10.0), (0.0, 10.0)), 90.0)
    assert vq[0][1] < vq[1][1] and vq[0][0] == vq[1][0]


def test_recognize_crops_applies_the_classifier_flip(monkeypatch):
    seen: dict = {}

    class _Engine:
        def text_classifier(self, imgs):
            return imgs, [("180", 0.99), ("0", 0.99)], 0.0

        def text_recognizer(self, imgs):
            seen["imgs"] = imgs
            return [("AB", 0.9), ("CD", 0.8)], 0.0

    monkeypatch.setattr(pe.PaddleRecBackend, "_engine", lambda self: _Engine())
    a = np.zeros((4, 6, 3), dtype=np.uint8)
    a[0, 0] = 255  # marker at the top-left (RGB)
    b = np.zeros((4, 6, 3), dtype=np.uint8)
    boxes = pe.PaddleRecBackend().recognize_crops([a, b])
    assert [(x.text, x.flip_deg) for x in boxes] == [("AB", 180), ("CD", 0)]
    assert seen["imgs"][0][-1, -1].tolist() == [255, 255, 255]  # flipped 180 before recognition


def test_recognize_crops_raw_never_flips(monkeypatch):
    class _Engine:
        def text_recognizer(self, imgs):
            return [("", 0.3) for _ in imgs], 0.0

    monkeypatch.setattr(pe.PaddleRecBackend, "_engine", lambda self: _Engine())
    boxes = pe.PaddleRecBackend().recognize_crops_raw([np.zeros((2, 2, 3), dtype=np.uint8)])
    assert boxes[0].flip_deg == 0 and boxes[0].confidence == 0.0  # blank read -> 0 confidence


def test_normalize_rotation_range():
    assert pe.normalize_rotation(90.0) == -90.0
    assert pe.normalize_rotation(179.0) == -1.0
    assert pe.normalize_rotation(-91.0) == 89.0


def test_score_penalises_single_characters_and_zeroes_blanks():
    assert pe.score(pe.OcrBox(text="", confidence=0.9)) == 0.0
    assert pe.score(pe.OcrBox(text="I", confidence=0.9)) == pytest.approx(0.45)
    assert pe.score(pe.OcrBox(text="IN", confidence=0.9)) == 0.9
