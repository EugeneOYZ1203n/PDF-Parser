from __future__ import annotations

import os
import shutil

import pytest

from rastervec.helpers.ocr_backend import (
    PaddleOcrBackend,
    TesseractOcrBackend,
    _undo_doc_rotation,
    _undo_doc_rotation_point,
)

# A real PaddleOCR round-trip needs its model weights (downloaded to
# ~/.paddlex/official_models on first use, which needs network the very
# first time) -- opt in explicitly, same spirit as this suite's other
# environment-dependent tests being skipif-gated.
_RUN_OCR_TESTS = os.environ.get("RASTERVEC_RUN_OCR_TESTS") == "1"
_HAS_TESSERACT = shutil.which("tesseract") is not None


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


@pytest.mark.skipif(
    not (_RUN_OCR_TESTS and _HAS_TESSERACT),
    reason="real Tesseract round-trip; opt in via RASTERVEC_RUN_OCR_TESTS=1 "
    "and requires the tesseract.exe binary installed",
)
def test_tesseract_backend_detects_word_level_boxes():
    import pymupdf as fitz
    from PIL import Image

    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    page.insert_text((20, 50), "HELLO WORLD", fontsize=28)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(4, 4))
    import io

    image = Image.open(io.BytesIO(pixmap.tobytes("png")))

    detection = TesseractOcrBackend().detect(image)

    assert len(detection.boxes) >= 2  # "HELLO" and "WORLD" as separate word boxes
    assert all(box.is_word for box in detection.boxes)
    # Exact recognition accuracy on a tiny synthetic render isn't the point
    # here (that's Tesseract's own accuracy, not this wiring) -- just check
    # each word came back as its own box with a plausible reading.
    joined = " ".join(box.text for box in detection.boxes).upper()
    assert "HELL" in joined and "WORL" in joined
