from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from rastervec.OCR.Paddle_OCR.light_backend import LightPaddleOcrBackend
from rastervec.OCR.Paddle_OCR.ocr_backend import OcrBox

_RUN_OCR_TESTS = os.environ.get("RASTERVEC_RUN_OCR_TESTS") == "1"


class _FakeRec:
    """Stand-in for paddleocr.TextRecognition -- echoes a fixed reading per
    crop, tracks the batch_size it was called with."""

    def __init__(self, text: str = "AB", score: float = 0.9) -> None:
        self.text, self.score = text, score
        self.calls: list[int] = []

    def predict(self, crops, batch_size: int = 1):
        self.calls.append(len(crops))
        return [{"rec_text": self.text, "rec_score": self.score} for _ in crops]


def _two_word_image() -> Image.Image:
    a = np.full((40, 200), 255, np.uint8)
    for x in (10, 15, 20, 25):
        a[12:28, x:x + 2] = 0
    for x in (140, 145, 150, 155):
        a[12:28, x:x + 2] = 0
    return Image.fromarray(a)


def test_candidate_angles_by_aspect():
    assert LightPaddleOcrBackend._candidate_angles(np.zeros((10, 100))) == [0]
    assert LightPaddleOcrBackend._candidate_angles(np.zeros((100, 10))) == [90, 270]


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_unrotate_box_round_trips(angle):
    g = np.zeros((6, 10), np.uint8)
    g[1, 8] = 200  # a single marked pixel in the original
    rot = LightPaddleOcrBackend._rotate_gray(g, angle)
    ys, xs = np.where(rot == 200)
    rx, ry = int(xs[0]), int(ys[0])
    rot_h, rot_w = rot.shape
    corners = LightPaddleOcrBackend._unrotate_box((rx, ry, rx + 1, ry + 1), angle, rot_w, rot_h)
    xs2 = [c[0] for c in corners]
    ys2 = [c[1] for c in corners]
    # the original pixel (8, 1) is bracketed by the unrotated box
    assert min(xs2) <= 8 <= max(xs2)
    assert min(ys2) <= 1 <= max(ys2)


def test_len_weighted_mean_conf():
    boxes = [OcrBox("AB", 1.0, [], True), OcrBox("CDEF", 0.5, [], True)]
    # (2*1.0 + 4*0.5) / 6
    assert LightPaddleOcrBackend._len_weighted_mean_conf(boxes) == pytest.approx(4 / 6)


def test_detect_builds_word_boxes_via_fake_engines(monkeypatch):
    backend = LightPaddleOcrBackend()
    fake = _FakeRec()
    monkeypatch.setattr(backend, "_rec_engine", lambda: fake)
    monkeypatch.setattr(backend, "_ori_engine", lambda: None)  # force aspect-gating

    detection = backend.detect(_two_word_image())

    assert len(detection.boxes) == 2
    assert all(b.is_word and b.text == "AB" for b in detection.boxes)
    assert detection.rotation in (0, 90, 180, 270)
    for b in detection.boxes:
        for x, y in b.corners:
            assert -1 <= x <= 201 and -1 <= y <= 41
    assert fake.calls  # recognition actually ran


def test_detect_blank_image_is_empty(monkeypatch):
    backend = LightPaddleOcrBackend()
    monkeypatch.setattr(backend, "_rec_engine", lambda: _FakeRec())
    monkeypatch.setattr(backend, "_ori_engine", lambda: None)
    detection = backend.detect(Image.new("L", (60, 60), 255))
    assert detection.boxes == []
    assert detection.rotation == 0


def test_detect_uses_doc_orientation_angle_when_confident(monkeypatch):
    backend = LightPaddleOcrBackend()
    monkeypatch.setattr(backend, "_rec_engine", lambda: _FakeRec())

    class _FakeOri:
        def predict(self, imgs):
            return [{"label_names": ["90"], "scores": [0.99]}]

    monkeypatch.setattr(backend, "_ori_engine", lambda: _FakeOri())
    detection = backend.detect(_two_word_image())
    assert detection.rotation == 90


@pytest.mark.skipif(not _RUN_OCR_TESTS, reason="real PaddleOCR rec weights; set RASTERVEC_RUN_OCR_TESTS=1")
def test_light_backend_reads_rendered_text():
    img = Image.new("L", (240, 60), 255)
    from PIL import ImageDraw

    ImageDraw.Draw(img).text((12, 12), "HELLO", fill=0)
    detection = LightPaddleOcrBackend().detect(img)
    joined = " ".join(b.text for b in detection.boxes).upper()
    assert "HELLO" in joined.replace(" ", "")
