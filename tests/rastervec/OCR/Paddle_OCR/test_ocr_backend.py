from __future__ import annotations

import numpy as np

from rastervec.config import OCR_LANG, OCR_VERSION
from rastervec.OCR.Paddle_OCR.ocr_backend import (
    OcrBox,
    PaddleRecBackend,
    _recognize_crops_job,
)


def test_backend_key_uses_config_defaults():
    assert PaddleRecBackend().key == (OCR_VERSION, OCR_LANG)


def test_recognize_crops_maps_engine_results_to_boxes(monkeypatch):
    class _FakeEngine:
        def text_recognizer(self, crops):
            return [("AB", 0.7), ("", 0.0)], 0.0

    backend = PaddleRecBackend()
    monkeypatch.setattr(backend, "_engine", lambda: _FakeEngine())

    boxes = backend.recognize_crops([np.zeros((8, 10, 3), np.uint8), np.zeros((8, 10, 3), np.uint8)])
    assert [b.text for b in boxes] == ["AB", ""]
    assert isinstance(boxes[0], OcrBox)
    assert boxes[0].confidence == 0.7
    assert boxes[1].confidence == 0.0


def test_recognize_crops_accepts_bare_list_result(monkeypatch):
    class _FakeEngine:
        def text_recognizer(self, crops):
            return [("HI", 0.9)]

    backend = PaddleRecBackend()
    monkeypatch.setattr(backend, "_engine", lambda: _FakeEngine())
    boxes = backend.recognize_crops([np.zeros((8, 10, 3), np.uint8)])
    assert boxes[0].text == "HI"
    assert boxes[0].confidence == 0.9


def test_recognize_crops_empty_input():
    assert PaddleRecBackend().recognize_crops([]) == []


def test_recognize_crops_job_delegates_to_backend(monkeypatch):
    calls = []

    def fake_recognize_crops(self, crops):
        calls.append((self.key, len(crops)))
        return [OcrBox(text="OK", confidence=1.0, corners=[])]

    monkeypatch.setattr(PaddleRecBackend, "recognize_crops", fake_recognize_crops)
    crops = [np.zeros((8, 10, 3), np.uint8)]
    boxes = _recognize_crops_job(crops, ocr_version="PP-OCRvX", lang="de")
    assert calls == [(("PP-OCRvX", "de"), 1)]
    assert boxes[0].text == "OK"
