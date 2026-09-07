from __future__ import annotations

import numpy as np

from rastervec.config import OCR_REC_MODEL, OCR_VERSION
from rastervec.OCR.Paddle_OCR.ocr_backend import OcrBox, PaddleRecBackend, _rec_field


def test_rec_field_handles_dict_and_attr_shapes():
    assert _rec_field({"rec_text": "hi"}, "rec_text") == "hi"

    class _R:
        rec_text = "yo"

    assert _rec_field(_R(), "rec_text") == "yo"
    assert _rec_field({}, "missing") is None


def test_rec_model_pinned_to_ocr_version():
    assert OCR_REC_MODEL == f"{OCR_VERSION}_mobile_rec"


def test_recognize_crops_maps_engine_results_to_boxes(monkeypatch):
    class _FakeRec:
        def predict(self, crops, batch_size=1):
            return [{"rec_text": "AB", "rec_score": 0.7}, {"rec_text": "", "rec_score": 0.0}]

    backend = PaddleRecBackend()
    monkeypatch.setattr(backend, "_engine", lambda: _FakeRec())

    boxes = backend.recognize_crops([np.zeros((8, 10), np.uint8), np.zeros((8, 10), np.uint8)])
    assert [b.text for b in boxes] == ["AB", ""]
    assert isinstance(boxes[0], OcrBox)
    assert boxes[0].confidence == 0.7
    assert boxes[1].confidence == 0.0


def test_recognize_crops_empty_input():
    assert PaddleRecBackend().recognize_crops([]) == []
