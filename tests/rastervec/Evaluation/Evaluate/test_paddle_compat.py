"""Unit tests for the PaddleOCR 2.x->3.x compat shim -- no models, no
network: the real inner engine + rec adapter are monkeypatched."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rastervec.Evaluation.Evaluate import _paddle_compat


class _FakeEngine:
    """Stand-in for a real 3.x `paddleocr.PaddleOCR`."""

    last_kwargs: dict | None = None

    def __init__(self, **kwargs):
        _FakeEngine.last_kwargs = kwargs

    def predict(self, img, **kwargs):
        return [
            {
                "rec_texts": ["HELLO", "WORLD"],
                "rec_scores": [0.9, 0.8],
                "rec_polys": [
                    [[0, 0], [10, 0], [10, 5], [0, 5]],
                    [[20, 0], [30, 0], [30, 5], [20, 5]],
                ],
            }
        ]


def _boom():
    raise RuntimeError("no models in tests")


@pytest.fixture
def shim(monkeypatch):
    fake_paddleocr = SimpleNamespace(PaddleOCR=_FakeEngine)
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddleocr)
    monkeypatch.setattr(_paddle_compat, "_original", None)
    monkeypatch.setattr(_paddle_compat, "_TextRecognizerAdapter", _boom)
    _paddle_compat.install()
    assert fake_paddleocr.PaddleOCR is _paddle_compat._PaddleOCRv2Compat
    yield
    _paddle_compat.uninstall()


def test_ctor_translates_and_drops_legacy_kwargs(shim):
    _paddle_compat._PaddleOCRv2Compat(
        use_angle_cls=True,
        lang="en",
        use_gpu=False,
        show_log=False,
        det_limit_side_len=4096,
        det_limit_type="max",
        drop_score=0.3,
    )
    kw = _FakeEngine.last_kwargs
    assert "use_gpu" not in kw and "show_log" not in kw and "drop_score" not in kw
    assert kw["text_rec_score_thresh"] == 0.3
    assert kw["text_det_limit_side_len"] == 4096
    assert kw["text_det_limit_type"] == "max"
    assert kw["use_textline_orientation"] is True
    assert kw["lang"] == "en"
    assert kw["use_doc_orientation_classify"] is False


def test_ocr_reshapes_to_2x_nested_shape(shim):
    ocr = _paddle_compat._PaddleOCRv2Compat(use_angle_cls=False, lang="en")
    result = ocr.ocr(np.zeros((5, 30, 3), dtype=np.uint8), cls=True, det=False)

    assert len(result) == 1  # one "page"
    lines = result[0]
    assert len(lines) == 2
    box, rec = lines[0]
    assert rec == ("HELLO", pytest.approx(0.9))
    assert box[0] == [0.0, 0.0] and box[2] == [10.0, 5.0]


def test_ocr_empty_result(shim, monkeypatch):
    ocr = _paddle_compat._PaddleOCRv2Compat(use_angle_cls=False)
    monkeypatch.setattr(ocr._engine, "predict", lambda *a, **k: [])
    assert ocr.ocr(np.zeros((2, 2, 3), dtype=np.uint8)) == [[]]


def test_text_recognizer_missing_raises_attributeerror(shim):
    ocr = _paddle_compat._PaddleOCRv2Compat(use_angle_cls=False)
    assert getattr(ocr, "text_recognizer", None) is None
