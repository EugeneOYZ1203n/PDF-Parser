"""v1_raster_to_vec step 3: text/drawing split (copied from tests/raster_parser/test_text_drawing_parser.py)."""

from __future__ import annotations

import numpy as np

from raster_parser.phase_raster_vectorize.v1_raster_to_vec.s3_text_drawing_split import line_mask


def test_line_mask_runs_on_blank_and_line_image():
    blank = np.zeros((64, 64), dtype=np.uint8)
    out = line_mask(blank, line_min=10, grow=1, threshold=20, max_gap=5)
    assert out.shape == blank.shape
    assert out.dtype == np.uint8

    img = np.zeros((64, 64), dtype=np.uint8)
    img[30, 5:55] = 255
    out = line_mask(img, line_min=10, grow=1, threshold=20, max_gap=5)
    assert out.shape == img.shape
    assert int(out.max()) == 255


# --- the PaddleOCR angle classifier follows RunSettings.use_angle_cls ---

import sys
import types

import pytest

from raster_parser.phase_raster_vectorize.v1_raster_to_vec import s3_text_drawing_split as s3


class _Engine:
    def __init__(self, use_angle_cls: bool):
        self.use_angle_cls = use_angle_cls
        self.calls: list[dict] = []

    def ocr(self, img, **kwargs):
        self.calls.append(kwargs)
        return [None]


@pytest.mark.parametrize("flag", [True, False])
def test_make_ocr_builds_the_engine_with_the_requested_classifier(monkeypatch, flag):
    seen = {}
    fake = types.ModuleType("paddleocr")
    fake.PaddleOCR = lambda **kw: seen.update(kw)
    monkeypatch.setitem(sys.modules, "paddleocr", fake)
    s3.make_ocr("en", 0.0, use_gpu=False, use_angle_cls=flag)
    assert seen["use_angle_cls"] is flag


def test_paddle_ocr_call_only_asks_for_the_classifier_when_the_engine_has_one():
    with_cls, without_cls = _Engine(True), _Engine(False)
    s3.paddle_ocr_call(with_cls, None, cls=True)
    s3.paddle_ocr_call(without_cls, None, cls=True)
    assert with_cls.calls == [{"cls": True}]
    assert without_cls.calls == [{"cls": False}]  # no "classifier not initialized" warning per call


@pytest.mark.parametrize("flag", [True, False])
def test_tile_workers_are_started_with_the_engines_classifier_setting(monkeypatch, flag):
    seen = {}

    class FakePool:
        def __init__(self, n, initializer=None, initargs=()):
            seen["initargs"] = initargs

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, fn, job):
            raise RuntimeError("stop after the pool is built")

    monkeypatch.setattr(s3, "spawn_pool", FakePool)
    img = np.full((40, 40, 3), 255, np.uint8)
    with pytest.raises(RuntimeError, match="stop after"):
        s3.ocr_boxes(img, 0.0, "en", 20, 0, ocr=_Engine(flag), workers=2)
    assert seen["initargs"][-1] is flag
