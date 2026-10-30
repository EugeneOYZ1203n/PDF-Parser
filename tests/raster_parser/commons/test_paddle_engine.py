"""`commons.paddle_engine`: the one PaddleOCR 2.x engine wrapper, shared by v2_vector_class and
the raster prelude (`commons.raster_text`)."""

from __future__ import annotations

import numpy as np

from raster_parser.commons import paddle_engine as pe


def test_v2_and_the_raster_prelude_use_the_shared_engine():
    from raster_parser.commons.raster_text import paddle_engine as rt
    from raster_parser.phase_vector_classify.v2_vector_class import paddle_engine as v2

    assert v2.PaddleBackend is pe.PaddleBackend and v2.OcrBox is pe.OcrBox
    assert rt.PaddleBackend is pe.PaddleBackend and rt.OcrBox is pe.OcrBox


def test_detect_returns_no_quads_when_paddle_finds_none(monkeypatch):
    class _Engine:
        def text_detector(self, bgr):
            return None, 0.0

    monkeypatch.setattr(pe.PaddleBackend, "_engine", lambda self: _Engine())
    assert pe.PaddleBackend().detect(np.zeros((4, 4, 3), np.uint8)) == []


def test_registry_has_the_junction_backend():
    from raster_parser.core.registry import VECTORIZE_BACKENDS

    assert "junction" in VECTORIZE_BACKENDS
