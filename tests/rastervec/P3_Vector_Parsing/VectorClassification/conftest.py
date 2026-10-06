from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _stub_fast_detector(monkeypatch):
    """No FAST checkpoint is available in test runs -- every test in this
    folder gets a FAST that highlights everything (score 1.0 everywhere), so
    the post-recognition text/drawing split keeps every vector it scores and
    tests unrelated to FAST see the same text/drawing they always did. A
    test that cares about FAST re-patches `FastDetector.detect` itself."""
    from rastervec.P3_Vector_Parsing.VectorClassification.fast_detect import FastDetector

    monkeypatch.setattr(
        FastDetector, "detect",
        lambda self, image: np.ones((image.height, image.width), dtype=np.float32),
    )
