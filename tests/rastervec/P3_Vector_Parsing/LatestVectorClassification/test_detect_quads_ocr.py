"""Real-PaddleOCR check that the text detector returns *rotated* quads for
slanted text (DB post-process: `cv2.minAreaRect`, `det_box_type="quad"`) --
the reason `Text.quad_points` and the quad debug layers exist. Opt in via
`RASTERVEC_RUN_OCR_TESTS=1` (first run downloads models)."""
from __future__ import annotations

import math
import os

import numpy as np
import pymupdf as fitz
import pytest

_RUN_OCR_TESTS = os.environ.get("RASTERVEC_RUN_OCR_TESTS") == "1"


def _render_slanted_text(angle_deg: float) -> np.ndarray:
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    center = fitz.Point(200, 150)
    page.insert_text(
        fitz.Point(110, 158), "SCHEDULE PANEL", fontsize=20,
        morph=(center, fitz.Matrix(-angle_deg)),
    )
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    doc.close()
    rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    return np.ascontiguousarray(rgb[:, :, ::-1])


@pytest.mark.skipif(not _RUN_OCR_TESTS, reason="real PaddleOCR; opt in via RASTERVEC_RUN_OCR_TESTS=1")
@pytest.mark.parametrize("angle", [20.0, -30.0])
def test_detector_returns_quads_along_the_text_slant(angle):
    from rastervec.P3_Vector_Parsing.LatestVectorClassification.paddle_engine import PaddleDetectBackend

    quads = PaddleDetectBackend().detect(_render_slanted_text(angle))
    assert quads, "nothing detected"
    quad = max(quads, key=lambda q: np.ptp(q[:, 0]) * np.ptp(q[:, 1]))
    # longest edge's angle (mod 180) = the text slant, not 0/90
    edges = [quad[(i + 1) % 4] - quad[i] for i in range(4)]
    longest = max(edges, key=lambda e: math.hypot(*e))
    got = math.degrees(math.atan2(longest[1], longest[0])) % 180.0
    want = angle % 180.0
    assert min(abs(got - want), 180.0 - abs(got - want)) < 5.0
