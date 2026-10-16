"""Real-PaddleOCR check that the text detector returns *rotated* quads for
slanted text (DB post-process: `cv2.minAreaRect`, `det_box_type="quad"`) --
the reason `Text.quad_points` and the quad debug layers exist. Opt in via
`RASTER_PARSER_RUN_OCR_TESTS=1` (first run downloads models)."""
from __future__ import annotations

import math
import os

import numpy as np
import pymupdf as fitz
import pytest

_RUN_OCR_TESTS = os.environ.get("RASTER_PARSER_RUN_OCR_TESTS") == "1"


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


@pytest.mark.skipif(not _RUN_OCR_TESTS, reason="real PaddleOCR; opt in via RASTER_PARSER_RUN_OCR_TESTS=1")
@pytest.mark.parametrize("angle", [20.0, -30.0])
def test_detector_returns_quads_along_the_text_slant(angle):
    from raster_parser.P3_Vector_Parsing.v2VectorClass.paddle_engine import PaddleDetectBackend

    quads = PaddleDetectBackend().detect(_render_slanted_text(angle))
    assert quads, "nothing detected"
    quad = max(quads, key=lambda q: np.ptp(q[:, 0]) * np.ptp(q[:, 1]))
    # longest edge's angle (mod 180) = the text slant, not 0/90
    edges = [quad[(i + 1) % 4] - quad[i] for i in range(4)]
    longest = max(edges, key=lambda e: math.hypot(*e))
    got = math.degrees(math.atan2(longest[1], longest[0])) % 180.0
    want = angle % 180.0
    assert min(abs(got - want), 180.0 - abs(got - want)) < 5.0


def _vector_text_pdf(tmp_path) -> str:
    """One page of words, far apart (one cluster each), redrawn as vector glyphs."""
    from raster_parser.evaluation.conversion import convert_page_text_only

    doc = fitz.open()
    page = doc.new_page(width=500, height=400)
    for i, word in enumerate(("DOOR", "FRAME", "PANEL", "SCHEDULE", "LEVEL", "WALL")):
        page.insert_text(fitz.Point(40 + 220 * (i % 2), 60 + 110 * (i // 2)), word, fontsize=16)
    native = tmp_path / "native.pdf"
    doc.save(native)
    doc.close()
    out = tmp_path / "vector_text.pdf"
    convert_page_text_only(str(native), 0, str(out))
    return str(out)


@pytest.mark.skipif(not _RUN_OCR_TESTS, reason="real PaddleOCR; opt in via RASTER_PARSER_RUN_OCR_TESTS=1")
def test_workers_two_gives_the_same_output_as_one(tmp_path, monkeypatch):
    from raster_parser.core.pipeline import run_pipeline
    from raster_parser.core.settings import RunSettings
    from raster_parser.P3_Vector_Parsing.v2VectorClass import parse as v2

    # Small recognition batches, so recognition runs on the pool too (both runs use the same
    # batches; the pool workers' engines are unaffected).
    monkeypatch.setattr(v2, "OCR_BATCH_SIZE", 2)
    pdf = _vector_text_pdf(tmp_path)

    def run(workers: int):
        res = run_pipeline(pdf, 0, p2="Stub", p3="v2VectorClass",
                           settings=RunSettings(workers=workers, dump_dir=tmp_path / f"w{workers}"))
        return ([(t.text, t.bbox, t.direction, t.quad_points) for t in res.texts],
                [v.seqno for v in res.vectors])

    one = run(1)
    assert {t[0] for t in one[0]} >= {"DOOR", "FRAME", "PANEL"}
    assert run(2) == one
    assert not any((tmp_path / "w2" / "v2VectorClass").iterdir())  # spill files removed
