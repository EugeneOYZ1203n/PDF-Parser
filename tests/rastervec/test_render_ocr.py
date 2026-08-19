from __future__ import annotations

import os

import pytest
from PIL import Image

from rastervec.helpers.render_ocr import RenderOCR
from rastervec.models import VectorPath
from rastervec.renderer import Renderer

# A real PaddleOCR round-trip needs its model weights (downloaded to
# ~/.paddlex/official_models on first use, which needs network the very
# first time) -- opt in explicitly rather than paying that cost/flakiness
# on every default test run, same spirit as this suite's other
# environment-dependent tests being skipif-gated.
_RUN_OCR_TESTS = os.environ.get("RASTERVEC_RUN_OCR_TESTS") == "1"


def test_render_rotations_count_and_size():
    image = Image.new("RGB", (40, 10), "white")
    rotations = RenderOCR().render_rotations(image, n=4)

    assert len(rotations) == 4
    assert rotations[0].size == (40, 10)  # 0 degrees: unchanged
    # 90 degrees: expand=True swaps width/height.
    assert rotations[1].size == (10, 40)


def test_render_rotations_rejects_non_positive_n():
    with pytest.raises(ValueError):
        RenderOCR().render_rotations(Image.new("RGB", (10, 10)), n=0)


def test_combine_rotation_results_empty_returns_blank():
    assert RenderOCR().combine_rotation_results([]) == ("", 0.0)


def test_combine_rotation_results_picks_highest_total_confidence_group():
    # Three near-identical "AB12" readings should outweigh one confident
    # but different single outlier reading.
    results = [
        ("AB12", 0.6, []),
        ("AB12", 0.55, []),
        ("Ab1Z", 0.5, []),  # close enough to "AB12" to group with it (case-insensitive)
        ("XYZQ", 0.99, []),
    ]
    text, confidence = RenderOCR().combine_rotation_results(results)

    assert text in ("AB12", "Ab1Z")  # winning group's best-confidence reading
    assert confidence == pytest.approx((0.6 + 0.55 + 0.5) / 3)


def test_combine_rotation_results_skips_blank_readings():
    results = [("", 0.9, []), ("", 0.8, []), ("hi", 0.1, [])]
    text, confidence = RenderOCR().combine_rotation_results(results)
    assert text == "hi"
    assert confidence == pytest.approx(0.1)


@pytest.mark.skipif(
    not _RUN_OCR_TESTS,
    reason="real PaddleOCR round-trip; opt in via RASTERVEC_RUN_OCR_TESTS=1 "
    "(downloads/loads real model weights, slow on first use)",
)
def test_ocr_cluster_reads_rendered_text(tmp_pdf_path):
    import pymupdf as fitz

    from rastervec.reader import Reader

    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    page.insert_text((20, 50), "HELLO", fontsize=28)
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        rastervec_page = reader.get_page(0)
        # Render the real inserted text glyphs back out as an image (via
        # the page itself, not VectorPath reconstruction) just to exercise
        # RenderOCR.ocr directly against a real rendered image.
        pixmap = rastervec_page.fitz_page.get_pixmap(matrix=fitz.Matrix(4, 4))
        image = Image.open(__import__("io").BytesIO(pixmap.tobytes("png")))

        text, confidence, _bbox = RenderOCR().ocr(image)

    assert "HELLO" in text.upper()
    assert confidence > 0.5


@pytest.mark.skipif(
    not _RUN_OCR_TESTS,
    reason="real PaddleOCR round-trip; opt in via RASTERVEC_RUN_OCR_TESTS=1 "
    "(downloads/loads real model weights, slow on first use)",
)
def test_ocr_cluster_end_to_end_on_vector_glyphs(tmp_pdf_path):
    import pymupdf as fitz

    from rastervec.reader import Reader

    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    page.insert_text((20, 50), "OK", fontsize=40)
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        rastervec_page = reader.get_page(0)
        # A synthetic VectorPath cluster: one big filled rect standing in
        # for a glyph, just to exercise the full ocr_cluster plumbing
        # (render_vector_cluster -> render_rotations -> ocr ->
        # combine_rotation_results) end to end, not OCR accuracy per se.
        cluster = [
            VectorPath(
                seq=0, item_index=0, kind="re", fill_rule="f",
                points=[(0, 0), (30, 30)], bbox=(0, 0, 30, 30),
                stroke_color=None, fill_color=(0, 0, 0),
                stroke_opacity=None, fill_opacity=None, stroke_width=None,
                dashes=None, closed=True, layer=None, page_index=0,
            )
        ]
        result = RenderOCR().ocr_cluster(cluster, rastervec_page, Renderer(), dpi=150)

    assert result.page_index == 0
    assert result.bbox == (0, 0, 30, 30)
    assert 0 <= result.rotation_used < 360
