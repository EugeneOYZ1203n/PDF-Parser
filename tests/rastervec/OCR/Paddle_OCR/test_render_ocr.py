from __future__ import annotations

import os

import pytest
from PIL import Image

from rastervec.OCR.Paddle_OCR.ocr_backend import OcrBox, OcrDetection
from rastervec.OCR.Paddle_OCR.render_ocr import RenderOCR
from rastervec.models import VectorPath

# A real PaddleOCR round-trip needs its model weights (downloaded to
# ~/.paddlex/official_models on first use, which needs network the very
# first time) -- opt in explicitly rather than paying that cost/flakiness
# on every default test run, same spirit as this suite's other
# environment-dependent tests being skipif-gated.
_RUN_OCR_TESTS = os.environ.get("RASTERVEC_RUN_OCR_TESTS") == "1"


class _FakeBackend:
    """A no-engine OcrBackend stand-in -- lets RenderOCR's own
    orchestration (word-building, pixel->page mapping, join logic) be
    unit-tested without needing Paddle or Tesseract installed."""

    def __init__(self, detection: OcrDetection) -> None:
        self.detection = detection

    def detect(self, image):
        return self.detection


def _rect_cluster(bbox=(0, 0, 40, 20)) -> list[VectorPath]:
    return [
        VectorPath(
            seq=0, item_index=0, kind="re", fill_rule="f",
            points=[(bbox[0], bbox[1]), (bbox[2], bbox[3])], bbox=bbox,
            stroke_color=None, fill_color=(0, 0, 0),
            stroke_opacity=None, fill_opacity=None, stroke_width=None,
            dashes=None, closed=True, layer=None, page_index=0,
        )
    ]


def test_ocr_cluster_builds_words_from_backend_boxes(tmp_pdf_path):
    import pymupdf as fitz

    from rastervec.Reader.reader import Reader

    doc = fitz.open()
    doc.new_page(width=200, height=100)
    path = tmp_pdf_path(doc)

    detection = OcrDetection(
        boxes=[
            OcrBox(text="HELLO", confidence=0.9, corners=[(0, 0), (10, 0), (10, 5), (0, 5)], is_word=True),
            OcrBox(text="WORLD", confidence=0.8, corners=[(12, 0), (22, 0), (22, 5), (12, 5)], is_word=True),
        ],
        rotation=0,
    )
    render_ocr = RenderOCR(backend=_FakeBackend(detection))

    with Reader(path) as reader:
        rastervec_page = reader.get_page(0)
        cluster = _rect_cluster()
        result = render_ocr.ocr_cluster(cluster, rastervec_page, dpi=150)

    assert result.text == "HELLO WORLD"
    assert result.words is not None
    assert [w.text for w in result.words] == ["HELLO", "WORLD"]
    assert result.rotation_used == 0


class _CapturingBackend:
    """Records the image it was handed, so a test can assert on the actual
    rendered pixel size ocr_cluster produced."""

    def __init__(self, detection: OcrDetection) -> None:
        self.detection = detection
        self.last_image: Image.Image | None = None

    def detect(self, image):
        self.last_image = image
        return self.detection


def test_ocr_cluster_bumps_dpi_so_tiny_cluster_still_renders_50px(tmp_pdf_path):
    import pymupdf as fitz

    from rastervec.Reader.reader import Reader

    doc = fitz.open()
    doc.new_page(width=200, height=100)
    path = tmp_pdf_path(doc)

    # A few PDF points across -- at the requested dpi=72 (1x) this would
    # render to well under 50px on its shorter side without the bump.
    tiny_cluster = _rect_cluster(bbox=(0, 0, 3, 2))
    backend = _CapturingBackend(OcrDetection(boxes=[], rotation=0))
    render_ocr = RenderOCR(backend=backend)

    with Reader(path) as reader:
        rastervec_page = reader.get_page(0)
        render_ocr.ocr_cluster(tiny_cluster, rastervec_page, dpi=72)

    assert backend.last_image is not None
    assert min(backend.last_image.size) >= 50


def test_ocr_cluster_words_none_when_nothing_detected(tmp_pdf_path):
    import pymupdf as fitz

    from rastervec.Reader.reader import Reader

    doc = fitz.open()
    doc.new_page(width=200, height=100)
    path = tmp_pdf_path(doc)

    render_ocr = RenderOCR(backend=_FakeBackend(OcrDetection(boxes=[], rotation=0)))

    with Reader(path) as reader:
        rastervec_page = reader.get_page(0)
        result = render_ocr.ocr_cluster(_rect_cluster(), rastervec_page, dpi=150)

    assert result.text == ""
    assert result.words is None


@pytest.mark.skipif(
    not _RUN_OCR_TESTS,
    reason="real PaddleOCR round-trip; opt in via RASTERVEC_RUN_OCR_TESTS=1 "
    "(downloads/loads real model weights, slow on first use)",
)
def test_ocr_cluster_reads_rendered_text(tmp_pdf_path):
    import pymupdf as fitz

    from rastervec.Reader.reader import Reader

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

    from rastervec.Reader.reader import Reader

    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    page.insert_text((20, 50), "OK", fontsize=40)
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        rastervec_page = reader.get_page(0)
        # A synthetic VectorPath cluster: one big filled rect standing in
        # for a glyph, just to exercise the full ocr_cluster plumbing
        # (render_vector_cluster -> predict -> _page_rotation) end to end,
        # not OCR accuracy per se.
        cluster = [
            VectorPath(
                seq=0, item_index=0, kind="re", fill_rule="f",
                points=[(0, 0), (30, 30)], bbox=(0, 0, 30, 30),
                stroke_color=None, fill_color=(0, 0, 0),
                stroke_opacity=None, fill_opacity=None, stroke_width=None,
                dashes=None, closed=True, layer=None, page_index=0,
            )
        ]
        result = RenderOCR().ocr_cluster(cluster, rastervec_page, dpi=150)

    assert result.page_index == 0
    assert result.bbox == (0, 0, 30, 30)
    assert 0 <= result.rotation_used < 360


def test_render_ocr_with_light_backend_flows_word_boxes(tmp_pdf_path):
    """A LightPaddleOcrBackend (fake engine) plugged into RenderOCR.ocr_cluster
    -- word corners map through pixel_to_page_bbox into the cluster page bbox."""
    import pymupdf as fitz

    from rastervec.OCR.Paddle_OCR.light_backend import LightPaddleOcrBackend
    from rastervec.Reader.reader import Reader

    class _FakeRec:
        def predict(self, crops, batch_size: int = 1):
            return [{"rec_text": "X", "rec_score": 0.8} for _ in crops]

    doc = fitz.open()
    doc.new_page(width=200, height=100)
    path = tmp_pdf_path(doc)

    backend = LightPaddleOcrBackend()
    backend._rec_engine = lambda: _FakeRec()
    backend._ori_engine = lambda: None

    # a cluster of thin strokes so ink segmentation finds real word boxes
    cluster = [
        VectorPath(
            seq=0, item_index=i, kind="re", fill_rule="f",
            points=[(x, 10), (x + 1, 25)], bbox=(x, 10, x + 1, 25),
            stroke_color=None, fill_color=(0, 0, 0), stroke_opacity=None,
            fill_opacity=None, stroke_width=None, dashes=None, closed=True,
            layer=None, page_index=0,
        )
        for i, x in enumerate((20, 23, 26, 29))
    ]

    with Reader(path) as reader:
        rastervec_page = reader.get_page(0)
        result = RenderOCR(backend=backend).ocr_cluster(cluster, rastervec_page, dpi=200)

    assert result.words is not None
    px0, py0, px1, py1 = -20.0, -10.0, 60.0, 40.0
    for w in result.words:
        wx0, wy0, wx1, wy1 = w.bbox
        assert px0 <= wx0 <= wx1 <= px1
        assert py0 <= wy0 <= wy1 <= py1
