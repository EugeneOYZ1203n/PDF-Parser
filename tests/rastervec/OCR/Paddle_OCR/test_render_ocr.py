from __future__ import annotations

import os

import numpy as np
import pytest

from rastervec.models import VectorPath
from rastervec.OCR.Paddle_OCR.ocr_backend import OcrBox
from rastervec.OCR.Paddle_OCR.render_ocr import RenderOCR, render_cluster_for_ocr
from rastervec.OCR.radon import ClusterSegmentation

_RUN_OCR_TESTS = os.environ.get("RASTERVEC_RUN_OCR_TESTS") == "1"


class _FakeBackend:
    """`recognize_crops` returns fixed text per crop. `flip_texts`, if set,
    is used when the crop set is the 180-rotated one (higher-confidence)."""

    def __init__(self, texts, conf=0.9, flip_texts=None, flip_conf=0.1) -> None:
        self.texts = texts
        self.conf = conf
        self.flip_texts = flip_texts
        self.flip_conf = flip_conf
        self.calls = 0

    def recognize_crops(self, crops):
        self.calls += 1
        # crop 0 of the second call is rot90(rot90(crop0)) -> detect the flip
        is_flip = self.flip_texts is not None and self.calls == 2
        texts = self.flip_texts if is_flip else self.texts
        conf = self.flip_conf if is_flip else self.conf
        return [
            OcrBox(text=t, confidence=conf if t else 0.0, corners=[])
            for t in texts[: len(crops)]
        ]


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


def _seg(n_words=2, dpi=150) -> ClusterSegmentation:
    crops = [np.full((10, 12), 0, np.uint8) for _ in range(n_words)]
    corners = [
        [(float(i * 15), 0.0), (float(i * 15 + 12), 0.0),
         (float(i * 15 + 12), 10.0), (float(i * 15), 10.0)]
        for i in range(n_words)
    ]
    return ClusterSegmentation(0.0, 12.0, crops, corners, render_dpi=dpi)


def test_render_cluster_for_ocr_bumps_dpi_for_tiny_cluster():
    image, dpi = render_cluster_for_ocr(_rect_cluster(bbox=(0, 0, 3, 2)), dpi=72)
    assert dpi > 72
    assert min(image.size) >= 50


def test_render_cluster_for_ocr_leaves_large_cluster_dpi_unchanged():
    image, dpi = render_cluster_for_ocr(_rect_cluster(bbox=(0, 0, 400, 300)), dpi=300)
    assert dpi == 300


def test_recognize_segmented_builds_words(tmp_pdf_path):
    import pymupdf as fitz

    from rastervec.Reader.reader import Reader

    doc = fitz.open()
    doc.new_page(width=200, height=100)
    path = tmp_pdf_path(doc)

    backend = _FakeBackend(["HELLO", "WORLD"])
    render_ocr = RenderOCR(backend=backend)
    with Reader(path) as reader:
        page = reader.get_page(0)
        result = render_ocr.recognize_segmented(_seg(2), _rect_cluster(), page)

    assert result.text == "HELLO WORLD"
    assert [w.text for w in result.words] == ["HELLO", "WORLD"]
    assert result.rotation_used == 0
    assert result.ocr_bbox is not None


def test_recognize_segmented_blank_when_no_crops(tmp_pdf_path):
    import pymupdf as fitz

    from rastervec.Reader.reader import Reader

    doc = fitz.open()
    doc.new_page(width=200, height=100)
    path = tmp_pdf_path(doc)

    seg = ClusterSegmentation(0.0, 0.0, [], [], render_dpi=150)
    with Reader(path) as reader:
        page = reader.get_page(0)
        result = RenderOCR(backend=_FakeBackend([])).recognize_segmented(
            seg, _rect_cluster(), page
        )
    assert result.text == ""
    assert result.words is None


def test_recognize_segmented_uses_injected_recognize_fn(tmp_pdf_path):
    import pymupdf as fitz

    from rastervec.Reader.reader import Reader

    doc = fitz.open()
    doc.new_page(width=200, height=100)
    path = tmp_pdf_path(doc)

    calls = []

    def fake_recognize_fn(crops):
        calls.append(len(crops))
        return [OcrBox(text=t, confidence=0.9, corners=[]) for t in ["HI", "THERE"][: len(crops)]]

    # No backend.recognize_crops call should ever happen -- passing a
    # backend whose recognize_crops raises proves recognize_fn is used
    # instead.
    class _ExplodingBackend:
        def recognize_crops(self, crops):
            raise AssertionError("backend.recognize_crops should not be called")

    render_ocr = RenderOCR(backend=_ExplodingBackend(), recognize_fn=fake_recognize_fn)
    with Reader(path) as reader:
        page = reader.get_page(0)
        result = render_ocr.recognize_segmented(_seg(2), _rect_cluster(), page)

    assert result.text == "HI THERE"
    assert calls == [2, 2]  # upright pass + 180-flip pass


def test_recognize_segmented_picks_flipped_when_more_confident(tmp_pdf_path):
    import pymupdf as fitz

    from rastervec.Reader.reader import Reader

    doc = fitz.open()
    doc.new_page(width=200, height=100)
    path = tmp_pdf_path(doc)

    backend = _FakeBackend(["aa"], conf=0.2, flip_texts=["GOOD"], flip_conf=0.95)
    with Reader(path) as reader:
        page = reader.get_page(0)
        result = RenderOCR(backend=backend).recognize_segmented(_seg(1), _rect_cluster(), page)

    assert result.text == "GOOD"
    assert result.rotation_used == 180


@pytest.mark.skipif(
    not _RUN_OCR_TESTS,
    reason="real PaddleOCR round-trip; opt in via RASTERVEC_RUN_OCR_TESTS=1",
)
def test_ocr_cluster_reads_rendered_text(tmp_pdf_path):
    import pymupdf as fitz

    from rastervec.Reader.reader import Reader

    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    page.insert_text((20, 50), "HELLO", fontsize=28)
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        rp = reader.get_page(0)
        pixmap = rp.fitz_page.get_pixmap(matrix=fitz.Matrix(4, 4))
        from io import BytesIO

        from PIL import Image

        image = Image.open(BytesIO(pixmap.tobytes("png")))
        text, confidence, _bbox = RenderOCR().ocr(image)

    assert "HELLO" in text.upper()
    assert confidence > 0.5
