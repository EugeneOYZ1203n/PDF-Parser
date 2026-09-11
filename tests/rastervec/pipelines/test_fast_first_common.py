from __future__ import annotations

import pymupdf as fitz
import pytest

from rastervec.models import Page
from rastervec.pipelines import _fast_first_common
from rastervec.pipelines.fast_first import STEP_NAMES, run_pipeline


@pytest.fixture(autouse=True)
def _stub_paddle_detect(monkeypatch):
    """Unit tests don't need the real PaddleOCR detection model -- return a
    fixed box per call so `paddle_detect` exercises the pipeline wiring
    without a network/model dependency."""

    def fake_detect_text_paddle(clusters, page, **kwargs):
        all_vectors = [v for cluster in clusters for v in cluster]
        return [(0.0, 0.0, 1.0, 1.0)] if all_vectors else []

    monkeypatch.setattr(_fast_first_common, "detect_text_paddle", fake_detect_text_paddle)


def _text_pdf(tmp_pdf_path):
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    page.insert_text((10, 20), "Hello", fontsize=10)
    return tmp_pdf_path(doc)


def _drawing_pdf(tmp_pdf_path):
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    line = page.new_shape()
    line.draw_line((10, 10), (16, 16))
    line.finish(color=(0, 0, 0), width=2)
    line.commit()
    return tmp_pdf_path(doc)


def _native_texts(res):
    return [t for t in res.texts if t.source == "native"]


def test_run_pipeline_text_only_page(tmp_pdf_path):
    res = run_pipeline(_text_pdf(tmp_pdf_path), 0, enable_fast=False)

    assert isinstance(res.page, Page)
    assert [w.text for w in _native_texts(res)] == ["Hello"]
    assert res.vectors == []
    assert list(res.step_durations) == STEP_NAMES
    assert res.engine == "fast_first"


def test_run_pipeline_verbose_toggles_intermediates(tmp_pdf_path):
    path = _drawing_pdf(tmp_pdf_path)

    lean = run_pipeline(path, 0, enable_fast=False, verbose=False)
    assert lean.vectors_raw is None
    assert lean.step_outputs is None
    assert lean.fast_passed is None
    assert lean.spatial_clusters is None

    full = run_pipeline(path, 0, enable_fast=False, verbose=True)
    assert full.vectors_raw is not None
    assert full.step_outputs is not None and set(full.step_outputs) == set(STEP_NAMES)
    assert full.fast_passed is not None
    assert full.spatial_clusters is not None
    assert full.paddle_boxes is not None


def test_run_pipeline_drawing_pdf_reaches_spatial_cluster(tmp_pdf_path):
    res = run_pipeline(_drawing_pdf(tmp_pdf_path), 0, enable_fast=False, verbose=True)
    # the drawn line survives the (disabled) FAST filter and forms one cluster
    assert res.spatial_clusters
    assert res.paddle_boxes  # populated by the _stub_paddle_detect fixture


def test_drawing_output_is_fast_dropped_vectors_sorted_by_seqno(tmp_pdf_path, monkeypatch):
    """`fast_first`'s drawing output has no classification drops (there is
    no classification step) -- it's exactly `fast_filter`'s own drops,
    sorted by seqno."""
    class _FakeFilterResult:
        passed = []
        dropped = None
        page_result = None

    def fake_filter_vectors_fast(vectors, page, **kwargs):
        result = _FakeFilterResult()
        result.dropped = sorted(vectors, key=lambda v: -v.seqno)  # deliberately out of order
        return result

    monkeypatch.setattr(_fast_first_common, "filter_vectors_fast", fake_filter_vectors_fast)

    res = run_pipeline(_drawing_pdf(tmp_pdf_path), 0, enable_fast=False)
    assert [v.seqno for v in res.vectors] == sorted(v.seqno for v in res.vectors)


def test_result_page_is_detached_and_open_page_reopens(tmp_pdf_path):
    res = run_pipeline(_text_pdf(tmp_pdf_path), 0, enable_fast=False)
    assert res.page.fitz_page is None
    with res.open_page() as p:
        assert p.meta.index == 0
        pix = p.fitz_page.get_pixmap()
        assert pix.width > 0 and pix.height > 0
    assert res.page.fitz_page is None
