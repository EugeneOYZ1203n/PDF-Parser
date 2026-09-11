from __future__ import annotations

import pymupdf as fitz
import pytest

from rastervec.models import Page
from rastervec.pipelines import _common
from rastervec.pipelines.current import STEP_NAMES, run_pipeline


@pytest.fixture(autouse=True)
def _stub_paddle_detect(monkeypatch):
    """Unit tests don't need the real PaddleOCR detection model -- return a
    fixed box per call so `paddle_detect` exercises the pipeline wiring
    without a network/model dependency."""

    def fake_detect_text_paddle(clusters, page, **kwargs):
        all_vectors = [v for cluster in clusters for v in cluster]
        return [(0.0, 0.0, 1.0, 1.0)] if all_vectors else []

    monkeypatch.setattr(_common, "detect_text_paddle", fake_detect_text_paddle)


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


def _ocr_texts(res):
    return [t for t in res.texts if t.source == "ocr"]


def test_run_pipeline_text_only_page(tmp_pdf_path):
    res = run_pipeline(_text_pdf(tmp_pdf_path), 0, enable_fast=False)
    assert isinstance(res.page, Page)
    assert [w.text for w in _native_texts(res)] == ["Hello"]
    assert _ocr_texts(res) == []
    assert res.vectors == []
    assert list(res.step_durations) == STEP_NAMES
    assert res.engine == "current"


def test_run_pipeline_verbose_toggles_intermediates(tmp_pdf_path):
    path = _drawing_pdf(tmp_pdf_path)

    lean = run_pipeline(path, 0, enable_fast=False, verbose=False)
    assert lean.vectors_raw is None
    assert lean.step_outputs is None
    assert lean.fast_passed is None

    full = run_pipeline(path, 0, enable_fast=False, verbose=True)
    assert full.vectors_raw is not None
    assert full.vectors_by_layer is not None
    assert full.step_outputs is not None and set(full.step_outputs) == set(STEP_NAMES)
    assert full.fast_passed is not None
    assert full.paddle_boxes is not None


def test_run_pipeline_drawing_pdf_detects_candidate(tmp_pdf_path):
    res = run_pipeline(_drawing_pdf(tmp_pdf_path), 0, enable_fast=False)
    # the small line survives classification and reaches the paddle_detect step
    assert res.paddle_boxes  # populated by the _stub_paddle_detect fixture


def test_verbose_text_clusters_are_tiered(tmp_pdf_path):
    res = run_pipeline(_drawing_pdf(tmp_pdf_path), 0, enable_fast=False, verbose=True)
    assert res.text_clusters
    for cluster in res.text_clusters:
        assert all(isinstance(group, list) for group in cluster)


def test_result_page_is_detached_and_open_page_reopens(tmp_pdf_path):
    res = run_pipeline(_text_pdf(tmp_pdf_path), 0, enable_fast=False)
    # the pipeline closed its Reader -- the returned page handle is dead
    assert res.page.fitz_page is None
    with res.open_page() as p:
        assert p.meta.index == 0
        pix = p.fitz_page.get_pixmap()
        assert pix.width > 0 and pix.height > 0
    # context manager closes the reopened doc; result.page stays detached
    assert res.page.fitz_page is None


def test_run_current_pipeline_threads_compute_to_fast(tmp_pdf_path, monkeypatch):
    sentinel = object()
    captured: dict = {}

    real_detect_text_fast = _common.detect_text_fast

    def spy_detect_text_fast(*args, **kwargs):
        captured["fast"] = kwargs.get("compute")
        return real_detect_text_fast(*args, **kwargs)

    monkeypatch.setattr(_common, "detect_text_fast", spy_detect_text_fast)

    run_pipeline(_text_pdf(tmp_pdf_path), 0, enable_fast=False, compute=sentinel)
    assert captured["fast"] is sentinel


def test_step_timer_propagates_when_not_verbose():
    timer = _common.StepTimer(verbose=False)
    with pytest.raises(RuntimeError):
        with timer("x"):
            raise RuntimeError("boom")
    assert "x" in timer.durations


def test_step_timer_suppresses_when_verbose():
    timer = _common.StepTimer(verbose=True)
    with timer("x"):
        raise RuntimeError("boom")
    assert timer.outcomes["x"].status == "error"
