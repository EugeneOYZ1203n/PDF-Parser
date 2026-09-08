from __future__ import annotations

import pymupdf as fitz
import pytest

from rastervec.models import Page, TextVectorResult
from rastervec.pipelines import _common
from rastervec.pipelines.current import STEP_NAMES, run_pipeline
from rastervec.pipelines.sub_pipelines import ocr as ocr_mod


class _StubRenderOCR:
    def __init__(self, backend=None, recognize_fn=None):
        self.backend = backend
        self.recognize_fn = recognize_fn

    def recognize_segmented(self, seg, cluster, page):
        return TextVectorResult(
            paths=cluster, text="TXT", confidence=0.9, bbox=(0.0, 0.0, 1.0, 1.0),
            ocr_bbox=(0.0, 0.0, 1.0, 1.0), rotation_used=0,
            page_index=page.meta.index, words=None,
        )


@pytest.fixture(autouse=True)
def _stub_ocr(monkeypatch):
    monkeypatch.setattr(ocr_mod, "RenderOCR", _StubRenderOCR)


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


def test_run_pipeline_text_only_page(tmp_pdf_path):
    res = run_pipeline(_text_pdf(tmp_pdf_path), 0, enable_fast=False)
    assert isinstance(res.page, Page)
    assert [w.text for w in res.native_words] == ["Hello"]
    assert res.text_clusters == []
    assert res.ocr_results == []
    assert res.drawing_vectors == []
    assert list(res.step_durations) == STEP_NAMES
    assert res.engine == "current"


def test_run_pipeline_verbose_toggles_intermediates(tmp_pdf_path):
    path = _drawing_pdf(tmp_pdf_path)

    lean = run_pipeline(path, 0, enable_fast=False, verbose=False)
    assert lean.vector_paths is None
    assert lean.similarity_groups is None
    assert lean.step_outputs is None
    assert lean.segmentations is None

    full = run_pipeline(path, 0, enable_fast=False, verbose=True)
    assert full.vector_paths is not None
    assert full.paths_by_layer is not None
    assert full.similarity_groups is not None
    assert full.step_outputs is not None and set(full.step_outputs) == set(STEP_NAMES)
    assert full.segmentations is not None


def test_run_pipeline_drawing_pdf_ocrs_candidate(tmp_pdf_path):
    res = run_pipeline(_drawing_pdf(tmp_pdf_path), 0, enable_fast=False)
    assert res.text_clusters  # the small line survives classification
    assert res.cluster_ocr_results
    assert all(r.resolved.text == "TXT" for r in res.cluster_ocr_results)


def test_cluster_groups_keyed_by_live_cluster_identity(tmp_pdf_path):
    res = run_pipeline(_drawing_pdf(tmp_pdf_path), 0, enable_fast=False)
    assert res.text_clusters
    assert all(id(c) in res.cluster_groups for c in res.text_clusters)


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


def test_run_current_pipeline_threads_compute_to_fast_and_ocr(tmp_pdf_path, monkeypatch):
    sentinel = object()
    captured: dict = {}

    real_detect_text_fast = _common.detect_text_fast

    def spy_detect_text_fast(*args, **kwargs):
        captured["fast"] = kwargs.get("compute")
        return real_detect_text_fast(*args, **kwargs)

    real_recognize = _common.recognize

    def spy_recognize(*args, **kwargs):
        captured["ocr"] = kwargs.get("compute")
        return real_recognize(*args, **kwargs)

    monkeypatch.setattr(_common, "detect_text_fast", spy_detect_text_fast)
    monkeypatch.setattr(_common, "recognize", spy_recognize)

    run_pipeline(_text_pdf(tmp_pdf_path), 0, enable_fast=False, compute=sentinel)
    assert captured["fast"] is sentinel
    assert captured["ocr"] is sentinel


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
