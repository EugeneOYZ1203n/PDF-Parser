from __future__ import annotations

import pymupdf as fitz
import pytest

from rastervec.models import Page
from rastervec.pipelines import _common
from rastervec.pipelines.current import STEP_NAMES, run_pipeline
from rastervec.pipelines.sub_pipelines import ocr as ocr_mod


@pytest.fixture(autouse=True)
def _stub_ocr(monkeypatch):
    def fake_recognize_segments(segments, *, recognize_fn=None):
        from rastervec.helpers.geometry import compute_origin, union_bbox
        from rastervec.models import Text

        texts = []
        for seg in segments:
            bbox = union_bbox([v.bbox for v in seg.vectors]) if seg.vectors else (0.0, 0.0, 1.0, 1.0)
            direction = (1.0, 0.0)
            texts.append(Text(
                text="TXT", bbox=bbox, direction=direction, origin=compute_origin(bbox, direction),
                font="", font_size=0.0, color=None, flags=0,
                ascender=None, descender=None, wmode=0,
                block_no=0, line_no=0, word_no=0, page_index=0, seqno=0,
                confidence=0.9, source="ocr",
            ))
        return texts

    monkeypatch.setattr(ocr_mod, "_recognize_segments", fake_recognize_segments)


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
    assert lean.similarity_groups is None
    assert lean.step_outputs is None
    assert lean.fast_passed is None
    assert lean.word_segments is None
    assert lean.unique_texts is None

    full = run_pipeline(path, 0, enable_fast=False, verbose=True)
    assert full.vectors_raw is not None
    assert full.vectors_by_layer is not None
    assert full.similarity_groups is not None
    assert full.step_outputs is not None and set(full.step_outputs) == set(STEP_NAMES)
    assert full.fast_passed is not None
    assert full.word_segments is not None
    assert full.unique_segments is not None
    assert full.unique_texts is not None


def test_run_pipeline_drawing_pdf_ocrs_candidate(tmp_pdf_path):
    res = run_pipeline(_drawing_pdf(tmp_pdf_path), 0, enable_fast=False)
    ocr_texts = _ocr_texts(res)
    assert ocr_texts  # the small line survives classification and gets OCR'd
    assert all(t.text == "TXT" for t in ocr_texts)


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


def test_run_current_pipeline_threads_compute_to_fast_and_ocr(tmp_pdf_path, monkeypatch):
    sentinel = object()
    captured: dict = {}

    real_detect_text_fast = _common.detect_text_fast

    def spy_detect_text_fast(*args, **kwargs):
        captured["fast"] = kwargs.get("compute")
        return real_detect_text_fast(*args, **kwargs)

    real_recognize = _common.recognize_unique_words

    def spy_recognize(*args, **kwargs):
        captured["ocr"] = kwargs.get("compute")
        return real_recognize(*args, **kwargs)

    monkeypatch.setattr(_common, "detect_text_fast", spy_detect_text_fast)
    monkeypatch.setattr(_common, "recognize_unique_words", spy_recognize)

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
