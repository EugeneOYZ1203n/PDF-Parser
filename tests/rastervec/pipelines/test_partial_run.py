"""`stop_after` partial-run support on the current pipeline."""
from __future__ import annotations

import pytest

from rastervec.pipelines.current import run_pipeline


@pytest.fixture
def one_page_pdf(synthetic_pdf_factory, tmp_pdf_path) -> str:
    doc = synthetic_pdf_factory([{
        "texts": [{"point": (20, 30), "text": "hello world"}],
        "drawings": [{"rects": [(10, 10, 40, 20)]}],
    }])
    return tmp_pdf_path(doc)


def test_stop_after_vectors(one_page_pdf):
    res = run_pipeline(one_page_pdf, 0, verbose=True, stop_after="vectors", enable_fast=False)
    assert set(res.step_durations) == {"read", "native", "vectors"}
    assert res.clustering is None
    assert res.word_segments is None
    assert res.text_clusters is None


def test_stop_after_classify(one_page_pdf):
    res = run_pipeline(one_page_pdf, 0, verbose=True, stop_after="classify", enable_fast=False)
    assert "classify" in res.step_durations
    assert "segment" not in res.step_durations
    assert res.word_segments is None
    assert res.clustering is not None


def test_invalid_stop_after(one_page_pdf):
    with pytest.raises(ValueError):
        run_pipeline(one_page_pdf, 0, stop_after="not-a-step")
