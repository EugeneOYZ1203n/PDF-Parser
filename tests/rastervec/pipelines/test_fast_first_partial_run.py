"""`stop_after` partial-run support on the `fast_first` pipeline."""
from __future__ import annotations

import pytest

from rastervec.pipelines.fast_first import run_pipeline


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
    assert res.fast_passed is None
    assert res.spatial_clusters is None


def test_stop_after_fast_filter(one_page_pdf):
    res = run_pipeline(one_page_pdf, 0, verbose=True, stop_after="fast_filter", enable_fast=False)
    assert "fast_filter" in res.step_durations
    assert "spatial_cluster" not in res.step_durations
    assert res.fast_passed is not None
    assert res.spatial_clusters is None


def test_invalid_stop_after(one_page_pdf):
    with pytest.raises(ValueError):
        run_pipeline(one_page_pdf, 0, stop_after="not-a-step")
