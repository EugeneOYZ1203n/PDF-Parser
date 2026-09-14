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
    assert res.spatial_clusters is None
    assert res.rotated_segments is None
    assert res.cluster_detections is None


def test_stop_after_clusters(one_page_pdf):
    res = run_pipeline(one_page_pdf, 0, verbose=True, stop_after="clusters", enable_fast=False)
    assert "clusters" in res.step_durations
    assert "paddle_detect" not in res.step_durations
    assert res.rotated_segments is None
    assert res.spatial_clusters is not None


def test_invalid_stop_after(one_page_pdf):
    with pytest.raises(ValueError):
        run_pipeline(one_page_pdf, 0, stop_after="not-a-step")
