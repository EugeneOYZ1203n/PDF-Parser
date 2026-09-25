from __future__ import annotations

import numpy as np

from rastervec.P3_Vector_Parsing.VectorClassification import fast_filter
from rastervec.P3_Vector_Parsing.VectorClassification.fast_detect import FastDetector


class _FakeImage:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height


def _half_mask_detect_tiled(self, image, **kwargs) -> np.ndarray:
    """A whole-page FAST mask that's 1.0 in the left half of the image and
    0.0 in the right half -- lets a test place one vector's bbox in each
    half to control its own individual FAST score precisely."""
    mask = np.zeros((image.height, image.width), dtype=np.float32)
    mask[:, : image.width // 2] = 1.0
    return mask


def _fake_page(page_meta, *, width: float = 200.0, height: float = 200.0):
    class _P:
        meta = page_meta(width=width, height=height)
    return _P()


def _patch_fast(monkeypatch) -> None:
    monkeypatch.setattr(fast_filter, "render_page_paths", lambda vecs, meta, dpi: _FakeImage(800, 800))
    monkeypatch.setattr(FastDetector, "detect_tiled", _half_mask_detect_tiled)


def test_detect_text_fast_keeps_whole_cluster_if_any_vector_passes(monkeypatch, vector, page_meta):
    _patch_fast(monkeypatch)
    strong = vector(bbox=(1.0, 1.0, 5.0, 5.0))  # page-left -> left half of mask (score 1.0)
    weak = vector(bbox=(150.0, 150.0, 160.0, 160.0))  # page-right -> right half (score 0.0)
    cluster = [weak, strong]

    res = fast_filter.detect_text_fast([cluster], _fake_page(page_meta), enable_fast=True)

    assert res.passed == [cluster]  # the whole cluster, including the weak-scoring vector
    assert res.dropped_vectors == []


def test_detect_text_fast_drops_whole_cluster_if_no_vector_passes(monkeypatch, vector, page_meta):
    _patch_fast(monkeypatch)
    a = vector(bbox=(150.0, 150.0, 160.0, 160.0))
    b = vector(bbox=(160.0, 160.0, 170.0, 170.0))
    cluster = [a, b]

    res = fast_filter.detect_text_fast([cluster], _fake_page(page_meta), enable_fast=True)

    assert res.passed == []
    assert res.dropped_vectors == cluster


def test_detect_text_fast_populates_cluster_counts(monkeypatch, vector, page_meta):
    _patch_fast(monkeypatch)
    passing = [vector(bbox=(1.0, 1.0, 5.0, 5.0))]
    failing = [vector(bbox=(150.0, 150.0, 160.0, 160.0))]

    res = fast_filter.detect_text_fast([passing, failing], _fake_page(page_meta), enable_fast=True)

    assert res.page_result.n_clusters == 2
    assert res.page_result.n_passed_clusters == 1


def test_detect_text_fast_disabled_passthrough(vector, page_meta):
    v = vector(bbox=(0.0, 0.0, 10.0, 10.0))
    res = fast_filter.detect_text_fast([[v]], _fake_page(page_meta), enable_fast=False)
    assert res.passed == [[v]]
    assert res.dropped_vectors == []
    assert res.page_result.n_clusters == 1
    assert res.page_result.n_passed_clusters == 1
