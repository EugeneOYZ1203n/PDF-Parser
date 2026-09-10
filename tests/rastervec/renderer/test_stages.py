"""Every render_<stage> emits a valid one-page PDF from a minimal result."""
from __future__ import annotations

import pymupdf as fitz
import pytest

from rastervec.models import Page
from rastervec.pipelines.result import FastPageResult, PipelineResult
from rastervec.renderer import stages


@pytest.fixture
def result(page_meta, text, vector):
    pm = page_meta(width=200, height=150)
    v = vector(kind="l", bbox=(10.0, 10.0, 40.0, 20.0))
    return PipelineResult(
        page=Page(doc_path="x.pdf", meta=pm, fitz_page=None),
        texts=[text(source="native"), text(text="ocr", source="ocr")],
        vectors=[v],
        step_durations={},
        engine="current",
        native_words=[text(source="native")],
        vectors_raw=[v],
        vectors_by_layer={"L": [v]},
        vectors_by_layer_color={"L": {(0, 0, 0): [v]}},
        text_clusters=[[[v]]],
        clustering={},
        fast_result=FastPageResult(None, None, 0.1, {0: 0.9}, skipped_tiles=[(0, 0, 5, 5)],
                                   tile_count=4, tile_seconds=[0.01]),
        fast_passed=[[v]],
        fast_dropped_vectors=[],
        word_segments=[],
        similarity_groups=[],
        unique_segments=[],
        segment_metas=[],
        unique_texts=[],
        restored_texts=[text(text="ocr", source="ocr")],
        segmentation_debug=[],
    )


@pytest.mark.parametrize("fn", [
    "render_native", "render_vectors", "render_layers", "render_layer_color_buckets",
    "render_clustering_steps", "render_text_candidates", "render_radon", "render_similarity",
    "render_fast", "render_drawing", "render_ocr_results", "render_restore",
    "render_reconstructed",
])
def test_render_returns_valid_pdf(result, fn):
    data = getattr(stages, fn)(result)
    assert data[:4] == b"%PDF"
    doc = fitz.open("pdf", data)
    assert doc.page_count == 1
    doc.close()


@pytest.mark.parametrize("stage_key", [
    "native", "vectors", "separation", "classify", "fast", "segment",
    "similarity", "ocr", "drawing", "reconstructed",
])
def test_render_stage_layers(result, stage_key):
    layers = stages.render_stage_layers(result, stage_key)
    assert layers and all(len(t) == 3 for t in layers)
    for label, hexc, data in layers:
        assert isinstance(label, str) and hexc.startswith("#")
        assert data[:4] == b"%PDF"
        doc = fitz.open("pdf", data)
        assert doc.page_count == 1
        doc.close()
