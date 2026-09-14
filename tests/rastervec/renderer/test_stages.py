"""Every render_<stage> emits a valid one-page PDF from a minimal result."""
from __future__ import annotations

import pymupdf as fitz
import pytest

from rastervec.commons.models import Page, Segment
from rastervec.pipelines.result import FastPageResult, PipelineResult
from rastervec.commons.renderer import stages


@pytest.fixture
def result(page_meta, text, vector):
    pm = page_meta(width=200, height=150)
    v = vector(kind="l", bbox=(10.0, 10.0, 40.0, 20.0))
    seg = Segment(vectors=[v], angle=1.5)
    return PipelineResult(
        page=Page(doc_path="x.pdf", meta=pm, fitz_page=None),
        texts=[text(source="native"), text(text="ocr", source="ocr")],
        vectors=[v],
        step_durations={},
        engine="current",
        native_words=[text(source="native")],
        vectors_raw=[v],
        fast_result=FastPageResult(None, None, 0.1, {0: 0.9}, skipped_tiles=[(0, 0, 5, 5)],
                                   tile_count=4, tile_seconds=[0.01]),
        fast_passed=[[v]],
        fast_dropped_vectors=[],
        separation_buckets=[[v]],
        spatial_clusters=[[v]],
        cluster_detections=[],
        reassigned_drawing=[],
        reassigned_text=[v],
        rotated_segments=[seg],
        rotation_debug=[{
            "detection_bbox": (10.0, 10.0, 40.0, 20.0),
            "resolved_theta": 1.5, "source_theta": 0.0,
        }],
        restored_texts=[text(text="ocr", source="ocr")],
    )


@pytest.mark.parametrize("fn", [
    "render_native", "render_vectors", "render_fast", "render_separation",
    "render_clusters", "render_paddle_detect", "render_assignment", "render_rotate",
    "render_drawing", "render_ocr_results", "render_reconstructed",
])
def test_render_returns_valid_pdf(result, fn):
    data = getattr(stages, fn)(result)
    assert data[:4] == b"%PDF"
    doc = fitz.open("pdf", data)
    assert doc.page_count == 1
    doc.close()


@pytest.mark.parametrize("stage_key", [
    "native", "vectors", "fast", "separation", "clusters", "paddle_detect",
    "assignment", "rotate", "ocr", "drawing", "reconstructed",
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


def test_render_stage_layers_unknown_stage_raises(result):
    with pytest.raises(ValueError):
        stages.render_stage_layers(result, "not-a-stage")


def test_rotate_layers_include_detection_and_rotated_bbox(result):
    labels = [lbl for lbl, _, _ in stages.render_stage_layers(result, "rotate")]
    assert labels == ["detection bbox", "rotated crop bbox"]
    assert len(labels) == len(stages.STAGE_COLOR_LEGEND["rotate.pdf"])
