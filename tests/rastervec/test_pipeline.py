from __future__ import annotations

import numpy as np
import pymupdf as fitz
import pytest

from rastervec.models import ClusterOcrResult, DrawingVector, Page, TextVectorResult, TextWord, VectorPath
from rastervec.pipeline import (
    ClusteringStageResult,
    Pipeline,
    PipelineContext,
    StageSpec,
    _run_drawing_vectors,
    _run_rotation_verify,
    _run_spatial_regroup,
    _sample_mask,
    _text_aspect_ratio,
)
from rastervec.Reader.reader import Reader
from rastervec.Vector_Classification.classification import CategoryResult, StepResult

_EXPECTED_STAGE_KEYS = [
    "reader",
    "native",
    "vector_extract",
    "layer_separation",
    "color_separation",
    "clustering",
    "text_candidates",
    "unique_clusters",
    "fast_text_detect",
    "spatial_regroup",
    "ocr_compare",
    "rotation_verify",
    "drawing_vectors",
]


def test_run_page_reader_and_native_ok(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory(
        [{"texts": [{"point": (10, 20), "text": "Hello"}]}]
    )
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        outputs = Pipeline().run_page(reader, 0)

    assert [o.key for o in outputs] == _EXPECTED_STAGE_KEYS
    assert all(o.status == "ok" for o in outputs)

    by_key = {o.key: o for o in outputs}

    assert isinstance(by_key["reader"].data, Page)
    assert by_key["reader"].data.meta.index == 0

    words: list[TextWord] = by_key["native"].data
    assert [w.text for w in words] == ["Hello"]

    assert isinstance(by_key["vector_extract"].data, list)
    assert isinstance(by_key["layer_separation"].data, dict)
    assert isinstance(by_key["color_separation"].data, dict)
    assert isinstance(by_key["clustering"].data, dict)
    assert by_key["text_candidates"].data == []  # no vector drawings on this text-only page
    fast_result = by_key["fast_text_detect"].data
    assert fast_result.page_image is None  # zero vector paths -- FastDetector never even constructed
    assert by_key["ocr_compare"].data == []
    assert isinstance(by_key["drawing_vectors"].data, list)


def test_run_page_final_stage_stops_early_and_skips_ocr_engine(
    synthetic_pdf_factory, tmp_pdf_path, monkeypatch,
):
    doc = synthetic_pdf_factory([{"texts": [{"point": (10, 20), "text": "Hello"}]}])
    path = tmp_pdf_path(doc)

    def _boom(*args, **kwargs):
        raise AssertionError("RenderOCR must never be constructed once final_stage stops before it")

    monkeypatch.setattr("rastervec.pipeline.RenderOCR", _boom)

    with Reader(path) as reader:
        outputs = Pipeline().run_page(reader, 0, final_stage="fast_text_detect")

    # everything up to and including fast_text_detect, but not
    # spatial_regroup, ocr_compare (which would construct RenderOCR),
    # rotation_verify, or drawing_vectors
    assert [o.key for o in outputs] == _EXPECTED_STAGE_KEYS[:-4]
    assert all(o.status == "ok" for o in outputs)


def test_run_page_final_stage_unknown_raises(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory([{}])
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        with pytest.raises(ValueError):
            Pipeline().run_page(reader, 0, final_stage="not_a_real_stage")


def test_run_page_vector_stages_on_drawing_pdf(tmp_pdf_path):
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)

    # a full-page rect -- dimension 200 exceeds MAX_DIMENSION_FRACTION
    # (10%) of the page's smaller side (100), so filter_large_items (the
    # 1st pipeline step) drops it as border/frame geometry.
    panel = page.new_shape()
    panel.draw_rect(fitz.Rect(0, 0, 200, 100))
    panel.finish(color=(0, 0, 0))
    panel.commit()

    # a real line, small enough (max dimension < 10% of the page's smaller
    # side, 100) to survive the large-item/large-group filters -- ends up
    # in a spatial cluster.
    line = page.new_shape()
    line.draw_line((10, 10), (16, 16))
    line.finish(color=(0, 0, 0), width=2)
    line.commit()

    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        outputs = Pipeline().run_page(reader, 0)

    by_key = {o.key: o for o in outputs}
    assert all(o.status == "ok" for o in outputs)

    paths: list[VectorPath] = by_key["vector_extract"].data
    assert len(paths) > 0

    clustering_results: dict = by_key["clustering"].data
    for result in clustering_results.values():
        assert isinstance(result, ClusteringStageResult)
        assert len(result.steps) == 12
        assert all(isinstance(step, StepResult) for step in result.steps)
        assert all("kept" in step.categories for step in result.steps)

    # filter_large_items is the 1st pipeline step -- the oversized panel
    # "re" is dropped there; the line survives into later steps.
    dropped_at_step_0 = [
        g
        for result in clustering_results.values()
        for g in result.steps[0].categories["dropped_oversized"].groups
    ]
    assert any(g[0].kind == "re" for g in dropped_at_step_0)

    # The line ends up as a single-item cluster -- with the high-frequency
    # filter removed, nothing else in the chain rejects a lone,
    # otherwise-unremarkable cluster, so it survives to the final "kept"
    # category as a text candidate.
    all_final_kept = [
        p
        for result in clustering_results.values()
        for g in result.steps[-1].categories["kept"].groups
        for p in g
    ]
    assert all_final_kept and all(p.kind == "l" for p in all_final_kept)

    drawing_vectors: list[DrawingVector] = by_key["drawing_vectors"].data
    assert isinstance(drawing_vectors, list)
    # the dropped panel ends up folded into drawing_vectors; the line
    # survived clustering as a text candidate, so it's drawing_vectors only
    # if FAST found no text signal in it -- either way it's not asserted on
    # here.
    assert any(dv.paths[0].kind == "re" for dv in drawing_vectors)


def _make_path(seq: int) -> VectorPath:
    return VectorPath(
        seq=seq, item_index=0, kind="l", fill_rule="s", points=[(0, 0), (1, 1)],
        bbox=(0, 0, 1, 1), stroke_color=(0, 0, 0), fill_color=None,
        stroke_opacity=None, fill_opacity=None, stroke_width=1.0, dashes=None,
        closed=False, layer=None, page_index=0,
    )


def test_run_drawing_vectors_restores_original_seq_order_across_groups():
    # Two (layer, color) buckets, keyed so dict iteration visits "a" fully
    # before "b" -- without the (seq, item_index) sort, this would produce
    # drawing_vectors in [0, 2, 1, 3] order (all of "a" then all of "b")
    # instead of the original PDF stacking order [0, 1, 2, 3].
    ctx = PipelineContext(reader=None, page_index=0)
    ctx.clustering = {
        ("", ()): ClusteringStageResult(steps=[
            StepResult("x", {
                "kept": CategoryResult([], "kept"),
                "dropped_x": CategoryResult([[_make_path(0)], [_make_path(2)]], "dropped"),
            }),
        ]),
        ("", (1,)): ClusteringStageResult(steps=[
            StepResult("x", {
                "kept": CategoryResult([], "kept"),
                "dropped_x": CategoryResult([[_make_path(1)], [_make_path(3)]], "dropped"),
            }),
        ]),
    }

    drawing_vectors = _run_drawing_vectors(ctx)

    assert [dv.paths[0].seq for dv in drawing_vectors] == [0, 1, 2, 3]


def _make_path_at(seq: int, bbox: tuple[float, float, float, float]) -> VectorPath:
    return VectorPath(
        seq=seq, item_index=0, kind="l", fill_rule="s",
        points=[(bbox[0], bbox[1]), (bbox[2], bbox[3])], bbox=bbox,
        stroke_color=(0, 0, 0), fill_color=None, stroke_opacity=None,
        fill_opacity=None, stroke_width=1.0, dashes=None, closed=False,
        layer=None, page_index=0,
    )


def test_sample_mask_weights_by_own_vector_footprint_not_aggregate_bbox():
    # A 100x100 mask, hot (1.0) only in the top-left 10x10 corner.
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[0:10, 0:10] = 1.0

    # Two paths whose own bboxes are each fully inside one region (one
    # fully hot, one fully cold) -- but whose aggregate bbox spans the
    # whole 100x100 mask, mostly cold. The old aggregate-bbox-mean version
    # would score this near 0 (~2% hot pixels of the full span); the
    # per-path-weighted version should score exactly the average of each
    # path's own footprint: (1.0 + 0.0) / 2 == 0.5.
    hot_path = _make_path_at(0, (0, 0, 10, 10))
    cold_path = _make_path_at(1, (90, 90, 100, 100))

    score = _sample_mask(mask, [hot_path, cold_path], zoom=1.0)

    assert score == pytest.approx(0.5)


def test_sample_mask_returns_zero_for_none_mask():
    assert _sample_mask(None, [_make_path_at(0, (0, 0, 10, 10))], zoom=1.0) == 0.0


def test_run_spatial_regroup_merges_clusters_whose_paths_overlap_across_buckets():
    # cluster_b's path touches cluster_a's path within 1px tolerance
    # (gap of 0.5pt) -- should merge into one, regardless of which
    # (layer, color) bucket each came from -- spatial_regroup deliberately
    # ignores that.
    cluster_a = [_make_path_at(0, (0, 0, 10, 10))]
    cluster_b = [_make_path_at(1, (10.5, 0, 20, 10))]
    ctx = PipelineContext(reader=None, page_index=0)
    ctx.fast_passed = [cluster_a, cluster_b]

    regrouped = _run_spatial_regroup(ctx)

    assert len(regrouped) == 1
    assert {p.seq for p in regrouped[0]} == {0, 1}
    # word_split reads regrouped_clusters, not fast_passed, after this stage.
    assert ctx.regrouped_clusters == regrouped


def test_run_spatial_regroup_keeps_clusters_separate_when_no_path_actually_overlaps():
    # cluster_a and cluster_b's own paths are 3pt apart -- over the 1px
    # tolerance -- even though this would have merged under the old
    # aggregate-bbox-gap threshold (5pt). Only real per-path overlap
    # should merge clusters now.
    cluster_a = [_make_path_at(0, (0, 0, 10, 10))]
    cluster_b = [_make_path_at(1, (13, 0, 20, 10))]
    ctx = PipelineContext(reader=None, page_index=0)
    ctx.fast_passed = [cluster_a, cluster_b]

    regrouped = _run_spatial_regroup(ctx)

    assert len(regrouped) == 2


def test_run_spatial_regroup_keeps_far_clusters_separate():
    cluster_a = [_make_path_at(0, (0, 0, 10, 10))]
    cluster_b = [_make_path_at(1, (100, 0, 110, 10))]
    ctx = PipelineContext(reader=None, page_index=0)
    ctx.fast_passed = [cluster_a, cluster_b]

    regrouped = _run_spatial_regroup(ctx)

    assert len(regrouped) == 2


def test_run_page_stage_error_is_caught(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory([{}])
    path = tmp_pdf_path(doc)

    def _boom(ctx: PipelineContext):
        raise RuntimeError("stage exploded")

    class _FailingPipeline(Pipeline):
        STAGES = [StageSpec(key="boom", label="Boom", run=_boom)]

    with Reader(path) as reader:
        outputs = _FailingPipeline().run_page(reader, 0)

    assert len(outputs) == 1
    assert outputs[0].status == "error"
    assert outputs[0].key == "boom"
    assert "stage exploded" in outputs[0].error


def test_run_page_stage_after_error_still_runs(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory([{}])
    path = tmp_pdf_path(doc)

    def _boom(ctx: PipelineContext):
        raise RuntimeError("stage exploded")

    def _ok(ctx: PipelineContext):
        return "fine"

    class _MixedPipeline(Pipeline):
        STAGES = [
            StageSpec(key="boom", label="Boom", run=_boom),
            StageSpec(key="ok", label="Ok", run=_ok),
        ]

    with Reader(path) as reader:
        outputs = _MixedPipeline().run_page(reader, 0)

    assert [o.status for o in outputs] == ["error", "ok"]
    assert outputs[1].data == "fine"


def _make_cluster_result(text: str, bbox, rotation_used: int = 0) -> ClusterOcrResult:
    resolved = TextVectorResult(
        paths=[], text=text, confidence=0.95, bbox=bbox, ocr_bbox=None,
        rotation_used=rotation_used, page_index=0,
    )
    return ClusterOcrResult(cluster=[], resolved=resolved, ocr_seconds=0.0)


def test_rotation_verify_flips_when_bbox_rotated_fits_better():
    text = "HELLO WORLD"
    ratio = _text_aspect_ratio(text)
    assert ratio > 1  # wide text

    # A tall/narrow bbox shaped like the text rotated 90 (ratio swapped)
    # should trigger the fix.
    bbox = (0, 0, 10, 10 * ratio)
    ctx = PipelineContext(reader=None, page_index=0)
    ctx.cluster_ocr_results = [_make_cluster_result(text, bbox, rotation_used=0)]

    checks = _run_rotation_verify(ctx)

    assert len(checks) == 1
    assert checks[0].applied
    assert checks[0].after_rotation == 90
    assert checks[0].resolved.rotation_used == 90
    # ocr_compare's own resolved reading is never mutated -- rotation_verify
    # reports the corrected orientation as a separate RotationCheck.resolved
    # object instead.
    assert ctx.cluster_ocr_results[0].resolved.rotation_used == 0


def test_rotation_verify_leaves_matching_bbox_alone():
    text = "HELLO WORLD"
    ratio = _text_aspect_ratio(text)
    bbox = (0, 0, 10 * ratio, 10)  # already matches the text's own aspect
    ctx = PipelineContext(reader=None, page_index=0)
    ctx.cluster_ocr_results = [_make_cluster_result(text, bbox, rotation_used=0)]

    checks = _run_rotation_verify(ctx)

    assert len(checks) == 1
    assert not checks[0].applied
    assert checks[0].after_rotation == 0
    # Not applied -- resolved is the exact same (unmutated) object.
    assert checks[0].resolved is ctx.cluster_ocr_results[0].resolved
    assert ctx.cluster_ocr_results[0].resolved.rotation_used == 0


def test_rotation_verify_skips_blank_text():
    ctx = PipelineContext(reader=None, page_index=0)
    ctx.cluster_ocr_results = [_make_cluster_result("", (0, 0, 10, 10))]

    checks = _run_rotation_verify(ctx)

    assert checks == []
