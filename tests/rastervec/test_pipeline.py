from __future__ import annotations

import numpy as np
import pymupdf as fitz
import pytest

from rastervec.models import (
    DrawingVector,
    Page,
    TextVectorResult,
    TextWord,
    VectorPath,
)
from rastervec.pipeline import (
    ClusteringStageResult,
    Pipeline,
    PipelineContext,
    StageSpec,
    _run_drawing_vectors,
    _run_spatial_regroup,
    _sample_mask,
    run_page_context,
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
    # spatial_regroup, ocr_compare (which would construct RenderOCR), or
    # drawing_vectors
    assert [o.key for o in outputs] == _EXPECTED_STAGE_KEYS[:-3]
    assert all(o.status == "ok" for o in outputs)


def test_run_page_records_stage_durations(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory([{"texts": [{"point": (10, 20), "text": "Hello"}]}])
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        ctx = PipelineContext(reader=reader, page_index=0)
        outputs = Pipeline._run_stages(ctx, final_stage=None)

    assert list(ctx.stage_durations) == _EXPECTED_STAGE_KEYS
    assert all(isinstance(v, float) and v >= 0.0 for v in ctx.stage_durations.values())
    assert all(o.duration_seconds is not None and o.duration_seconds >= 0.0 for o in outputs)


def test_run_page_final_stage_only_times_stages_that_ran(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory([{"texts": [{"point": (10, 20), "text": "Hello"}]}])
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        ctx = run_page_context(reader, 0, final_stage="text_candidates")

    assert list(ctx.stage_durations) == _EXPECTED_STAGE_KEYS[: _EXPECTED_STAGE_KEYS.index("text_candidates") + 1]


def test_run_page_final_stage_unknown_raises(synthetic_pdf_factory, tmp_pdf_path):
    doc = synthetic_pdf_factory([{}])
    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        with pytest.raises(ValueError):
            Pipeline().run_page(reader, 0, final_stage="not_a_real_stage")


class _StubRenderOCR:
    """Stands in for rastervec.pipeline.RenderOCR -- no real engine, every
    cluster reads back as "TXT"."""

    def __init__(self, backend=None):
        self.backend = backend

    def ocr_cluster(self, cluster, page, dpi: int = 300):
        return TextVectorResult(
            paths=cluster, text="TXT", confidence=0.9,
            bbox=(0.0, 0.0, 1.0, 1.0), ocr_bbox=(0.0, 0.0, 1.0, 1.0),
            rotation_used=0, page_index=page.meta.index, words=None,
        )


def test_run_page_vector_stages_on_drawing_pdf(tmp_pdf_path, monkeypatch):
    monkeypatch.setattr("rastervec.pipeline.RenderOCR", _StubRenderOCR)
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
        # enable_fast=False -- this test exercises the clustering / drawing_vectors
        # stages, not FAST; the passthrough keeps it independent of the
        # fast_tiny_ic17mlt_640.pth weights file being present.
        outputs = Pipeline().run_page(reader, 0, enable_fast=False)

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


def test_run_page_context_enable_fast_false_is_passthrough_and_still_ocrs(
    tmp_pdf_path, monkeypatch,
):
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    line = page.new_shape()
    line.draw_line((10, 10), (16, 16))
    line.finish(color=(0, 0, 0), width=2)
    line.commit()
    path = tmp_pdf_path(doc)

    monkeypatch.setattr("rastervec.pipeline.RenderOCR", _StubRenderOCR)

    with Reader(path) as reader:
        ctx = run_page_context(reader, 0, enable_fast=False)

    # every stage still ran and got timed -- passthrough is a real stage,
    # not a skip
    assert list(ctx.stage_durations) == _EXPECTED_STAGE_KEYS
    # FAST kept every text candidate, dropped none, and never rendered
    assert ctx.fast_passed == ctx.text_clusters
    assert ctx.fast_dropped == []
    assert ctx.fast_result is not None and ctx.fast_result.page_image is None
    assert ctx.fast_result.detect_seconds is None
    # OCR still ran over the passed-through candidates
    assert ctx.text_clusters
    assert ctx.cluster_ocr_results
    assert all(r.resolved.text == "TXT" for r in ctx.cluster_ocr_results)


def test_run_page_context_threads_explicit_ocr_backend(tmp_pdf_path, monkeypatch):
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    line = page.new_shape()
    line.draw_line((10, 10), (16, 16))
    line.finish(color=(0, 0, 0), width=2)
    line.commit()
    path = tmp_pdf_path(doc)

    seen = {}

    class _Stub(_StubRenderOCR):
        def __init__(self, backend=None):
            super().__init__(backend)
            seen["backend"] = backend

    monkeypatch.setattr("rastervec.pipeline.RenderOCR", _Stub)
    sentinel = object()

    with Reader(path) as reader:
        run_page_context(reader, 0, enable_fast=False, ocr_backend=sentinel)

    assert seen["backend"] is sentinel


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


def _make_path_at(
    seq: int, bbox: tuple[float, float, float, float], *,
    stroke_color: tuple | None = (0, 0, 0), layer: str | None = None,
) -> VectorPath:
    return VectorPath(
        seq=seq, item_index=0, kind="l", fill_rule="s",
        points=[(bbox[0], bbox[1]), (bbox[2], bbox[3])], bbox=bbox,
        stroke_color=stroke_color, fill_color=None, stroke_opacity=None,
        fill_opacity=None, stroke_width=1.0, dashes=None, closed=False,
        layer=layer, page_index=0,
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


def test_run_spatial_regroup_merges_overlapping_same_layer_color_clusters():
    # Aggregate bboxes within the 1px tolerance (gap 0.5pt) and same
    # (layer, color) -- merge into one.
    cluster_a = [_make_path_at(0, (0, 0, 10, 10))]
    cluster_b = [_make_path_at(1, (10.5, 0, 20, 10))]
    ctx = PipelineContext(reader=None, page_index=0)
    ctx.fast_passed = [cluster_a, cluster_b]

    regrouped = _run_spatial_regroup(ctx)

    assert len(regrouped) == 1
    assert {p.seq for p in regrouped[0]} == {0, 1}
    # ocr_compare reads regrouped_clusters, not fast_passed, after this stage.
    assert ctx.regrouped_clusters == regrouped


def test_run_spatial_regroup_keeps_clusters_separate_when_bboxes_beyond_tolerance():
    # Aggregate bboxes 3pt apart -- over the 1px tolerance.
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


def test_run_spatial_regroup_keeps_overlapping_clusters_separate_when_different_color():
    # Overlapping aggregate bboxes but different stroke_color -> different
    # (layer, color) bucket -> not merged.
    cluster_a = [_make_path_at(0, (0, 0, 10, 10), stroke_color=(0, 0, 0))]
    cluster_b = [_make_path_at(1, (2, 0, 12, 10), stroke_color=(1, 0, 0))]
    ctx = PipelineContext(reader=None, page_index=0)
    ctx.fast_passed = [cluster_a, cluster_b]

    assert len(_run_spatial_regroup(ctx)) == 2


def test_run_spatial_regroup_keeps_overlapping_clusters_separate_when_different_layer():
    cluster_a = [_make_path_at(0, (0, 0, 10, 10), layer="A")]
    cluster_b = [_make_path_at(1, (2, 0, 12, 10), layer="B")]
    ctx = PipelineContext(reader=None, page_index=0)
    ctx.fast_passed = [cluster_a, cluster_b]

    assert len(_run_spatial_regroup(ctx)) == 2


def test_run_spatial_regroup_merges_overlapping_clusters_same_layer_and_color():
    cluster_a = [_make_path_at(0, (0, 0, 10, 10), stroke_color=(1, 0, 0), layer="A")]
    cluster_b = [_make_path_at(1, (2, 0, 12, 10), stroke_color=(1, 0, 0), layer="A")]
    ctx = PipelineContext(reader=None, page_index=0)
    ctx.fast_passed = [cluster_a, cluster_b]

    regrouped = _run_spatial_regroup(ctx)
    assert len(regrouped) == 1
    assert {p.seq for p in regrouped[0]} == {0, 1}


def test_cluster_lc_key_matches_on_paint_and_splits_on_opacity():
    from rastervec.pipeline import _cluster_lc_key

    a = [_make_path_at(0, (0, 0, 10, 10))]
    b = [_make_path_at(1, (5, 5, 15, 15))]
    assert _cluster_lc_key(a) == _cluster_lc_key(b)

    faint = _make_path_at(2, (0, 0, 10, 10))
    faint.stroke_opacity = 0.5
    assert _cluster_lc_key([faint]) != _cluster_lc_key(a)
    assert _cluster_lc_key([]) == ("", (None, None, None, None))


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


