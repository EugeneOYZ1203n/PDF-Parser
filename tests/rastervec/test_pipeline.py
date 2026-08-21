from __future__ import annotations

from dataclasses import dataclass

import pymupdf as fitz
import pytest

from rastervec.models import DrawingVector, Page, TextWord, VectorPath
from rastervec.pipeline import (
    ClusteringStageResult,
    Pipeline,
    PipelineContext,
    StageSpec,
    _run_drawing_vectors,
)
from rastervec.reader import Reader
from rastervec.vector import CategoryResult, StepResult

_EXPECTED_STAGE_KEYS = [
    "reader",
    "native",
    "vector_extract",
    "layer_separation",
    "color_separation",
    "clustering",
    "drawing_vectors",
    "ocr_text_clusters",
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
    assert isinstance(by_key["drawing_vectors"].data, list)
    assert by_key["ocr_text_clusters"].data == []  # no text-as-vector clusters on this page


def test_run_page_final_stage_stops_early_and_skips_ocr_engine(
    synthetic_pdf_factory, tmp_pdf_path, monkeypatch,
):
    doc = synthetic_pdf_factory([{"texts": [{"point": (10, 20), "text": "Hello"}]}])
    path = tmp_pdf_path(doc)

    def _boom(*args, **kwargs):
        raise AssertionError("RenderOCR must never be constructed once final_stage stops before it")

    monkeypatch.setattr("rastervec.pipeline.RenderOCR", _boom)

    with Reader(path) as reader:
        outputs = Pipeline().run_page(reader, 0, final_stage="drawing_vectors")

    assert [o.key for o in outputs] == _EXPECTED_STAGE_KEYS[:-1]  # everything but ocr_text_clusters
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
        assert len(result.steps) == 13
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

    # The line ends up as a single-item cluster -- dropped somewhere
    # later in the chain (exactly which step depends on current threshold
    # tuning); there's nothing left with enough shape variety on this
    # tiny synthetic page to survive to the final "kept" category.
    all_final_kept = [
        p
        for result in clustering_results.values()
        for g in result.steps[-1].categories["kept"].groups
        for p in g
    ]
    assert all_final_kept == []

    drawing_vectors: list[DrawingVector] = by_key["drawing_vectors"].data
    assert isinstance(drawing_vectors, list)
    # both the dropped panel and the low-variety line end up folded into
    # drawing_vectors.
    assert any(dv.paths[0].kind == "re" for dv in drawing_vectors)
    assert any(dv.paths[0].kind == "l" for dv in drawing_vectors)


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
