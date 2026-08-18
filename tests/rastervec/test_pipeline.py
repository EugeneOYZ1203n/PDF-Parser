from __future__ import annotations

from dataclasses import dataclass

import pymupdf as fitz

from rastervec.models import DrawingVector, Page, TextWord, VectorPath
from rastervec.pipeline import Pipeline, PipelineContext, StageSpec, VectorStageBuckets
from rastervec.reader import Reader

_EXPECTED_STAGE_KEYS = [
    "reader",
    "native",
    "vector_extract",
    "layer_separation",
    "color_separation",
    "filter_layout_panels",
    "filter_background_fill",
    "cluster_spatial",
    "cluster_by_dimension",
    "cluster_by_seq",
    "drawing_vectors",
]

_VECTOR_CLASSIFICATION_STAGE_KEYS = [
    "filter_layout_panels",
    "filter_background_fill",
    "cluster_spatial",
    "cluster_by_dimension",
    "cluster_by_seq",
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
    for key in _VECTOR_CLASSIFICATION_STAGE_KEYS:
        assert isinstance(by_key[key].data, dict)
    assert isinstance(by_key["drawing_vectors"].data, list)


def test_run_page_vector_stages_on_drawing_pdf(tmp_pdf_path):
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)

    # single-item "re" -> dropped by filter_layout_panels (a lone panel/border rect).
    panel = page.new_shape()
    panel.draw_rect(fitz.Rect(0, 0, 200, 100))
    panel.finish(color=(0, 0, 0))
    panel.commit()

    # a real line -> survives filtering, ends up in a spatial cluster.
    line = page.new_shape()
    line.draw_line((10, 10), (60, 60))
    line.finish(color=(0, 0, 0), width=2)
    line.commit()

    path = tmp_pdf_path(doc)

    with Reader(path) as reader:
        outputs = Pipeline().run_page(reader, 0)

    by_key = {o.key: o for o in outputs}
    assert all(o.status == "ok" for o in outputs)

    paths: list[VectorPath] = by_key["vector_extract"].data
    assert len(paths) > 0

    layout_buckets: dict = by_key["filter_layout_panels"].data
    dropped_panels = [
        group
        for buckets in layout_buckets.values()
        for group in buckets.this_stage
        if group[0].kind == "re"
    ]
    assert len(dropped_panels) == 1

    spatial_buckets: dict = by_key["cluster_spatial"].data
    all_spatial_clusters = [
        cluster for buckets in spatial_buckets.values() for cluster in buckets.this_stage
    ]
    assert any(any(p.kind == "l" for p in cluster) for cluster in all_spatial_clusters)
    # the dropped panel carries forward into "previous" for every later stage.
    assert all(
        any(g[0].kind == "re" for g in buckets.previous)
        for buckets in spatial_buckets.values()
        if buckets.previous
    )

    seq_buckets: dict = by_key["cluster_by_seq"].data
    assert isinstance(seq_buckets, dict)
    for buckets in seq_buckets.values():
        assert isinstance(buckets, VectorStageBuckets)

    drawing_vectors: list[DrawingVector] = by_key["drawing_vectors"].data
    assert isinstance(drawing_vectors, list)


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
