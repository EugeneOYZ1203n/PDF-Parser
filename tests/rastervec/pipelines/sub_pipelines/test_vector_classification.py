from __future__ import annotations

import pymupdf as fitz

from rastervec.pipelines._steps import extract_vectors
from rastervec.pipelines.sub_pipelines.vector_classification import (
    ClassificationResult,
    _classify_bucket,
    classify_vectors,
)
from rastervec.Reader.reader import Reader


def _drawing_page(tmp_pdf_path):
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    panel = page.new_shape()
    panel.draw_rect(fitz.Rect(0, 0, 200, 100))  # oversized -> dropped step 1
    panel.finish(color=(0, 0, 0))
    panel.commit()
    line = page.new_shape()
    line.draw_line((10, 10), (16, 16))
    line.finish(color=(0, 0, 0), width=2)
    line.commit()
    return tmp_pdf_path(doc)


def test_classify_vectors_splits_candidates_and_drops(tmp_pdf_path):
    with Reader(_drawing_page(tmp_pdf_path)) as reader:
        page = reader.get_page(0)
        vectors = extract_vectors(page)
        res = classify_vectors(vectors.paths, page, verbose=True)

    assert isinstance(res, ClassificationResult)
    # oversized panel dropped as drawing content
    assert any(p.kind == "re" for p in res.dropped)
    # small line survives as a text candidate
    kept = [p for c in res.text_clusters for p in c]
    assert kept and all(p.kind == "l" for p in kept)
    # lineage keyed by live identity
    assert all(id(c) in res.cluster_groups for c in res.text_clusters)
    assert res.paths_by_layer is not None  # verbose


def test_classify_bucket_returns_twelve_step_results(tmp_pdf_path):
    with Reader(_drawing_page(tmp_pdf_path)) as reader:
        page = reader.get_page(0)
        vectors = extract_vectors(page)
        steps = _classify_bucket(vectors.paths, page)
    assert len(steps) == 12
    assert all("kept" in s.categories for s in steps)
    assert steps[-1].cluster_groups is not None
