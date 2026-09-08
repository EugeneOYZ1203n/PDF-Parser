from __future__ import annotations

import numpy as np
import pymupdf as fitz
import pytest

from rastervec.Evaluation.Evaluate import golden_regression
from rastervec.Evaluation.Evaluate.golden_schema import GoldenCase, GoldenCaseBank
from rastervec.models import TextVectorResult
from rastervec.OCR.radon import ClusterSegmentation
from rastervec.pipelines.current import run_pipeline
from rastervec.pipelines.result import ClusteringStageResult, PipelineResult
from rastervec.pipelines.sub_pipelines import ocr as ocr_mod
from rastervec.renderer import page_points_to_pixel
from rastervec.Vector_Classification.classification import CategoryResult, StepResult


def _pr(**overrides) -> PipelineResult:
    fields = dict(
        page=None, native_words=[], drawing_vectors=[], ocr_results=[],
        cluster_ocr_results=[], text_clusters=[], regrouped_clusters=[],
        clustering={}, cluster_groups={}, fast_dropped=[], ocr_failed=[],
        step_durations={}, engine="current",
    )
    fields.update(overrides)
    return PipelineResult(**fields)


def _case(**overrides) -> GoldenCase:
    fields = dict(
        stage="classification", label="positive", pdf_path="a.pdf", page_index=0,
        signature="sig", bbox=(0.0, 0.0, 10.0, 10.0), role="kept",
    )
    fields.update(overrides)
    return GoldenCase(**fields)


def _seg(word_corners, *, dpi=300) -> ClusterSegmentation:
    return ClusterSegmentation(
        skew_deg=0.0, line_spacing_px=0.0,
        word_crops=[np.zeros((1, 1), dtype=np.uint8) for _ in word_corners],
        word_corners=word_corners, render_dpi=dpi,
    )


def _pixel_corners(cluster, dpi, bbox):
    x0, y0, x1, y1 = bbox
    return page_points_to_pixel(cluster, dpi, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


# --------------------------------------------------------------------------
# compare_case: classification / fast
# --------------------------------------------------------------------------
def test_compare_case_passes_on_matching_role_and_bbox(vector_path):
    cluster = [vector_path(seq=0, bbox=(0.0, 0.0, 10.0, 10.0))]
    res = _pr(text_clusters=[cluster])
    result = golden_regression.compare_case(_case(bbox=(0.0, 0.0, 10.0, 10.0), role="kept"), res)
    assert result.passed
    assert result.best_iou == pytest.approx(1.0)


def test_compare_case_fails_when_role_changed(vector_path):
    dropped_group = [vector_path(seq=0, bbox=(0.0, 0.0, 10.0, 10.0))]
    bucket = ClusteringStageResult(steps=[StepResult(
        label="Step",
        categories={"dropped": CategoryResult(groups=[dropped_group], role="dropped")},
    )])
    res = _pr(clustering={("", None): bucket})
    result = golden_regression.compare_case(_case(bbox=(0.0, 0.0, 10.0, 10.0), role="kept"), res)
    assert not result.passed
    assert "role changed" in result.detail


def test_compare_case_fails_when_no_cluster_within_tolerance(vector_path):
    cluster = [vector_path(seq=0, bbox=(100.0, 100.0, 110.0, 110.0))]
    res = _pr(text_clusters=[cluster])
    result = golden_regression.compare_case(_case(bbox=(0.0, 0.0, 10.0, 10.0), role="kept"), res)
    assert not result.passed
    assert "no cluster" in result.detail


# --------------------------------------------------------------------------
# compare_case: word_split
# --------------------------------------------------------------------------
def test_compare_case_word_split_passes_on_full_match(vector_path):
    cluster = [vector_path(seq=0, bbox=(0.0, 0.0, 20.0, 10.0))]
    corners = _pixel_corners(cluster, 300, (0.0, 0.0, 5.0, 10.0))
    res = _pr(regrouped_clusters=[cluster], segmentations=[_seg([corners])])
    case = _case(
        stage="word_split", bbox=(0.0, 0.0, 20.0, 10.0), role=None,
        word_bboxes=[(0.0, 0.0, 5.0, 10.0)],
    )
    result = golden_regression.compare_case(case, res)
    assert result.passed


def test_compare_case_word_split_fails_on_word_count_mismatch(vector_path):
    cluster = [vector_path(seq=0, bbox=(0.0, 0.0, 20.0, 10.0))]
    corners = _pixel_corners(cluster, 300, (0.0, 0.0, 5.0, 10.0))
    res = _pr(regrouped_clusters=[cluster], segmentations=[_seg([corners])])
    case = _case(
        stage="word_split", bbox=(0.0, 0.0, 20.0, 10.0), role=None,
        word_bboxes=[(0.0, 0.0, 5.0, 10.0), (10.0, 0.0, 15.0, 10.0)],
    )
    result = golden_regression.compare_case(case, res)
    assert not result.passed
    assert "word count changed" in result.detail


def test_compare_case_word_split_fails_on_drifted_word(vector_path):
    cluster = [vector_path(seq=0, bbox=(0.0, 0.0, 20.0, 10.0))]
    corners = _pixel_corners(cluster, 300, (0.0, 0.0, 5.0, 10.0))
    res = _pr(regrouped_clusters=[cluster], segmentations=[_seg([corners])])
    case = _case(
        stage="word_split", bbox=(0.0, 0.0, 20.0, 10.0), role=None,
        word_bboxes=[(100.0, 100.0, 105.0, 110.0)],
    )
    result = golden_regression.compare_case(case, res)
    assert not result.passed
    assert "drifted" in result.detail


# --------------------------------------------------------------------------
# candidate listing
# --------------------------------------------------------------------------
def test_list_classification_candidates_splits_kept_and_dropped(vector_path):
    kept = [vector_path(seq=0, bbox=(0.0, 0.0, 10.0, 10.0))]
    dropped_group = [vector_path(seq=1, bbox=(50.0, 50.0, 60.0, 60.0))]
    bucket = ClusteringStageResult(steps=[StepResult(
        label="Step",
        categories={"dropped": CategoryResult(groups=[dropped_group], role="dropped")},
    )])
    res = _pr(text_clusters=[kept], clustering={("", None): bucket})

    positives, negatives = golden_regression.list_classification_candidates(res, n=3)
    assert len(positives) == 1 and positives[0].role == "kept"
    assert positives[0].image is not None
    assert len(negatives) == 1 and negatives[0].role == "dropped"
    assert negatives[0].note == "Step / dropped"


def test_list_fast_candidates_caps_at_n(vector_path):
    passed = [[vector_path(seq=i, bbox=(float(i), 0.0, float(i) + 1, 1.0))] for i in range(5)]
    dropped = [[vector_path(seq=10 + i, bbox=(float(i), 10.0, float(i) + 1, 11.0))] for i in range(2)]
    res = _pr(fast_passed=passed, fast_dropped=dropped)

    positives, negatives = golden_regression.list_fast_candidates(res, n=3)
    assert len(positives) == 3
    assert len(negatives) == 2


def test_list_word_split_candidates_skips_empty_segmentations(vector_path):
    cluster_a = [vector_path(seq=0, bbox=(0.0, 0.0, 10.0, 10.0))]
    cluster_b = [vector_path(seq=1, bbox=(20.0, 0.0, 30.0, 10.0))]
    empty_seg = ClusterSegmentation(skew_deg=0.0, line_spacing_px=0.0, word_crops=[], word_corners=[])
    good_corners = _pixel_corners(cluster_b, 300, (20.0, 0.0, 25.0, 10.0))
    good_seg = _seg([good_corners])
    res = _pr(regrouped_clusters=[cluster_a, cluster_b], segmentations=[empty_seg, good_seg])

    cands = golden_regression.list_word_split_candidates(res, n=6)
    assert len(cands) == 1
    assert cands[0].cluster == cluster_b
    assert cands[0].image is not None


def test_list_stage_candidates_combines_roles_and_is_seed_stable(vector_path):
    kept = [vector_path(seq=0, bbox=(0.0, 0.0, 10.0, 10.0))]
    dropped_group = [vector_path(seq=1, bbox=(50.0, 50.0, 60.0, 60.0))]
    bucket = ClusteringStageResult(steps=[StepResult(
        label="Step",
        categories={"dropped": CategoryResult(groups=[dropped_group], role="dropped")},
    )])
    res = _pr(text_clusters=[kept], clustering={("", None): bucket})

    cands = golden_regression.list_stage_candidates(res, "classification", seed=0)
    assert {c.role for c in cands} == {"kept", "dropped"}
    assert all(c.image is not None for c in cands)
    again = golden_regression.list_stage_candidates(res, "classification", seed=0)
    assert [c.role for c in cands] == [c.role for c in again]


def test_list_stage_candidates_caps_at_n(vector_path):
    passed = [[vector_path(seq=i, bbox=(float(i), 0.0, float(i) + 1, 1.0))] for i in range(5)]
    res = _pr(fast_passed=passed, fast_dropped=[])
    assert len(golden_regression.list_stage_candidates(res, "fast", n=2)) == 2


def test_format_case_bank_numbers_cases():
    bank = GoldenCaseBank(cases=[
        _case(stage="fast", label="positive", role="passed"),
        _case(stage="classification", label="negative", role="kept"),
    ])
    report = golden_regression.format_case_bank(bank)
    assert "[0] fast/positive" in report
    assert "[1] classification/negative" in report
    assert "2 case(s)" in report
    assert golden_regression.format_case_bank(GoldenCaseBank()) == "(no cases)"


# --------------------------------------------------------------------------
# run_regression dedup
# --------------------------------------------------------------------------
def test_run_regression_dedupes_by_page(monkeypatch, vector_path):
    calls = []
    cluster = [vector_path(seq=0, bbox=(0.0, 0.0, 10.0, 10.0))]

    def _stub(pdf_path, page_index, *, enable_fast=True, verbose=False):
        calls.append((pdf_path, page_index))
        return _pr(text_clusters=[cluster])

    monkeypatch.setattr(golden_regression, "run_pipeline", _stub)

    bank = GoldenCaseBank(cases=[
        _case(pdf_path="a.pdf", page_index=0, bbox=(0.0, 0.0, 10.0, 10.0)),
        _case(pdf_path="a.pdf", page_index=0, bbox=(0.0, 0.0, 10.0, 10.0), label="negative", role="dropped"),
        _case(pdf_path="b.pdf", page_index=1, bbox=(0.0, 0.0, 10.0, 10.0)),
    ])
    results = golden_regression.run_regression(bank)
    assert len(calls) == 2
    assert len(results) == 3


# --------------------------------------------------------------------------
# end-to-end capture -> replay round trip
# --------------------------------------------------------------------------
class _StubRenderOCR:
    def __init__(self, backend=None):
        self.backend = backend

    def recognize_segmented(self, seg, cluster, page):
        return TextVectorResult(
            paths=cluster, text="TXT", confidence=0.9, bbox=(0.0, 0.0, 1.0, 1.0),
            ocr_bbox=(0.0, 0.0, 1.0, 1.0), rotation_used=0,
            page_index=page.meta.index, words=None,
        )


@pytest.fixture(autouse=True)
def _stub_ocr(monkeypatch):
    monkeypatch.setattr(ocr_mod, "RenderOCR", _StubRenderOCR)


def test_capture_and_replay_word_split_round_trip(tmp_pdf_path):
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    line = page.new_shape()
    line.draw_line((10, 10), (16, 16))
    line.finish(color=(0, 0, 0), width=2)
    line.commit()
    pdf_path = tmp_pdf_path(doc)

    res = run_pipeline(pdf_path, 0, enable_fast=False, verbose=True)
    candidates = golden_regression.list_word_split_candidates(res)
    assert candidates

    case = golden_regression.capture_case(
        candidates[0], stage="word_split", label="positive",
        pdf_path=pdf_path, page_index=0,
    )

    res2 = run_pipeline(pdf_path, 0, enable_fast=False, verbose=True)
    result = golden_regression.compare_case(case, res2)
    assert result.passed
