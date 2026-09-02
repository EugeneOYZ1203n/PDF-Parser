from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from rastervec.Evaluation.Evaluate.metrics import (
    GtRegion,
    MetricConfig,
    Prediction,
    Ratio,
    aggregate_suite,
    build_overlap_graph,
    evaluate_metrics,
)
from rastervec.pipeline import ClusteringStageResult
from rastervec.Vector_Classification.classification import CategoryResult, StepResult


def _pred(text, bbox, rotation=0, blank=False):
    return Prediction(
        text=text, bbox=bbox, rotation=rotation, reached_ocr=True,
        ocr_blank=blank, source_cluster_id=id(bbox),
    )


def _gt(text, bbox, rotation=0):
    return GtRegion(page_index=0, bbox=bbox, text=text, expected_rotation=rotation)


def _box_group(bbox):
    return [SimpleNamespace(bbox=bbox)]


# --------------------------------------------------------------------------
# overlap graph
# --------------------------------------------------------------------------
def test_graph_disjoint_no_edges():
    g = build_overlap_graph([_gt("a", (0, 0, 10, 10))], [_pred("A", (50, 50, 60, 60))])
    assert g.edges == []
    assert g.missed_gt_idxs == [0]
    assert g.gt_has_overlap == [False]


def test_graph_n_to_1_assignment():
    gt = [_gt("foo bar baz", (0, 0, 30, 10))]
    preds = [
        _pred("FOO", (0, 0, 10, 10)),
        _pred("BAR", (10, 0, 20, 10)),
        _pred("BAZ", (20, 0, 30, 10)),
    ]
    g = build_overlap_graph(gt, preds)
    assert sorted(g.assigned_preds_by_gt[0]) == [0, 1, 2]
    assert g.localized_gt_idxs == [0]


def test_graph_blank_predictions_excluded():
    g = build_overlap_graph(
        [_gt("a", (0, 0, 10, 10))],
        [_pred("", (0, 0, 10, 10), blank=True)],
    )
    assert g.preds == []
    assert g.missed_gt_idxs == [0]


# --------------------------------------------------------------------------
# N:1 scene -- the case the legacy 1:1 matcher corrupts
# --------------------------------------------------------------------------
def test_n_to_1_scene_metrics():
    gt = [_gt("foo bar baz", (0, 0, 30, 10))]
    preds = [
        _pred("FOO", (0, 0, 10, 10)),
        _pred("BAR", (10, 0, 20, 10)),
        _pred("BAZ", (20, 0, 30, 10)),
    ]
    res = evaluate_metrics(gt, preds, text_candidate_boxes=[(0, 0, 30, 10)])

    assert res.get("page_char_multiset_recall") == pytest.approx(1.0)
    assert res.get("page_word_multiset_recall") == pytest.approx(1.0)
    assert res.get("page_word_multiset_f1") == pytest.approx(1.0)
    assert res.ratios["gt_text_word_coverage_by_overlapping_preds"] == Ratio(3.0, 3.0)
    assert res.ratios["pred_text_fully_contained_in_overlapping_gt_rate"] == Ratio(3.0, 3.0)
    assert res.get("per_gt_best_single_pred_iou_mean") == pytest.approx(1 / 3)
    assert res.get("per_gt_union_pred_iou_mean") == pytest.approx(1.0)
    assert res.ratios["undetected_gt_area_ratio"] == Ratio(0.0, 300.0)
    assert res.ratios["rotation_accuracy_localized_gt"] == Ratio(1.0, 1.0)
    assert res.ratios["classification_recall_gt_reached_ocr"] == Ratio(1.0, 1.0)
    # no misses + clustering not supplied -> attribution n/a
    assert math.isnan(res.get("gt_miss_attributed_to_classification_frac"))
    assert res.per_stage_miss_counts == {}
    assert res.counts.n_gt == 1
    assert res.counts.n_pred_nonblank == 3
    assert res.counts.n_gt_localized == 1


def test_perfect_1_to_1():
    gt = [_gt("Hello", (0, 0, 10, 5)), _gt("World", (20, 20, 30, 25), rotation=90)]
    preds = [_pred("hello", (0, 0, 10, 5)), _pred("world", (20, 20, 30, 25), rotation=90)]
    res = evaluate_metrics(gt, preds, text_candidate_boxes=[(0, 0, 10, 5), (20, 20, 30, 25)])
    assert res.get("page_char_multiset_recall") == pytest.approx(1.0)
    assert res.get("page_char_multiset_precision") == pytest.approx(1.0)
    assert res.get("rotation_accuracy_localized_gt") == pytest.approx(1.0)
    assert res.get("per_gt_union_pred_iou_mean") == pytest.approx(1.0)
    assert res.get("undetected_gt_area_ratio") == pytest.approx(0.0)
    assert res.get("classification_recall_gt_reached_ocr") == pytest.approx(1.0)
    assert res.get("classification_precision_candidate_is_text") == pytest.approx(1.0)


# --------------------------------------------------------------------------
# text-precision cases
# --------------------------------------------------------------------------
def test_hallucinated_pred_not_contained():
    gt = [_gt("foo", (0, 0, 10, 10))]
    preds = [_pred("FOO ZZZ", (0, 0, 10, 10))]
    res = evaluate_metrics(gt, preds, text_candidate_boxes=[(0, 0, 10, 10)])
    assert res.ratios["pred_text_fully_contained_in_overlapping_gt_rate"] == Ratio(0.0, 1.0)
    assert res.get("page_char_multiset_precision") < 1.0
    # gt side: all of "foo" is present in the overlapping pred
    assert res.ratios["gt_text_word_coverage_by_overlapping_preds"] == Ratio(1.0, 1.0)


def test_case_and_whitespace_normalised():
    gt = [_gt("Setback  Line", (0, 0, 20, 5))]
    preds = [_pred("SETBACK LINE", (0, 0, 20, 5))]
    res = evaluate_metrics(gt, preds, text_candidate_boxes=[(0, 0, 20, 5)])
    assert res.get("page_char_multiset_recall") == pytest.approx(1.0)
    assert res.get("page_word_multiset_recall") == pytest.approx(1.0)


# --------------------------------------------------------------------------
# misses
# --------------------------------------------------------------------------
def test_missed_gt_no_prediction():
    gt = [_gt("gone", (0, 0, 10, 10))]
    res = evaluate_metrics(gt, [], text_candidate_boxes=[])
    assert res.ratios["undetected_gt_area_ratio"] == Ratio(100.0, 100.0)
    assert res.ratios["classification_recall_gt_reached_ocr"] == Ratio(0.0, 1.0)
    assert res.counts.n_gt_missed == 1


def test_blank_ocr_reached_candidate_but_missed():
    gt = [_gt("text", (0, 0, 10, 10))]
    preds = [_pred("", (0, 0, 10, 10), blank=True)]
    res = evaluate_metrics(gt, preds, text_candidate_boxes=[(0, 0, 10, 10)])
    assert res.ratios["classification_recall_gt_reached_ocr"] == Ratio(1.0, 1.0)
    assert res.counts.n_gt_missed == 1
    assert res.get("page_char_multiset_recall") == pytest.approx(0.0)  # nothing recalled
    assert math.isnan(res.get("page_char_multiset_precision"))  # no predicted chars at all


def test_miss_attribution_to_classification():
    gt = [_gt("dropped text", (0, 0, 10, 10))]
    dropped = CategoryResult(groups=[_box_group((0, 0, 10, 10))], role="dropped")
    kept = CategoryResult(groups=[], role="kept")
    step = StepResult(label="Tiny groups", categories={"kept": kept, "dropped": dropped})
    clustering = {("", ()): ClusteringStageResult(steps=[step])}

    res = evaluate_metrics(
        gt, [], text_candidate_boxes=[], clustering=clustering,
    )
    assert res.ratios["gt_miss_attributed_to_classification_frac"] == Ratio(1.0, 1.0)
    assert res.ratios["gt_miss_attributed_to_not_found_frac"] == Ratio(0.0, 1.0)
    assert res.per_stage_miss_counts == {"classification:Tiny groups": 1}


def test_miss_attribution_fast_and_ocr_and_not_found():
    gt = [
        _gt("a", (0, 0, 10, 10)),
        _gt("b", (100, 0, 110, 10)),
        _gt("c", (200, 0, 210, 10)),
    ]
    fast_dropped = [_box_group((0, 0, 10, 10))]
    ocr_failed = [_box_group((100, 0, 110, 10))]
    clustering = {("", ()): ClusteringStageResult(steps=[])}

    res = evaluate_metrics(
        gt, [], text_candidate_boxes=[], clustering=clustering,
        fast_dropped=fast_dropped, ocr_failed=ocr_failed,
    )
    assert res.ratios["gt_miss_attributed_to_fast_frac"] == Ratio(1.0, 3.0)
    assert res.ratios["gt_miss_attributed_to_ocr_blank_frac"] == Ratio(1.0, 3.0)
    assert res.ratios["gt_miss_attributed_to_not_found_frac"] == Ratio(1.0, 3.0)


def test_miss_attribution_na_without_clustering():
    gt = [_gt("x", (0, 0, 10, 10))]
    res = evaluate_metrics(gt, [], text_candidate_boxes=[])
    for name in (
        "gt_miss_attributed_to_classification_frac",
        "gt_miss_attributed_to_fast_frac",
        "gt_miss_attributed_to_ocr_blank_frac",
        "gt_miss_attributed_to_not_found_frac",
    ):
        assert math.isnan(res.get(name))


# --------------------------------------------------------------------------
# aggregation -- micro-average, not mean of ratios
# --------------------------------------------------------------------------
def test_aggregate_is_micro_averaged():
    r1 = evaluate_metrics([_gt("abc", (0, 0, 30, 10))], [_pred("A", (0, 0, 10, 10))],
                          text_candidate_boxes=[(0, 0, 10, 10)])
    # page 1: char recall numerator 1 (just "A"), denominator 3
    assert r1.ratios["page_char_multiset_recall"] == Ratio(1.0, 3.0)

    r2 = evaluate_metrics(
        [_gt("abcdefghij", (0, 0, 30, 10)), _gt("klmno", (0, 20, 30, 30))],
        [_pred("ABCDEFGHIJKLMNO", (0, 0, 30, 30))],
        text_candidate_boxes=[(0, 0, 30, 30)],
    )
    # page 2: all 15 gt chars present -> Ratio(15, 15); but wait pred covers
    # both -> assigned to both. char recall numerator 15, denominator 15.
    assert r2.ratios["page_char_multiset_recall"] == Ratio(15.0, 15.0)

    agg = aggregate_suite([r1, r2])
    assert agg.ratios["page_char_multiset_recall"] == Ratio(16.0, 18.0)
    assert agg.get("page_char_multiset_recall") == pytest.approx(16 / 18)
    # NOT the mean of per-page ratios:
    assert agg.get("page_char_multiset_recall") != pytest.approx((1 / 3 + 1.0) / 2)


def test_aggregate_skips_na_pages_and_sums_counts():
    r_na = evaluate_metrics([], [], text_candidate_boxes=[])  # empty gt -> na everywhere
    r_ok = evaluate_metrics([_gt("hi", (0, 0, 10, 10))], [_pred("HI", (0, 0, 10, 10))],
                            text_candidate_boxes=[(0, 0, 10, 10)])
    agg = aggregate_suite([r_na, r_ok])
    assert agg.ratios["page_char_multiset_recall"] == Ratio(2.0, 2.0)
    assert agg.counts.n_gt == 1


def test_aggregate_f1_from_aggregated_pr():
    r1 = evaluate_metrics([_gt("ab", (0, 0, 10, 10))], [_pred("A", (0, 0, 10, 10))],
                          text_candidate_boxes=[(0, 0, 10, 10)])
    r2 = evaluate_metrics([_gt("cd", (0, 0, 10, 10))], [_pred("CD", (0, 0, 10, 10))],
                          text_candidate_boxes=[(0, 0, 10, 10)])
    agg = aggregate_suite([r1, r2])
    rec = agg.ratios["page_char_multiset_recall"].value
    prec = agg.ratios["page_char_multiset_precision"].value
    assert agg.get("page_char_multiset_f1") == pytest.approx(2 * rec * prec / (rec + prec))
