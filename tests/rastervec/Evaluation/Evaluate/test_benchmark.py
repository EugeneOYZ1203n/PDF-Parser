from __future__ import annotations

import math

import pytest

from rastervec.Evaluation.Evaluate.benchmark import (
    aggregate_results,
    distribution_stats,
    format_aggregate,
    format_report,
    format_timing_report,
    summarize_stage_timings,
)
from rastervec.Evaluation.Evaluate.metrics import MetricCounts, MetricSuiteResult, Ratio


def _suite(**ratios) -> MetricSuiteResult:
    base = {name: Ratio(0.0, math.nan) for name in (
        "page_char_multiset_recall", "page_char_multiset_precision",
        "page_word_multiset_recall", "page_word_multiset_precision",
        "pred_text_fully_contained_in_overlapping_gt_rate",
        "gt_text_word_coverage_by_overlapping_preds",
        "per_gt_best_single_pred_iou_mean", "per_gt_union_pred_iou_mean",
        "undetected_gt_area_ratio", "rotation_accuracy_localized_gt",
        "classification_recall_gt_reached_ocr",
        "classification_precision_candidate_is_text",
        "gt_miss_attributed_to_classification_frac", "gt_miss_attributed_to_fast_frac",
        "gt_miss_attributed_to_ocr_blank_frac", "gt_miss_attributed_to_not_found_frac",
    )}
    base.update(ratios)
    return MetricSuiteResult(ratios=base)


def test_format_report_groups_metrics_with_absolute_counts():
    result = _suite(
        page_char_multiset_recall=Ratio(9.0, 10.0),
        page_char_multiset_precision=Ratio(9.0, 12.0),
    )
    result.per_stage_miss_counts = {"ocr_blank": 1}
    result.counts = MetricCounts(n_gt=4, n_pred=3, n_pred_nonblank=3, n_text_candidates=3)

    report = format_report("x.pdf", 0, result)

    assert "x.pdf page 0:" in report
    assert "[character]" in report
    assert "page_char_multiset_recall: 9/10  (0.900)" in report
    assert "page_char_multiset_f1: 0.818" in report  # 2*.9*.75/(.9+.75)
    assert "rotation_accuracy_localized_gt: 0/nan  (n/a)" in report
    assert "per_stage_miss_counts: {'ocr_blank': 1}" in report


def test_aggregate_results_empty_returns_none():
    assert aggregate_results([]) is None
    assert "no results" in format_aggregate(None, 0)


def test_aggregate_results_micro_averages():
    a = _suite(page_char_multiset_recall=Ratio(1.0, 3.0))
    b = _suite(page_char_multiset_recall=Ratio(10.0, 15.0))

    agg = aggregate_results([a, b])

    assert agg.ratios["page_char_multiset_recall"] == Ratio(11.0, 18.0)
    assert agg.get("page_char_multiset_recall") == pytest.approx(11 / 18)


def test_aggregate_results_merges_miss_counts_and_sums_counts():
    a = _suite()
    a.per_stage_miss_counts = {"ocr_blank": 1, "fast_text_detect": 1}
    a.counts = MetricCounts(n_gt=2)
    b = _suite()
    b.per_stage_miss_counts = {"ocr_blank": 1}
    b.counts = MetricCounts(n_gt=3)

    agg = aggregate_results([a, b])

    assert agg.per_stage_miss_counts == {"ocr_blank": 2, "fast_text_detect": 1}
    assert agg.counts.n_gt == 5


def test_distribution_stats_empty_returns_empty_dict():
    assert distribution_stats([]) == {}


def test_distribution_stats_basic():
    stats = distribution_stats([1.0, 2.0, 3.0, 4.0, 5.0])

    assert stats["n"] == 5
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
    assert stats["median"] == 3.0
    assert stats["mean"] == 3.0
    assert stats["q1"] == 2.0
    assert stats["q3"] == 4.0


def test_summarize_stage_timings_per_stage_and_total():
    per_page = [
        {"reader": 0.1, "native": 0.2},
        {"reader": 0.3, "native": 0.4},
    ]

    summary = summarize_stage_timings(per_page, ["reader", "native", "vector_extract"])

    assert list(summary) == ["reader", "native", "total"]  # vector_extract never ran
    assert summary["reader"]["mean"] == pytest.approx(0.2)
    assert summary["total"]["min"] == pytest.approx(0.3)
    assert summary["total"]["max"] == pytest.approx(0.7)


def test_summarize_stage_timings_empty():
    assert summarize_stage_timings([], ["reader"]) == {}


def test_format_timing_report_has_rows_and_handles_empty():
    assert "no timing data" in format_timing_report({})

    report = format_timing_report(summarize_stage_timings([{"reader": 0.5}], ["reader"]))
    assert "reader" in report
    assert "total" in report
