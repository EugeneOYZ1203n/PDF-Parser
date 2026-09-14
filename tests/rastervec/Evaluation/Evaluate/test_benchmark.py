from __future__ import annotations

import pytest

from rastervec.Evaluation.Evaluate.benchmark import (
    aggregate_results,
    distribution_stats,
    format_aggregate,
    format_aggregate_comparison,
    format_text_report,
    format_timing_report,
    format_variant_timing_comparison,
    summarize_stage_timings,
)
from rastervec.Evaluation.Evaluate.metrics import (
    TEXT_TYPES,
    GtRegion,
    Prediction,
    evaluate_text_metrics,
)


def _result(recall_pair=("HELLO", "HELLO")):
    gt_by_type = {t: [] for t in TEXT_TYPES}
    gt_by_type["native_to_vector"] = [GtRegion(0, (0, 0, 50, 10), recall_pair[0], 0, "native_to_vector")]
    entries_by_type = {t: [] for t in TEXT_TYPES}
    preds = [Prediction(recall_pair[1], (0, 0, 50, 10), 0)]
    return evaluate_text_metrics(gt_by_type, entries_by_type, preds)


def test_format_text_report_has_type_blocks():
    report = format_text_report("x.pdf", 0, _result())
    assert "x.pdf page 0 [text]:" in report
    assert "[native_to_vector]" in report
    assert "char overlap:" in report


def test_aggregate_results_empty_returns_none():
    assert aggregate_results([]) is None
    assert "no results" in format_aggregate(None, 0)


def test_aggregate_results_micro_averages():
    a = _result(("HI", "HI"))  # 2/2 matched
    b = _result(("HELLO", "HELLO"))  # 5/5 matched

    agg = aggregate_results([a, b])

    co = agg.by_type["native_to_vector"].char_overlap
    assert co.matched == 7
    assert co.total_gt == 7


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


def test_format_variant_timing_comparison_columns_and_delta():
    with_fast = {
        "reader": {"median": 0.1}, "fast_text_detect": {"median": 5.0},
        "ocr_compare": {"median": 2.0}, "total": {"median": 8.0},
    }
    without_fast = {
        "reader": {"median": 0.1}, "fast_text_detect": {"median": 0.0},
        "ocr_compare": {"median": 6.0}, "total": {"median": 7.0},
    }
    out = format_variant_timing_comparison(
        {"current_heavy": with_fast, "current_heavy_nofast": without_fast}
    )
    assert "current_heavy" in out and "current_heavy_nofast" in out
    assert "d:current_heavy_nofast" in out
    ocr_line = next(ln for ln in out.splitlines() if ln.strip().startswith("ocr_compare"))
    assert "+4.000" in ocr_line
    total_line = next(ln for ln in out.splitlines() if ln.strip().startswith("total"))
    assert "-1.000" in total_line

    assert "no timing data" in format_variant_timing_comparison({})


def test_format_variant_timing_comparison_tolerates_empty_variant_summary():
    out = format_variant_timing_comparison(
        {"current_light": {"reader": {"median": 0.2}, "total": {"median": 0.2}}, "legacy": {}}
    )
    assert "legacy" in out
    assert "nan" in out  # legacy has no per-stage medians


def test_format_aggregate_comparison_shows_all_variants():
    a = _result(("HI", "HI"))
    out = format_aggregate_comparison({"current": a, "legacy": None})
    assert "current" in out and "legacy" in out
    assert "(no results)" in out
