from __future__ import annotations

import pytest

from rastervec.Evaluation.Evaluate.benchmark import (
    aggregate_results,
    distribution_stats,
    format_report,
    format_timing_report,
    summarize_stage_timings,
)
from rastervec.Evaluation.Evaluate.evaluate import EvaluationResult, MissAttribution
from rastervec.Evaluation.Labelling.label_schema import LabelEntry


def _make_label(text="Hello") -> LabelEntry:
    return LabelEntry(
        page_index=0, cluster_bbox=(0, 0, 10, 5), cluster_signature="a",
        text=text, source="manual",
    )


def test_format_report_includes_metrics_and_counts():
    result = EvaluationResult(
        characters_found_pct=1.0, character_accuracy=0.9, character_error_rate=0.1,
        rotation_accuracy=1.0, bbox_accuracy=0.95, classification_precision=0.8,
        classification_recall=0.7, drawing_vector_count=3,
        miss_attributions=[MissAttribution(label=_make_label(), reason="ocr_blank")],
    )

    report = format_report("x.pdf", 0, result)

    assert "x.pdf page 0:" in report
    assert "character_accuracy: 0.900" in report
    assert "drawing_vector_count: 3" in report
    assert "miss reasons: {'ocr_blank': 1}" in report


def test_aggregate_results_empty_returns_empty_dict():
    assert aggregate_results([]) == {}


def test_aggregate_results_averages_numeric_fields():
    result_a = EvaluationResult(character_accuracy=1.0, drawing_vector_count=2)
    result_b = EvaluationResult(character_accuracy=0.5, drawing_vector_count=4)

    aggregate = aggregate_results([result_a, result_b])

    assert aggregate["character_accuracy"] == 0.75
    assert aggregate["drawing_vector_count_total"] == 6


def test_aggregate_results_sums_miss_reason_counts():
    result_a = EvaluationResult(
        miss_attributions=[
            MissAttribution(label=_make_label(), reason="ocr_blank"),
            MissAttribution(label=_make_label(), reason="fast_text_detect"),
        ]
    )
    result_b = EvaluationResult(
        miss_attributions=[MissAttribution(label=_make_label(), reason="ocr_blank")]
    )

    aggregate = aggregate_results([result_a, result_b])

    assert aggregate["miss_reason_counts"] == {"ocr_blank": 2, "fast_text_detect": 1}


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
