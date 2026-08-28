from __future__ import annotations

from rastervec.Evaluation.Evaluate.benchmark import aggregate_results, format_report
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
