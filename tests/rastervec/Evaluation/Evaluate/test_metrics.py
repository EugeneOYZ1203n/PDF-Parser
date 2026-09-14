"""Tests for the reworked 4-text-type metrics suite."""
from __future__ import annotations

from rastervec.Evaluation.Evaluate.metrics import (
    TEXT_TYPES,
    GtRegion,
    MetricConfig,
    Prediction,
    bbox_accuracy_stats,
    bbox_accuracy_unclassified,
    build_overlap_graph,
    build_overlap_graphs_by_type,
    char_overlap_stats,
    classification_funnel_stats,
    evaluate_text_metrics,
    aggregate_text_metrics,
    overlay_boxes_by_type,
    rotation_stats,
    text_label_stats,
    word_overlap_stats,
)


def _gt(text, bbox=(0, 0, 10, 10), rot=0, text_type="native_to_vector"):
    return GtRegion(page_index=0, bbox=bbox, text=text, expected_rotation=rot, text_type=text_type)


def _pred(text, bbox=(0, 0, 10, 10), rot=0):
    return Prediction(text=text, bbox=bbox, rotation=rot)


def test_char_overlap_stats_exact_match():
    g = _gt("HELLO")
    graph = build_overlap_graph([g], [_pred("HELLO")])
    stats = char_overlap_stats(graph, "native_to_vector")
    assert stats.matched == 5
    assert stats.total_gt == 5
    assert stats.unclassified == 0
    assert stats.missing == 0
    assert stats.precision.value == 1.0
    assert stats.recall.value == 1.0


def test_char_overlap_stats_unclassified_prediction():
    g = _gt("HELLO")
    spurious = _pred("WORLD", bbox=(100, 100, 110, 110))
    graph = build_overlap_graph([g], [_pred("HELLO"), spurious])
    stats = char_overlap_stats(graph, "native_to_vector")
    assert stats.matched == 5
    assert stats.unclassified == 5  # WORLD's 5 chars, zero overlap
    assert stats.precision.value == 0.5


def test_word_overlap_stats_partial_recall():
    g = _gt("HELLO WORLD")
    graph = build_overlap_graph([g], [_pred("HELLO")])
    stats = word_overlap_stats(graph, "native_to_vector")
    assert stats.matched == 1
    assert stats.total_gt == 2
    assert stats.missing == 1


def test_bbox_accuracy_stats_perfect_iou():
    g = _gt("X", bbox=(0, 0, 10, 10))
    graph = build_overlap_graph([g], [_pred("X", bbox=(0, 0, 10, 10))])
    stats = bbox_accuracy_stats(graph, "native_to_vector")
    assert stats.mean_iou.value == 1.0
    assert stats.n_gt == 1
    assert stats.n_localized == 1


def test_bbox_accuracy_unclassified_counts_spurious_across_all_types():
    gt_by_type = {t: [] for t in TEXT_TYPES}
    gt_by_type["native_to_vector"] = [_gt("A", bbox=(0, 0, 5, 5))]
    preds = [_pred("A", bbox=(0, 0, 5, 5)), _pred("SPURIOUS", bbox=(50, 50, 60, 60))]
    graphs = build_overlap_graphs_by_type(gt_by_type, preds)
    stats = bbox_accuracy_unclassified(graphs)
    assert stats.spurious_pred_count == 1


def test_rotation_stats_buckets():
    g0 = _gt("A", rot=0)
    graph = build_overlap_graph([g0], [_pred("A", rot=90)])
    stats = rotation_stats(graph, "native_to_vector")
    assert stats.buckets.off_90 == 1
    assert stats.buckets.correct == 0
    assert stats.mean_error_deg == 90.0
    assert stats.rmse_deg == 90.0


def test_rotation_stats_circular_270_is_off_by_90():
    g0 = _gt("A", rot=0)
    graph = build_overlap_graph([g0], [_pred("A", rot=270)])
    stats = rotation_stats(graph, "native_to_vector")
    assert stats.buckets.off_90 == 1


def test_classification_funnel_stats():
    stats = classification_funnel_stats({"a", "b", "c"}, {"a", "b"}, "native_to_vector")
    assert stats.n_gt_vectors == 3
    assert stats.n_survived == 2
    assert stats.survival_rate.value == 2 / 3


def test_classification_funnel_stats_no_gt_vectors_is_na():
    stats = classification_funnel_stats(set(), set(), "native_to_vector")
    assert not stats.survival_rate.applicable


def test_text_label_stats_counts():
    class E:
        def __init__(self, text, sigs=()):
            self.text = text
            self.vector_signatures = list(sigs)

    entries_by_type = {t: [] for t in TEXT_TYPES}
    entries_by_type["original_vector"] = [E("HELLO WORLD", sigs=["a", "b"])]
    stats = {s.text_type: s for s in text_label_stats(entries_by_type)}
    ov = stats["original_vector"]
    assert ov.label_count == 1
    assert ov.word_count == 2
    assert ov.char_count == 10
    assert ov.vector_count == 2
    assert stats["native_to_vector"].label_count == 0


def test_evaluate_text_metrics_builds_all_four_types():
    gt_by_type = {t: [] for t in TEXT_TYPES}
    gt_by_type["native_to_vector"] = [_gt("HELLO")]
    entries_by_type = {t: [] for t in TEXT_TYPES}
    result = evaluate_text_metrics(gt_by_type, entries_by_type, [_pred("HELLO")])
    assert set(result.by_type) == set(TEXT_TYPES)
    assert result.by_type["native_to_vector"].char_overlap.matched == 5
    assert result.by_type["original_vector"].label_stats.label_count == 0
    assert result.by_type["native_to_vector"].funnel is not None
    assert result.by_type["vector_to_raster"].funnel is None


def test_aggregate_text_metrics_micro_averages():
    gt_by_type = {t: [] for t in TEXT_TYPES}
    entries_by_type = {t: [] for t in TEXT_TYPES}
    gt_by_type["native_to_vector"] = [_gt("HELLO")]
    r1 = evaluate_text_metrics(gt_by_type, entries_by_type, [_pred("HELLO")])
    gt_by_type2 = {t: [] for t in TEXT_TYPES}
    gt_by_type2["native_to_vector"] = [_gt("WORLD")]
    r2 = evaluate_text_metrics(gt_by_type2, entries_by_type, [_pred("WRLD")])
    agg = aggregate_text_metrics([r1, r2])
    co = agg.by_type["native_to_vector"].char_overlap
    assert co.total_gt == 10
    assert co.matched == 5 + 4


def test_overlay_boxes_by_type_dashes_and_colors():
    gt_by_type = {t: [] for t in TEXT_TYPES}
    gt_by_type["native_to_vector"] = [_gt("A", bbox=(0, 0, 5, 5))]
    graphs = build_overlap_graphs_by_type(gt_by_type, [_pred("A", bbox=(0, 0, 5, 5))])
    boxes = overlay_boxes_by_type(graphs)
    assert boxes  # at least the matched gt + pred box
    assert any(dashes == "[4 3] 0" for _bbox, _rgb, dashes in boxes)
