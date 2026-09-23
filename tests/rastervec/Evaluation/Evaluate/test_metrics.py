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
    combine_text_metrics_by_type,
    evaluate_text_metrics,
    aggregate_text_metrics,
    font_size_distribution,
    overlay_boxes_by_type,
    rotation_outcome,
    rotation_stats,
    text_label_stats,
    word_overlap_stats,
)
from rastervec.Evaluation.Evaluate.confusion_metrics import extra_predicted_chars


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


def test_combine_text_metrics_by_type_routes_each_type_from_its_owning_result():
    vec_gt = {t: [] for t in TEXT_TYPES}
    vec_gt["native_to_vector"] = [_gt("HELLO")]
    res_vec = evaluate_text_metrics(vec_gt, {t: [] for t in TEXT_TYPES}, [_pred("HELLO")])

    raster_gt = {t: [] for t in TEXT_TYPES}
    raster_gt["vector_to_raster"] = [_gt("WORLD", text_type="vector_to_raster")]
    res_raster = evaluate_text_metrics(
        raster_gt, {t: [] for t in TEXT_TYPES}, [_pred("SPURIOUS", bbox=(100, 100, 110, 110))],
    )

    combined = combine_text_metrics_by_type({
        "native_to_vector": res_vec, "original_vector": res_vec,
        "vector_to_raster": res_raster, "original_raster": res_raster,
        "native_to_raster": res_raster,
    })
    assert combined.by_type["native_to_vector"].char_overlap.matched == 5
    assert combined.by_type["vector_to_raster"].char_overlap.total_gt == 5  # "WORLD"
    # spurious predictions summed once per distinct result, not once per type
    assert combined.bbox_unclassified.spurious_pred_count == (
        res_vec.bbox_unclassified.spurious_pred_count
        + res_raster.bbox_unclassified.spurious_pred_count
    )


def test_font_size_distribution_pt_units_for_vector_types():
    g = _gt("HELLO", bbox=(0, 0, 10, 12))  # height 12
    graph = build_overlap_graph([g], [_pred("HELLO", bbox=(0, 0, 10, 12))])
    dist = font_size_distribution(graph, "native_to_vector")
    assert dist.unit == "pt"
    assert dist.all_sizes == (12.0,)
    assert dist.detected_sizes == (12.0,)


def test_font_size_distribution_px_units_for_original_raster():
    g = _gt("HELLO", bbox=(0, 0, 10, 12), text_type="original_raster")
    graph = build_overlap_graph([g], [])
    dist = font_size_distribution(graph, "original_raster", dpi=144.0)
    assert dist.unit == "px"
    assert dist.all_sizes == (24.0,)  # 12pt * (144/72)
    assert dist.detected_sizes == ()


def test_font_size_distribution_detected_excludes_missed_gt():
    g1 = _gt("A", bbox=(0, 0, 5, 10))
    g2 = _gt("B", bbox=(100, 100, 105, 106))
    graph = build_overlap_graph([g1, g2], [_pred("A", bbox=(0, 0, 5, 10))])
    dist = font_size_distribution(graph, "native_to_vector")
    assert dist.all_sizes == (10.0, 6.0)
    assert dist.detected_sizes == (10.0,)


def test_extra_predicted_chars_counts_only_zero_overlap_predictions():
    g = _gt("HELLO")
    graph = build_overlap_graph([g], [_pred("HELLO"), _pred("WORLD", bbox=(100, 100, 110, 110))])
    extra = extra_predicted_chars(graph)
    assert extra == {"W": 1, "O": 1, "R": 1, "L": 1, "D": 1}


def test_extra_predicted_chars_empty_when_all_predictions_overlap():
    g = _gt("HELLO")
    graph = build_overlap_graph([g], [_pred("HELLO")])
    assert extra_predicted_chars(graph) == {}


def test_evaluate_text_metrics_includes_font_size_and_extra_chars():
    gt_by_type = {t: [] for t in TEXT_TYPES}
    gt_by_type["native_to_vector"] = [_gt("HELLO")]
    entries_by_type = {t: [] for t in TEXT_TYPES}
    preds = [_pred("HELLO"), _pred("SPURIOUS", bbox=(100, 100, 110, 110))]
    result = evaluate_text_metrics(gt_by_type, entries_by_type, preds)
    r = result.by_type["native_to_vector"]
    assert r.font_size.unit == "pt"
    assert len(r.font_size.all_sizes) == 1
    assert sum(result.extra_chars.values()) == len("SPURIOUS")


def test_evaluate_text_metrics_extra_chars_excludes_preds_overlapping_any_type():
    # A prediction overlapping one type's GT but not another's must NOT be
    # counted as "extra" -- only truly cross-type-unclassified predictions
    # (zero overlap in EVERY type's graph) count.
    gt_by_type = {t: [] for t in TEXT_TYPES}
    gt_by_type["native_to_vector"] = [_gt("HELLO", bbox=(0, 0, 50, 10))]
    entries_by_type = {t: [] for t in TEXT_TYPES}
    preds = [_pred("HELLO", bbox=(0, 0, 50, 10))]
    result = evaluate_text_metrics(gt_by_type, entries_by_type, preds)
    assert sum(result.extra_chars.values()) == 0


def test_overlay_boxes_by_type_dashes_and_colors():
    gt_by_type = {t: [] for t in TEXT_TYPES}
    gt_by_type["native_to_vector"] = [_gt("A", bbox=(0, 0, 5, 5))]
    graphs = build_overlap_graphs_by_type(gt_by_type, [_pred("A", bbox=(0, 0, 5, 5))])
    boxes = overlay_boxes_by_type(graphs)
    assert boxes  # at least the matched gt + pred box
    assert any(dashes == "[4 3] 0" for _bbox, _rgb, dashes in boxes)


def test_rotation_outcome_matches_rotation_stats():
    gts = [_gt("A", bbox=(0, 0, 10, 10), rot=0), _gt("B", bbox=(20, 0, 30, 10), rot=90)]
    preds = [_pred("A", bbox=(0, 0, 10, 10), rot=180), _pred("B", bbox=(20, 0, 30, 10), rot=90)]
    graph = build_overlap_graph(gts, preds)
    assert rotation_outcome(graph, 0) == (180, 180, "off_180")
    assert rotation_outcome(graph, 1) == (90, 0, "correct")
    stats = rotation_stats(graph, "native_to_vector")
    assert (stats.buckets.correct, stats.buckets.off_90, stats.buckets.off_180) == (1, 0, 1)


def test_aggregate_text_metrics_merges_char_stats():
    entries_by_type = {t: [] for t in TEXT_TYPES}
    gt1 = {t: [] for t in TEXT_TYPES}
    gt1["native_to_vector"] = [_gt("HELLO")]
    gt2 = {t: [] for t in TEXT_TYPES}
    gt2["native_to_vector"] = [_gt("HELP")]
    r1 = evaluate_text_metrics(gt1, entries_by_type, [_pred("HELLO")])
    r2 = evaluate_text_metrics(gt2, entries_by_type, [_pred("HEAP")])
    agg = aggregate_text_metrics([r1, r2])
    cs = agg.by_type["native_to_vector"].char_stats
    assert cs["L"].total == 3
    assert cs["L"].detected == 2
    assert cs["L"].misclassified == 1
    assert cs["L"].replacements == {"A": 1}
