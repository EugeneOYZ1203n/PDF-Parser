"""Tests for confusion_metrics.py -- reading order (category 7) and OCR
confusion characters (category 8)."""
from __future__ import annotations

from rastervec.Evaluation.Evaluate.confusion_metrics import (
    _reading_order_sort_key,
    align_chars,
    closest_pred_word,
    confusion_table,
    reading_order_stats,
)
from rastervec.Evaluation.Evaluate.metrics import GtRegion, Prediction, build_overlap_graph


def _gt(text, bbox=(0, 0, 10, 10), rot=0):
    return GtRegion(page_index=0, bbox=bbox, text=text, expected_rotation=rot, text_type="native_to_vector")


def _pred(text, bbox):
    return Prediction(text=text, bbox=bbox, rotation=0)


def test_reading_order_sort_key_horizontal():
    assert _reading_order_sort_key((5, 0, 6, 1), 0) == (5, 0)


def test_reading_order_sort_key_180_reverses_x():
    assert _reading_order_sort_key((5, 0, 6, 1), 180) == (-5, 0)


def test_reading_order_sort_key_90_downward():
    assert _reading_order_sort_key((5, 3, 6, 4), 90) == (3, 5)


def test_reading_order_sort_key_270_upward():
    assert _reading_order_sort_key((5, 3, 6, 4), 270) == (-3, 5)


def test_reading_order_stats_in_order():
    g = _gt("HELLO WORLD", bbox=(0, 0, 20, 10))
    preds = [_pred("HELLO", (0, 0, 8, 10)), _pred("WORLD", (9, 0, 20, 10))]
    graph = build_overlap_graph([g], preds)
    stats = reading_order_stats(graph, "native_to_vector")
    assert stats.n_with_overlap == 1
    assert stats.n_in_order == 1
    assert stats.edit_distance_mean == 0


def test_reading_order_stats_out_of_order():
    g = _gt("HELLO WORLD", bbox=(0, 0, 20, 10))
    preds = [_pred("WORLD", (0, 0, 8, 10)), _pred("HELLO", (9, 0, 20, 10))]
    graph = build_overlap_graph([g], preds)
    stats = reading_order_stats(graph, "native_to_vector")
    assert stats.n_in_order == 0
    assert stats.edit_distance_min == 2


def test_closest_pred_word_picks_min_edit_distance():
    assert closest_pred_word("HELLO", ["HELLO", "HELP"]) == "HELLO"
    assert closest_pred_word("HELLO", ["HELP", "WORLD"]) == "HELP"


def test_closest_pred_word_empty_returns_none():
    assert closest_pred_word("HELLO", []) is None


def test_align_chars_substitution():
    pairs = align_chars("CAT", "CUT")
    assert pairs == [("C", "C"), ("A", "U"), ("T", "T")]


def test_align_chars_deletion():
    pairs = align_chars("CAT", "CT")
    assert pairs == [("C", "C"), ("A", ""), ("T", "T")]


def test_confusion_table_records_unmatched_word_substitution():
    g = _gt("CAT")
    preds = [_pred("CUT", (0, 0, 10, 10))]
    graph = build_overlap_graph([g], preds)
    table = confusion_table(graph)
    assert table["A"]["U"] == 1


def test_confusion_table_excludes_self_matches():
    # "CAT" vs "COT" is one real substitution (A->O); align_chars also
    # emits correct C->C / T->T match steps, which must NOT be counted.
    g = _gt("CAT")
    preds = [_pred("COT", (0, 0, 10, 10))]
    graph = build_overlap_graph([g], preds)
    table = confusion_table(graph)
    assert table == {"A": {"O": 1}}
    assert "C" not in table
    assert "T" not in table


def test_confusion_table_skips_exact_word_matches():
    g = _gt("CAT")
    preds = [_pred("CAT", (0, 0, 10, 10))]
    graph = build_overlap_graph([g], preds)
    table = confusion_table(graph)
    assert table == {}
