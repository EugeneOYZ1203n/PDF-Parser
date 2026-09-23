"""Tests for confusion_metrics.py -- reading order (category 7) and OCR
confusion characters (category 8)."""
from __future__ import annotations

import math

from rastervec.Evaluation.Evaluate.confusion_metrics import (
    CharStats,
    _reading_order_sort_key,
    align_chars,
    align_ops,
    char_events,
    char_stats_table,
    merge_char_stats,
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


# --------------------------------------------------------------------------
# Per-character accounting (char_events / char_stats_table)
# --------------------------------------------------------------------------
def test_align_ops_keeps_insertions():
    assert align_ops("AB", "AXB") == [("A", "A"), ("", "X"), ("B", "B")]


def test_align_chars_still_drops_insertions():
    assert align_chars("AB", "AXB") == [("A", "A"), ("B", "B")]


def test_char_stats_exact_match_all_detected():
    graph = build_overlap_graph([_gt("CAT")], [_pred("CAT", (0, 0, 10, 10))])
    table = char_stats_table(graph)
    assert {ch: (cs.detected, cs.total) for ch, cs in table.items()} == {
        "C": (1, 1), "A": (1, 1), "T": (1, 1),
    }
    assert table["A"].error_rate == 0.0


def test_char_stats_substitution_deletion_insertion():
    # CAT -> CUT: A misread as U; DOG -> DG: O dropped; BE -> BXE: X inserted.
    graph = build_overlap_graph(
        [_gt("CAT DOG BE", bbox=(0, 0, 30, 10))],
        [_pred("CUT DG BXE", (0, 0, 30, 10))],
    )
    table = char_stats_table(graph)
    assert table["A"].misclassified == 1
    assert table["A"].replacements == {"U": 1}
    assert table["O"].dropped == 1
    assert table["X"].inserted == 1
    assert table["X"].total == 0
    assert math.isnan(table["X"].error_rate)
    assert table["C"].detected == 1
    assert table["E"].detected == 1


def test_char_stats_unreached_region():
    graph = build_overlap_graph([_gt("AB", bbox=(0, 0, 10, 10))], [_pred("AB", (100, 100, 110, 110))])
    table = char_stats_table(graph)
    assert table["A"].unreached == 1
    assert table["A"].detected == 0
    assert table["A"].error_rate == 1.0


def test_char_events_carry_word_index():
    graph = build_overlap_graph([_gt("AA BC", bbox=(0, 0, 20, 10))], [_pred("AA BD", (0, 0, 20, 10))])
    mis = [ev for ev in char_events(graph) if ev.kind == "misclassified"]
    assert len(mis) == 1
    assert (mis[0].word_idx, mis[0].gt_word, mis[0].pred_word, mis[0].char, mis[0].replacement) == (
        1, "BC", "BD", "C", "D",
    )


def test_char_stats_totals_partition_gt_chars():
    graph = build_overlap_graph(
        [_gt("HELLO WORLD", bbox=(0, 0, 30, 10)), _gt("FAR", bbox=(200, 200, 210, 210))],
        [_pred("HELO W0RLDZ", (0, 0, 30, 10))],
    )
    table = char_stats_table(graph)
    assert sum(cs.total for cs in table.values()) == len("HELLOWORLD") + len("FAR")


def test_merge_char_stats_sums_fields():
    a = {"A": CharStats(detected=1, misclassified=1)}
    a["A"].replacements["4"] += 1
    b = {"A": CharStats(dropped=2), "B": CharStats(inserted=1)}
    merged = merge_char_stats([a, b])
    assert merged["A"].total == 4
    assert merged["A"].replacements == {"4": 1}
    assert merged["B"].inserted == 1
