"""evaluation/evaluate/timing.py: per-page rows, display order, summaries."""
from __future__ import annotations

import math

import pytest

from raster_parser.evaluation.evaluate import timing


def test_flatten_page_timing_totals_and_debug_rows():
    row = timing.flatten_page_timing(
        {"phase1": 0.1, "phase2": 0.0, "phase3": 2.0, "phase4": 0.05},
        {"conversion": 0.3, "reconstructed": 0.2},
    )
    assert row["total"] == pytest.approx(2.15)
    assert row["debug.conversion"] == 0.3
    assert row["debug.total"] == pytest.approx(0.5)
    assert row["total_incl_debug"] == pytest.approx(2.65)


def test_flatten_empty_page_and_doc():
    assert timing.flatten_page_timing({}, {"conversion": 1.0}) == {}
    assert timing.flatten_doc_timing({}) == {}
    assert timing.flatten_doc_timing({"layer_save": 1.0, "label_overlays": 2.0})["total"] == 3.0


def test_row_order_and_leaf_rows():
    rows = [
        timing.flatten_page_timing({"phase3": 1.0, "phase1": 0.5}, {"extra_predictions": 0.1, "conversion": 0.2}),
        timing.flatten_page_timing({"phase1": 0.4, "phase4": 0.1}),
    ]
    assert timing.row_order(rows) == [
        "phase1", "phase3", "phase4", "total",
        "debug.conversion", "debug.extra_predictions", "debug.total", "total_incl_debug",
    ]
    assert timing.leaf_rows(timing.summarize_timings(rows)) == ["phase1", "phase3", "phase4"]


def test_summarize_timings_stats_and_sum():
    summary = timing.summarize_timings([{"phase1": 1.0, "total": 1.0}, {"phase1": 3.0, "total": 3.0}])
    st = summary["phase1"]
    assert (st["n"], st["min"], st["median"], st["max"], st["sum"]) == (2, 1.0, 2.0, 3.0, 4.0)
    assert timing.summarize_timings([]) == {}


def test_distribution_stats_empty_is_nan():
    st = timing.distribution_stats([])
    assert st["n"] == 0 and math.isnan(st["median"])


def test_format_timing_table():
    text = timing.format_timing_table(
        timing.summarize_timings([{"phase1": 1.0, "debug.conversion": 0.5}]), title="T")
    assert text.splitlines()[0] == "T"
    assert "debug conversion" in text
    assert timing.format_timing_table({}, title="T") == "T\n  (no timing data)"


def test_sub_steps_follow_their_phase_and_are_not_counted_twice():
    row = timing.flatten_page_timing(
        {"phase1": 0.5, "phase3": 3.0}, {"conversion": 0.2},
        {"phase3": {"ocr_detect": 2.0, "classify_pattern": 0.5}},
    )
    assert row["total"] == pytest.approx(3.5)
    assert row["phase3.ocr_detect"] == 2.0
    assert timing.row_order([row]) == [
        "phase1", "phase3", "phase3.ocr_detect", "phase3.classify_pattern", "total",
        "debug.conversion", "debug.total", "total_incl_debug",
    ]
    assert timing.leaf_rows(timing.summarize_timings([row])) == ["phase1", "phase3"]
    assert "phase3 ocr_detect" in timing.format_timing_table(timing.summarize_timings([row]), title="T")
