"""evaluation/evaluate/timing.py: per-page rows, display order, summaries."""

from __future__ import annotations

import math

import pytest

from raster_parser.evaluation.evaluate import timing


def test_flatten_page_timing_totals_and_debug_rows():
    row = timing.flatten_page_timing(
        {
            "native_read": 0.1,
            "raster_vectorize": 0.0,
            "vector_classify": 2.0,
            "output_assemble": 0.05,
        },
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
        timing.flatten_page_timing(
            {"vector_classify": 1.0, "native_read": 0.5},
            {"extra_predictions": 0.1, "conversion": 0.2},
        ),
        timing.flatten_page_timing({"native_read": 0.4, "output_assemble": 0.1}),
    ]
    assert timing.row_order(rows) == [
        "native_read",
        "vector_classify",
        "output_assemble",
        "total",
        "debug.conversion",
        "debug.extra_predictions",
        "debug.total",
        "total_incl_debug",
    ]
    assert timing.leaf_rows(timing.summarize_timings(rows)) == [
        "native_read",
        "vector_classify",
        "output_assemble",
    ]


def test_summarize_timings_stats_and_sum():
    summary = timing.summarize_timings(
        [{"native_read": 1.0, "total": 1.0}, {"native_read": 3.0, "total": 3.0}]
    )
    st = summary["native_read"]
    assert (st["n"], st["min"], st["median"], st["max"], st["sum"]) == (2, 1.0, 2.0, 3.0, 4.0)
    assert timing.summarize_timings([]) == {}


def test_distribution_stats_empty_is_nan():
    st = timing.distribution_stats([])
    assert st["n"] == 0 and math.isnan(st["median"])


def test_format_timing_table():
    text = timing.format_timing_table(
        timing.summarize_timings([{"native_read": 1.0, "debug.conversion": 0.5}]), title="T"
    )
    assert text.splitlines()[0] == "T"
    assert "debug conversion" in text
    assert timing.format_timing_table({}, title="T") == "T\n  (no timing data)"
