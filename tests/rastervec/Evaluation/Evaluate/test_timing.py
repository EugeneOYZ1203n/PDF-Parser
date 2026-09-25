"""Evaluation.Evaluate.timing -- per-page timing flattening + summaries."""
from __future__ import annotations

import pytest

from rastervec.Evaluation.Evaluate import timing


def test_flatten_adds_substeps_other_and_total():
    row = timing.flatten_page_timing(
        {"phase1": 1.0, "phase2": 0.5, "phase3": 10.0, "phase4": 0.5},
        {"fast": 3.0, "ocr": 6.0},
    )
    assert row["phase3.fast"] == 3.0
    assert row["phase3.ocr"] == 6.0
    assert row["phase3.other"] == pytest.approx(1.0)
    # total = phases only, sub-steps never double-count
    assert row["total"] == pytest.approx(12.0)


def test_flatten_legacy_and_empty():
    assert timing.flatten_page_timing({"legacy": 4.0}) == {"legacy": 4.0, "total": 4.0}
    assert timing.flatten_page_timing({}) == {}


def test_flatten_other_clamped_at_zero():
    row = timing.flatten_page_timing({"phase3": 1.0}, {"ocr": 1.5})
    assert row["phase3.other"] == 0.0


def test_row_order_places_substeps_under_phase3():
    rows = [
        timing.flatten_page_timing(
            {"phase4": 0.1, "phase3": 2.0, "phase1": 0.1, "phase2": 0.1},
            {"ocr": 1.0, "fast": 0.5},
        ),
    ]
    assert timing.row_order(rows) == [
        "phase1", "phase2", "phase3", "phase3.ocr", "phase3.fast", "phase3.other", "phase4", "total",
    ]


def test_summarize_timings_stats_and_sum():
    rows = [
        timing.flatten_page_timing({"phase1": 1.0, "phase3": 2.0}),
        timing.flatten_page_timing({"phase1": 3.0, "phase3": 4.0}),
    ]
    summary = timing.summarize_timings(rows)
    assert summary["phase1"]["n"] == 2
    assert summary["phase1"]["median"] == 2.0
    assert summary["phase1"]["sum"] == 4.0
    assert summary["total"]["sum"] == 10.0
    assert summary["total"]["max"] == 7.0
    assert timing.summarize_timings([]) == {}


def test_leaf_rows_replace_phase3_with_substeps():
    summary = timing.summarize_timings([
        timing.flatten_page_timing({"phase1": 1.0, "phase3": 2.0}, {"ocr": 1.0}),
    ])
    assert timing.leaf_rows(summary) == ["phase1", "phase3.ocr", "phase3.other"]
    no_subs = timing.summarize_timings([timing.flatten_page_timing({"legacy": 3.0})])
    assert timing.leaf_rows(no_subs) == ["legacy"]


def test_format_timing_table():
    summary = timing.summarize_timings([
        timing.flatten_page_timing({"phase3": 2.0}, {"ocr": 1.0}),
    ])
    text = timing.format_timing_table(summary, title="T")
    assert text.splitlines()[0] == "T"
    assert "  ocr" in text
    assert "total" in text
    assert "(no timing data)" in timing.format_timing_table({}, title="T")


def test_flatten_moves_backend_debug_render_out_of_pipeline_total():
    row = timing.flatten_page_timing(
        {"phase1": 1.0, "phase3": 10.0},
        {"ocr": 6.0, "debug_render": 3.0},
        {"stage_layers": 0.5, "conversion": 1.5},
    )
    assert row["phase3"] == pytest.approx(7.0)  # production cost only
    assert "phase3.debug_render" not in row
    assert row["phase3.other"] == pytest.approx(1.0)
    assert row["total"] == pytest.approx(8.0)
    assert row["debug.backend_layers"] == 3.0
    assert row["debug.total"] == pytest.approx(5.0)
    assert row["total_incl_debug"] == pytest.approx(13.0)


def test_row_order_and_leaf_rows_with_debug_group():
    rows = [timing.flatten_page_timing(
        {"phase3": 2.0}, {"ocr": 1.0, "debug_render": 0.5}, {"conversion": 0.1, "stage_layers": 0.2},
    )]
    assert timing.row_order(rows) == [
        "phase3", "phase3.ocr", "phase3.other", "total",
        "debug.backend_layers", "debug.stage_layers", "debug.conversion",
        "debug.total", "total_incl_debug",
    ]
    summary = timing.summarize_timings(rows)
    assert timing.leaf_rows(summary) == ["phase3.ocr", "phase3.other"]  # chart stays pipeline-only
    assert "debug stage_layers" in timing.format_timing_table(summary, title="T")


def test_legacy_page_gets_debug_rows_and_doc_timing():
    row = timing.flatten_page_timing({"legacy": 4.0}, None, {"stage_layers": 1.0})
    assert row["total"] == 4.0 and row["total_incl_debug"] == 5.0
    assert timing.flatten_doc_timing({"layer_save": 0.5, "label_overlays": 1.0}) == {
        "layer_save": 0.5, "label_overlays": 1.0, "total": 1.5,
    }
    assert timing.flatten_doc_timing({}) == {}
