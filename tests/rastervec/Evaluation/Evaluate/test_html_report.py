"""Smoke tests for html_report.py -- pure string templating, no rendering lib."""
from __future__ import annotations

from pathlib import Path

from rastervec.Evaluation.Evaluate.html_report import ReportBuilder


def test_render_includes_title_and_key_section():
    b = ReportBuilder("My Report")
    b.add_key_section("pdf:A")
    html = b.render()
    assert "My Report" in html
    assert "pdf:A" in html


def test_add_text_subsection_renders_table_and_charts():
    b = ReportBuilder("t")
    b.add_key_section("k")
    b.add_text_subsection(
        "Char", ["type", "matched"], [["native_to_vector", "10/12"]],
        chart_paths=[Path("charts/k__char.png")],
    )
    html = b.render()
    assert "Char" in html
    assert "native_to_vector" in html
    assert "10/12" in html
    assert 'src="charts/k__char.png"' in html


def test_add_text_subsection_empty_rows_shows_placeholder():
    b = ReportBuilder("t")
    b.add_text_subsection("Char", ["a"], [])
    assert "(no data)" in b.render()


def test_add_image_gallery():
    b = ReportBuilder("t")
    b.add_image_gallery("current", "paddle_detect_images", [Path("a.png"), Path("b.png")])
    html = b.render()
    assert 'src="a.png"' in html
    assert 'src="b.png"' in html


def test_add_viewer_command_escapes_and_includes_text():
    b = ReportBuilder("t")
    b.add_viewer_command("pdf:A", 'python scripts/pipeline_report_viewer.py "a" "b"')
    html = b.render()
    assert "pipeline_report_viewer.py" in html
