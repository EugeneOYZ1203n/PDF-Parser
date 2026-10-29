"""evaluation/evaluate/html_report.py."""

from __future__ import annotations

from pathlib import Path

from raster_parser.evaluation.evaluate.html_report import ReportBuilder


def test_render_includes_title_sections_tables_and_charts():
    b = ReportBuilder("Bench <1>")
    b.add_key_section("pdf:A")
    b.add_subsection(
        "Char overlap", ["type", "run"], [["native_to_vector", "R 1.0"]], [Path("charts/a.png")]
    )
    b.add_subsection("Empty", ["x"], [])
    b.add_viewer_command("pdf:A", 'python viewer.py "a<b"')
    html = b.render()
    assert "<title>Bench &lt;1&gt;</title>" in html
    assert '<h2 id="pdf:A">pdf:A</h2>' in html
    assert "<td>native_to_vector</td>" in html
    assert '<img src="charts/a.png">' in html
    assert '<p class="empty">(no data)</p>' in html
    assert "a&lt;b" in html
