"""Plain stdlib HTML report builder for `scripts/pipeline_report_benchmark.py`.

`html.escape` + f-strings, no dependency. Charts stay as separate PNG files on disk; this module
only emits relative `<img src="...">` references, so `report.html` must stay next to its
`charts/` folder.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

_CSS = """
body { font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; background: #fafafa; }
h1 { border-bottom: 3px solid #333; padding-bottom: .3rem; }
h2 { margin-top: 2.5rem; border-bottom: 1px solid #ccc; padding-bottom: .2rem; }
h3 { margin-top: 1.5rem; color: #333; }
h4 { margin-top: 1rem; color: #555; }
table { border-collapse: collapse; margin: .75rem 0; font-size: .9rem; }
th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: left; }
th { background: #eee; }
.charts { display: flex; flex-wrap: wrap; gap: 1rem; margin: .75rem 0; }
.charts img { max-width: 480px; border: 1px solid #ddd; }
.viewer-cmd { background: #222; color: #eee; padding: .6rem 1rem; font-family: monospace;
              font-size: .85rem; white-space: pre-wrap; overflow-x: auto; }
.section { background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 1rem 1.5rem; margin-bottom: 1rem; }
.empty { color: #888; font-style: italic; }
"""


def table(headers: "list[str]", rows: "list[list[str]]") -> str:
    if not rows:
        return '<p class="empty">(no data)</p>'
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _img(src: "Path | str") -> str:
    href = src.as_posix() if isinstance(src, Path) else str(src)
    return f'<img src="{escape(href)}">'


class ReportBuilder:
    def __init__(self, title: str) -> None:
        self._title = title
        self._parts: "list[str]" = []

    def add_key_section(self, key: str) -> None:
        self._parts.append(f'<h2 id="{escape(key)}">{escape(key)}</h2>')

    def add_group_header(self, name: str) -> None:
        """A "Text" / "Timing" group header within the current section."""
        self._parts.append(f"<h3>{escape(name)}</h3>")

    def add_subsection(
        self, name: str, headers: "list[str]", rows: "list[list[str]]", chart_paths: "list[Path]" = (),
    ) -> None:
        self._parts.append(f"<h4>{escape(name)}</h4>")
        self._parts.append('<div class="section">')
        self._parts.append(table(headers, rows))
        if chart_paths:
            self._parts.append('<div class="charts">' + "".join(_img(p) for p in chart_paths) + "</div>")
        self._parts.append("</div>")

    def add_charts(self, chart_paths: "list[Path]") -> None:
        if chart_paths:
            self._parts.append('<div class="charts">' + "".join(_img(p) for p in chart_paths) + "</div>")

    def add_viewer_command(self, key: str, command: str) -> None:
        self._parts.append(f"<h4>Viewer command -- {escape(key)}</h4>")
        self._parts.append(f'<div class="viewer-cmd">{escape(command)}</div>')

    def add_raw_html(self, html: str) -> None:
        self._parts.append(html)

    def render(self) -> str:
        body = "\n".join(self._parts)
        return (
            f"<!doctype html>\n<meta charset=\"utf-8\">\n<title>{escape(self._title)}</title>\n"
            f"<style>{_CSS}</style>\n<h1>{escape(self._title)}</h1>\n{body}\n"
        )
