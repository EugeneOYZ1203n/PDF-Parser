"""Plain stdlib HTML report builder for `scripts/pipeline_report_benchmark.py`.

No new dependency -- `html.escape` + f-strings. Charts stay as separate
PNG files on disk; this module only ever emits relative `<img src="...">`
references to them, so the rendered `report.html` must stay alongside the
`charts/` folder (and the linked run folders) it points at (same convention
`generate_pipeline_report.py` already uses for its own manifest).
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
.gallery { display: flex; flex-wrap: wrap; gap: .75rem; margin: .5rem 0 1.25rem; }
.card { width: 220px; font-size: .8rem; }
.card img { max-width: 100%; border: 1px solid #ccc; display: block; }
.card .cap { margin-top: 2px; word-break: break-word; }
.viewer-cmd { background: #222; color: #eee; padding: .6rem 1rem; font-family: monospace;
              font-size: .85rem; white-space: pre-wrap; overflow-x: auto; }
.section { background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 1rem 1.5rem; margin-bottom: 1rem; }
.empty { color: #888; font-style: italic; }
"""


def _table(headers: "list[str]", rows: "list[list[str]]") -> str:
    if not rows:
        return '<p class="empty">(no data)</p>'
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def _img(src: "Path | str", *, width: "int | None" = None) -> str:
    w = f' width="{width}"' if width else ""
    href = src.as_posix() if isinstance(src, Path) else str(src)
    return f'<img src="{escape(href)}"{w}>'


class ReportBuilder:
    def __init__(self, title: str) -> None:
        self._title = title
        self._parts: "list[str]" = []

    def add_key_section(self, key: str) -> None:
        self._parts.append(f'<h2 id="{escape(key)}">{escape(key)}</h2>')

    def add_group_header(self, name: str) -> None:
        """A "Text" / "Vector" major group header within the current key."""
        self._parts.append(f"<h3>{escape(name)}</h3>")

    def add_text_subsection(
        self, name: str, headers: "list[str]", rows: "list[list[str]]",
        chart_paths: "list[Path]" = (),
    ) -> None:
        self._parts.append(f"<h4>{escape(name)}</h4>")
        self._parts.append('<div class="section">')
        self._parts.append(_table(headers, rows))
        if chart_paths:
            imgs = "".join(_img(p) for p in chart_paths)
            self._parts.append(f'<div class="charts">{imgs}</div>')
        self._parts.append("</div>")

    def add_vector_subsection(
        self, name: str, headers: "list[str]", rows: "list[list[str]]",
        chart_paths: "list[Path]" = (),
    ) -> None:
        self.add_text_subsection(name, headers, rows, chart_paths)

    def add_image_gallery(self, run: str, folder_name: str, image_paths: "list[Path | str]") -> None:
        self._parts.append(f"<h4>{escape(run)} -- {escape(folder_name)}</h4>")
        if not image_paths:
            self._parts.append('<p class="empty">(no images)</p>')
            return
        cards = "".join(
            f'<div class="card">{_img(p, width=200)}'
            f'<div class="cap">{escape(Path(str(p)).name)}</div></div>'
            for p in image_paths
        )
        self._parts.append(f'<div class="gallery">{cards}</div>')

    def add_viewer_command(self, key: str, command: str) -> None:
        self._parts.append(f"<h4>Viewer command -- {escape(key)}</h4>")
        self._parts.append(f'<div class="viewer-cmd">{escape(command)}</div>')

    def add_raw_html(self, html: str) -> None:
        self._parts.append(html)

    def render(self) -> str:
        body = "\n".join(self._parts)
        return (
            f"<title>{escape(self._title)}</title>\n"
            f"<style>{_CSS}</style>\n"
            f"<h1>{escape(self._title)}</h1>\n{body}\n"
        )
