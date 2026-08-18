"""All PyMuPDF (pymupdf/fitz) calls live here. Nothing outside this module
should import pymupdf directly for extraction purposes."""
from __future__ import annotations

from itertools import groupby

import pymupdf as fitz

from layers import OverlayItem, rgb_to_hex


class PdfDocument:
    def __init__(self, path: str):
        self.path = path
        self.doc = fitz.open(path)

    @property
    def page_count(self) -> int:
        return self.doc.page_count

    def get_page(self, n: int) -> "fitz.Page":
        return self.doc[n]

    def render_pixmap(self, n: int, zoom: float) -> "fitz.Pixmap":
        page = self.get_page(n)
        matrix = fitz.Matrix(zoom, zoom)
        return page.get_pixmap(matrix=matrix)

    def close(self) -> None:
        self.doc.close()


def extract_text_items(page: "fitz.Page") -> list[OverlayItem]:
    words = page.get_text("words")
    # Natural reading-order sort: bucket by rounded y0, then sort each
    # bucket left-to-right (per the PyMuPDF "natural reading order" wiki).
    words = sorted(words, key=lambda w: (round(w[1]), w[0]))
    items = []
    for y_bucket, group in groupby(words, key=lambda w: round(w[1])):
        for w in sorted(group, key=lambda w: w[0]):
            x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
            items.append(
                OverlayItem(
                    bbox=fitz.Rect(x0, y0, x1, y1),
                    kind="word",
                    shape="rect",
                    label=text,
                )
            )
    return items


def extract_image_items(page: "fitz.Page") -> list[OverlayItem]:
    items = []
    for img in page.get_images(full=True):
        xref = img[0]
        try:
            bboxes = page.get_image_bbox(img, transform=False)
        except ValueError:
            continue
        if isinstance(bboxes, fitz.Rect):
            bboxes = [bboxes]
        for bbox in bboxes:
            items.append(
                OverlayItem(
                    bbox=bbox,
                    kind="image",
                    shape="rect",
                    attrs={"xref": xref},
                    label=f"xref={xref}",
                )
            )
    return items


def extract_annot_items(page: "fitz.Page") -> list[OverlayItem]:
    items = []
    for annot in page.annots() or []:
        items.append(
            OverlayItem(
                bbox=annot.rect,
                kind="annot",
                shape="rect",
                attrs={"type": annot.type[1] if annot.type else ""},
                label=annot.type[1] if annot.type else "",
            )
        )
    return items


def _round_color(color):
    if not color:
        return None
    return tuple(round(c, 3) for c in color)


def extract_drawing_items(page: "fitz.Page") -> list[OverlayItem]:
    items = []
    for d in page.get_drawings():
        path_type = d.get("type", "")
        stroke_color = _round_color(d.get("color"))
        fill_color = _round_color(d.get("fill"))
        attrs = {
            "path_type": path_type,
            "stroke_color": stroke_color,
            "fill_color": fill_color,
        }
        for it in d.get("items", []):
            op = it[0]
            if op == "l":
                p1, p2 = it[1], it[2]
                items.append(
                    OverlayItem(
                        bbox=fitz.Rect(p1, p2),
                        kind="l",
                        shape="line",
                        points=[p1, p2],
                        attrs={**attrs, "kind": "l"},
                    )
                )
            elif op == "re":
                rect = it[1]
                items.append(
                    OverlayItem(
                        bbox=fitz.Rect(rect),
                        kind="re",
                        shape="rect",
                        attrs={**attrs, "kind": "re"},
                    )
                )
            elif op == "qu":
                quad = it[1]
                points = [quad.ul, quad.ur, quad.lr, quad.ll]
                items.append(
                    OverlayItem(
                        bbox=quad.rect,
                        kind="qu",
                        shape="polygon",
                        points=points,
                        attrs={**attrs, "kind": "qu"},
                    )
                )
            elif op == "c":
                p1, p2, p3, p4 = it[1], it[2], it[3], it[4]
                points = [p1, p2, p3, p4]
                xs = [p.x for p in points]
                ys = [p.y for p in points]
                items.append(
                    OverlayItem(
                        bbox=fitz.Rect(min(xs), min(ys), max(xs), max(ys)),
                        kind="c",
                        shape="line",
                        points=[p1, p4],
                        attrs={**attrs, "kind": "c"},
                    )
                )
    return items


def collect_drawing_colors(page: "fitz.Page") -> tuple[dict[str, str], dict[str, str]]:
    """Returns (stroke_options, fill_options): value -> display label,
    for populating the dynamic color sub-filter swatches."""
    strokes: dict[str, str] = {}
    fills: dict[str, str] = {}
    saw_no_stroke = False
    saw_no_fill = False
    for d in page.get_drawings():
        stroke = _round_color(d.get("color"))
        fill = _round_color(d.get("fill"))
        if stroke:
            strokes.setdefault(stroke, rgb_to_hex(stroke))
        else:
            saw_no_stroke = True
        if fill:
            fills.setdefault(fill, rgb_to_hex(fill))
        else:
            saw_no_fill = True
    if saw_no_stroke:
        strokes.setdefault("__none__", "(none)")
    if saw_no_fill:
        fills.setdefault("__none__", "(none)")
    return strokes, fills
