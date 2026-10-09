from __future__ import annotations

import pymupdf as pymupdf
from raster_parser.models import Vector

# Join tolerance (SVG/PDF units before dpi scale; ~pixels from autotrace).
_STROKE_JOIN_EPS = 0.05

def _item_points(item: tuple) -> list[pymupdf.Point]:
    """give me every coordinate point involved in this segment" (used to compute bounding boxes)."""
    op = item[0]
    if op == "l":
        return [item[1], item[2]]
    if op == "c":
        return [item[1], item[2], item[3], item[4]]
    if op == "re":
        r = item[1]
        return [pymupdf.Point(r.x0, r.y0), pymupdf.Point(r.x1, r.y1)]
    if op == "qu":
        q = item[1]
        return [q.ul, q.ur, q.ll, q.lr]
    return []


def _item_start(item: tuple) -> pymupdf.Point | None:
    #"where does this segment begin"
    op = item[0]
    if op in ("l", "c"):
        return item[1]
    if op == "qu":
        return item[1].ul
    if op == "re":
        r = item[1]
        return pymupdf.Point(r.x0, r.y0)
    return None


def _item_end(item: tuple) -> pymupdf.Point | None:
    #"where does this segment end"
    op = item[0]
    if op == "l":
        return item[2]
    if op == "c":
        return item[4]
    if op == "qu":
        return item[1].lr
    if op == "re":
        r = item[1]
        return pymupdf.Point(r.x1, r.y1)
    return None


def _items_rect(items: list) -> tuple[float, float, float, float]:
    # folds a whole list of segments down into one (min_x, min_y, max_x, max_y) bounding box
    xs: list[float] = []
    ys: list[float] = []
    for it in items:
        for p in _item_points(it):
            xs.append(float(p.x))
            ys.append(float(p.y))
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def _split_items_into_strokes(items: list, *, join_eps: float = _STROKE_JOIN_EPS) -> list[list]:
    """Split a flat get_drawings() ``items`` list into continuous strokes.

    Autotrace writes one SVG ``<path d="M...M...M...">`` with a moveto per stroke.
    MuPDF folds that into a single drawing whose segments jump whenever a new ``M``
    starts. A new stroke begins when the next segment's start is not near the previous
    end (or the op is a standalone ``re``/``qu``).
    """
    if not items:
        return []
    strokes: list[list] = [[items[0]]]
    for it in items[1:]:
        op = it[0]
        # Rects / quads are closed primitives; always their own stroke.
        if op in ("re", "qu") or strokes[-1][-1][0] in ("re", "qu"):
            strokes.append([it])
            continue
        prev_end = _item_end(strokes[-1][-1])
        cur_start = _item_start(it)
        if prev_end is None or cur_start is None:
            strokes.append([it])
            continue
        dx = float(cur_start.x) - float(prev_end.x)
        dy = float(cur_start.y) - float(prev_end.y)
        if dx * dx + dy * dy > join_eps * join_eps:
            strokes.append([it])
        else:
            strokes[-1].append(it)
    return strokes


def _vectors_from_drawing(d: dict) -> list[Vector]:
    """One Vector per continuous stroke inside a MuPDF drawing dict."""
    base = Vector.from_drawing(d)
    parts = _split_items_into_strokes(base.items)
    if len(parts) <= 1:
        return [base]
    out: list[Vector] = []
    for i, items in enumerate(parts):
        out.append(
            Vector(
                items=items,
                type=base.type,
                rect=_items_rect(items),
                fill=base.fill,
                color=base.color,
                width=base.width,
                even_odd=base.even_odd,
                close_path=base.close_path,
                dashes=base.dashes,
                fill_opacity=base.fill_opacity,
                stroke_opacity=base.stroke_opacity,
                # Preserve drawing seqno; sub-index via local offset so they're distinct.
                seqno=base.seqno if i == 0 else base.seqno * 1_000_000 + i,
                layer=base.layer,
                line_cap=base.line_cap,
                line_join=base.line_join,
                level=base.level,
                scissor=base.scissor,
                blendmode=base.blendmode,
                isolated=base.isolated,
                knockout=base.knockout,
                opacity=base.opacity,
            )
        )
    return out