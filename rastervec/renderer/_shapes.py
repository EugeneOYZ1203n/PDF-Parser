"""Shared shape-drawing helpers for the renderer package.

`replay_drawing_paths` is the accuracy-critical bit: PyMuPDF's
`get_drawings()` returns a filled glyph outline (an "o", "e", "8", "A", ...)
as *one* drawing whose `items` list holds the outer contour and the inner
counter, meant to be filled as a single even-odd path. Drawing each item
primitive on its own and calling `Shape.finish(closePath=True)` per
primitive fills every contour solid, so the counter disappears -- a direct
hit to OCR of vector text, which is this project's whole point. Since a
`Vector` is never decomposed (its `items` stay nested exactly as
`get_drawings()` returned them), this just replays every item of one
`Vector` into the shape, then calls `finish()` **once** per `Vector` with
its real `even_odd` / `line_join` / `line_cap` / opacity -- ported from
`archive/raster_parser/rendering/pdf_render/reconstruct.py`
(`_replay_items` + `_finish_kwargs_reconstruct`).
"""
from __future__ import annotations

import pymupdf as fitz

from rastervec.helpers.geometry import item_points
from rastervec.models import Vector

_DEFAULT_PATH_COLOR = "#111827"


def path_color_hex(vector: Vector, default: str = _DEFAULT_PATH_COLOR) -> str:
    """A Vector's own stroke/fill color as a hex string -- callers should
    render the PDF's real color; any B/W-style simplification is purely an
    internal classification concern, never something substituted in its
    place for display."""
    color = vector.color if vector.color is not None else vector.fill
    if color is None:
        return default
    return "#%02x%02x%02x" % tuple(min(255, max(0, round(c * 255))) for c in color)


def _replay_item(shape: "fitz.Shape", item: tuple, dx: float, dy: float) -> bool:
    kind = item[0]
    pts = [(x + dx, y + dy) for x, y in item_points(item)]
    if kind == "l":
        shape.draw_line(pts[0], pts[1])
    elif kind == "re":
        shape.draw_rect(fitz.Rect(*pts[0], *pts[1]))
    elif kind == "qu":
        # item points are stored in cyclic box order (ul, ur, lr, ll) --
        # fitz.Quad's own constructor instead expects (ul, ur, ll, lr), so
        # passing pts straight through swaps the last two corners and draws
        # a crossed "hourglass" instead of a box.
        shape.draw_quad(fitz.Quad(pts[0], pts[1], pts[3], pts[2]))
    elif kind == "c":
        shape.draw_bezier(pts[0], pts[1], pts[2], pts[3])
    else:
        return False
    return True


def replay_drawing_paths(
    shape: "fitz.Shape",
    vectors: list[Vector],
    *,
    dx: float = 0.0,
    dy: float = 0.0,
) -> None:
    """Replay `vectors` onto `shape`: every item of a `Vector` is drawn (in
    its own stored order), then a single `shape.finish()` for that `Vector`
    carrying its real fill/stroke/width/dashes/closePath plus `even_odd`,
    `line_join`, `line_cap` and stroke/fill opacity. Points are offset by
    `(dx, dy)` (used to translate a cluster into its own isolated canvas).

    `finish()` is called once per `Vector` but `commit()` is left to the
    caller -- multiple `finish()` calls before one `commit()` is the
    documented `Shape` pattern, and committing per drawing would mean
    thousands of commits on a dense page.

    A `Vector` carrying neither `color` nor `fill` is skipped outright:
    `Shape.finish()` emits a stroke operator whenever `fill` is `None` even
    with `color=None`, falling back to the default black graphics-state
    color instead of staying invisible.
    """
    for v in vectors:
        if v.color is None and v.fill is None:
            continue

        drawn = False
        for item in v.items:
            drawn = _replay_item(shape, item, dx, dy) or drawn
        if not drawn:
            continue

        kwargs: dict = {
            "width": v.width or 0,
            "closePath": True if v.closePath is None else bool(v.closePath),
            "even_odd": bool(v.even_odd),
            "lineJoin": v.lineJoin or 0,
            "lineCap": v.lineCap or 0,
        }
        if v.color is not None:
            kwargs["color"] = v.color
        if v.fill is not None:
            kwargs["fill"] = v.fill
        if v.dashes:
            kwargs["dashes"] = v.dashes
        if v.stroke_opacity is not None:
            kwargs["stroke_opacity"] = v.stroke_opacity
        if v.fill_opacity is not None:
            kwargs["fill_opacity"] = v.fill_opacity
        shape.finish(**kwargs)
