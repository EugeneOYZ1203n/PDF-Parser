"""Shared shape-drawing helpers for the renderer package.

`replay_drawing_paths` is the accuracy-critical bit: PyMuPDF's
`get_drawings()` returns a filled glyph outline (an "o", "e", "8", "A", ...)
as *one* drawing whose `items` list holds the outer contour and the inner
counter, meant to be filled as a single even-odd path. Drawing each
`VectorPath` primitive on its own and calling `Shape.finish(closePath=True)`
per primitive (the pre-split behaviour) fills every contour solid, so the
counter disappears -- a direct hit to OCR of vector text, which is this
project's whole point. Instead we regroup a cluster's paths by their parent
drawing (`VectorPath.seq`), replay every item of a drawing into the shape,
then call `finish()` **once** for that drawing with its real `even_odd` /
`line_join` / `line_cap` / opacity -- ported from
`archive/raster_parser/rendering/pdf_render/reconstruct.py`
(`_replay_items` + `_finish_kwargs_reconstruct`).
"""
from __future__ import annotations

from itertools import groupby

import pymupdf as fitz

from rastervec.models import VectorPath

_DEFAULT_PATH_COLOR = "#111827"


def path_color_hex(path: VectorPath, default: str = _DEFAULT_PATH_COLOR) -> str:
    """A path's own stroke/fill color as a hex string -- callers should
    render the PDF's real color; any B/W-style simplification (e.g.
    Vector's background-fill heuristic) is purely an internal
    classification concern, never something substituted in its place for
    display."""
    color = path.stroke_color if path.stroke_color is not None else path.fill_color
    if color is None:
        return default
    return "#%02x%02x%02x" % tuple(min(255, max(0, round(c * 255))) for c in color)


def _replay_item(shape: "fitz.Shape", path: VectorPath, dx: float, dy: float) -> None:
    pts = [(x + dx, y + dy) for x, y in path.points]
    if path.kind == "l":
        shape.draw_line(pts[0], pts[1])
    elif path.kind == "re":
        shape.draw_rect(fitz.Rect(*pts[0], *pts[1]))
    elif path.kind == "qu":
        # path.points is stored in cyclic box order (ul, ur, lr, ll) --
        # fitz.Quad's own constructor instead expects (ul, ur, ll, lr), so
        # passing pts straight through swaps the last two corners and draws
        # a crossed "hourglass" instead of a box.
        shape.draw_quad(fitz.Quad(pts[0], pts[1], pts[3], pts[2]))
    elif path.kind == "c":
        shape.draw_bezier(pts[0], pts[1], pts[2], pts[3])


def replay_drawing_paths(
    shape: "fitz.Shape",
    paths: list[VectorPath],
    *,
    dx: float = 0.0,
    dy: float = 0.0,
) -> None:
    """Replay `paths` onto `shape`, grouped by their parent drawing
    (`VectorPath.seq`): every item of a drawing is drawn (in `item_index`
    order), then a single `shape.finish()` for that drawing carrying its
    real fill/stroke/width/dashes/closePath plus `even_odd`, `line_join`,
    `line_cap` and stroke/fill opacity. Points are offset by `(dx, dy)`
    (used to translate a cluster into its own isolated canvas).

    `finish()` is called once per drawing but `commit()` is left to the
    caller -- multiple `finish()` calls before one `commit()` is the
    documented `Shape` pattern, and committing per drawing would mean
    thousands of commits on a dense page.

    A drawing whose paths carry neither `stroke_color` nor `fill_color` is
    skipped outright: `Shape.finish()` emits a stroke operator whenever
    `fill` is `None` even with `color=None`, falling back to the default
    black graphics-state color instead of staying invisible.
    """
    ordered = sorted(paths, key=lambda p: (p.seq, p.item_index))
    for _seq, group_iter in groupby(ordered, key=lambda p: p.seq):
        group = list(group_iter)
        head = group[0]
        if head.stroke_color is None and head.fill_color is None:
            continue

        drawn = False
        for path in group:
            _replay_item(shape, path, dx, dy)
            drawn = drawn or path.kind in ("l", "re", "qu", "c")
        if not drawn:
            continue

        kwargs: dict = {
            "width": head.stroke_width or 0,
            "closePath": True if head.closed is None else bool(head.closed),
            "even_odd": bool(head.even_odd),
            "lineJoin": head.line_join or 0,
            "lineCap": head.line_cap or 0,
        }
        if head.stroke_color is not None:
            kwargs["color"] = head.stroke_color
        if head.fill_color is not None:
            kwargs["fill"] = head.fill_color
        if head.dashes:
            kwargs["dashes"] = head.dashes
        if head.stroke_opacity is not None:
            kwargs["stroke_opacity"] = head.stroke_opacity
        if head.fill_opacity is not None:
            kwargs["fill_opacity"] = head.fill_opacity
        shape.finish(**kwargs)
