"""Shared shape-drawing helpers for the renderer package."""

from __future__ import annotations

import pymupdf
from raster_parser.commons.helpers.geometry import item_points


def replay_item(shape: "pymupdf.Shape", item: tuple, dx: float, dy: float) -> bool:
    """Draw one `Vector.items` entry into `shape`, offset by `(dx, dy)`; False for an
    op it doesn't know."""
    kind = item[0]
    pts = [(x + dx, y + dy) for x, y in item_points(item)]
    if kind == "l":
        shape.draw_line(pts[0], pts[1])
    elif kind == "re":
        shape.draw_rect(pymupdf.Rect(*pts[0], *pts[1]))
    elif kind == "qu":
        # item points are stored in cyclic box order (ul, ur, lr, ll) --
        # pymupdf.Quad's own constructor instead expects (ul, ur, ll, lr), so
        # passing pts straight through swaps the last two corners and draws
        # a crossed "hourglass" instead of a box.
        shape.draw_quad(pymupdf.Quad(pts[0], pts[1], pts[3], pts[2]))
    elif kind == "c":
        shape.draw_bezier(pts[0], pts[1], pts[2], pts[3])
    else:
        return False
    return True
