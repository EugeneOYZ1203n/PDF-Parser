"""Replay shared `Vector`s onto a PyMuPDF page the way the original raster_parser reconstructor did
(`rendering/pdf_render/reconstruct.py`): every item drawn, then one `Shape.finish()` per vector.

Unlike `commons.renderer._shapes.replay_drawing_paths` there is no hairline fix-up, blend-mode
wrapping or clipping, and unset fields take the old defaults (width 0, opacity 1, closePath
False). The v1 steps depend on these exact pixels: `v1_vector_class` OCRs them.
"""

from __future__ import annotations

import pymupdf

from raster_parser.models.vector import Vector
from raster_parser.commons.renderer._shapes import replay_item


def finish_kwargs(v: Vector) -> dict:
    """`Shape.finish()` arguments for `v` (see `models.vector.Vector` for what each field means)."""
    kwargs: dict = {
        "width": v.width or 0,
        "even_odd": bool(v.even_odd),
        "closePath": bool(v.closePath),
        "lineJoin": v.lineJoin,
        "stroke_opacity": v.stroke_opacity if v.stroke_opacity is not None else 1.0,
        "fill_opacity": v.fill_opacity if v.fill_opacity is not None else 1.0,
        "lineCap": v.line_cap,
    }
    if v.fill is not None:
        kwargs["fill"] = v.fill
    if v.color is not None:
        kwargs["color"] = v.color
    if v.dashes:
        kwargs["dashes"] = v.dashes
    return kwargs


def replay_items(shape: pymupdf.Shape, items: list, dx: float = 0.0, dy: float = 0.0) -> None:
    """Draw plain `Vector.items` into `shape`, every point moved by `(-dx, -dy)`."""
    for item in items:
        if not replay_item(shape, item, -dx, -dy):
            raise ValueError(f"unhandled drawing op {item[0]!r}: {item!r}")
