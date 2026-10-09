from __future__ import annotations

import pymupdf as pymupdf
from raster_parser.models import Vector


def _scale_point(p: pymupdf.Point, scale: float) -> pymupdf.Point:
    return pymupdf.Point(p.x * scale, p.y * scale)


def _scale_item(item: tuple, scale: float) -> tuple:
    """Scale one get_drawings() path segment into PDF-point space."""
    op = item[0]
    if op == "l":
        return ("l", _scale_point(item[1], scale), _scale_point(item[2], scale))
    if op == "c":
        return (
            "c",
            _scale_point(item[1], scale),
            _scale_point(item[2], scale),
            _scale_point(item[3], scale),
            _scale_point(item[4], scale),
        )
    if op == "re":
        r = item[1]
        return ("re", pymupdf.Rect(r.x0 * scale, r.y0 * scale, r.x1 * scale, r.y1 * scale))
    if op == "qu":
        q = item[1]
        return (
            "qu",
            pymupdf.Quad(
                _scale_point(q.ul, scale),
                _scale_point(q.ur, scale),
                _scale_point(q.ll, scale),
                _scale_point(q.lr, scale),
            ),
        )
    return item


def _scale_vector(v: Vector, scale: float) -> Vector:
    """Map a Vector from SVG-PDF pixel-as-point space into PDF points (``72/dpi``)."""
    if scale == 1.0:
        return v
    x0, y0, x1, y1 = v.rect
    scissor = None
    if v.scissor is not None:
        sx0, sy0, sx1, sy1 = v.scissor
        scissor = (sx0 * scale, sy0 * scale, sx1 * scale, sy1 * scale)
    return Vector(
        items=[_scale_item(it, scale) for it in v.items],
        type=v.type,
        rect=(x0 * scale, y0 * scale, x1 * scale, y1 * scale),
        fill=v.fill,
        color=v.color,
        width=v.width * scale,
        even_odd=v.even_odd,
        close_path=v.close_path,
        dashes=v.dashes,
        fill_opacity=v.fill_opacity,
        stroke_opacity=v.stroke_opacity,
        seqno=v.seqno,
        layer=v.layer,
        line_cap=v.line_cap,
        line_join=v.line_join,
        level=v.level,
        scissor=scissor,
        blendmode=v.blendmode,
        isolated=v.isolated,
        knockout=v.knockout,
        opacity=v.opacity,
    )
