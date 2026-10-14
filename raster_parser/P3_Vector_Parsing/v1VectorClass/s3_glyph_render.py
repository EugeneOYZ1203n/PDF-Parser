"""Step 3 -- render one word group's vectors, alone, to a PIL image for OCR.

Copied from `rendering/png_render/png_renderer.py` (`PngRenderer`) and the two helpers it
used from `rendering/pdf_render/reconstruct.py` (`drawing_from_vector`,
`_finish_kwargs_reconstruct`).
"""
from __future__ import annotations

from typing import List

import pymupdf as pymupdf
from PIL import Image

from raster_parser.commons.helpers.fitz_geometry import fitz_item
from raster_parser.commons.models import Vector

MIN_RENDER_PX = 64


def _finish_kwargs_reconstruct(path: dict) -> dict:
    kwargs: dict = {
        "width": path.get("width") or 0,
        "even_odd": path.get("even_odd", False),
        "closePath": path.get("closePath", False),
        "lineJoin": path.get("lineJoin", 0),
        "stroke_opacity": path.get("stroke_opacity", 1),
        "fill_opacity": path.get("fill_opacity", 1),
    }
    fill = path.get("fill")
    if fill is not None:
        kwargs["fill"] = fill
    color = path.get("color")
    if color is not None:
        kwargs["color"] = color
    dashes = path.get("dashes")
    if dashes:
        kwargs["dashes"] = dashes
    line_cap = path.get("lineCap")
    if line_cap is not None:
        kwargs["lineCap"] = line_cap[0] if isinstance(line_cap, (list, tuple)) else line_cap
    return kwargs


def drawing_from_vector(v: Vector) -> dict:
    """A `get_drawings()`-shaped dict for replaying `v`; unset fields get the defaults the
    original raster_parser `Vector` gave them (width 0, dashes "", opacities 1, closePath False)."""
    path = {
        "type": v.type,
        "items": [fitz_item(item) for item in v.items],
        "fill": v.fill,
        "color": v.color,
        "dashes": v.dashes or "",
        "even_odd": bool(v.even_odd),
        "closePath": bool(v.closePath),
        "width": float(v.width) if v.width is not None else 0.0,
        "stroke_opacity": float(v.stroke_opacity) if v.stroke_opacity is not None else 1.0,
        "fill_opacity": float(v.fill_opacity) if v.fill_opacity is not None else 1.0,
        "seqno": v.seqno,
        "layer": v.layer,
        "level": v.level,
    }
    path["lineCap"] = v.lineCap
    path["lineJoin"] = v.lineJoin
    if v.scissor is not None:
        path["scissor"] = v.scissor
    if v.blendmode is not None:
        path["blendmode"] = v.blendmode
    if v.isolated is not None:
        path["isolated"] = v.isolated
    if v.knockout is not None:
        path["knockout"] = v.knockout
    if v.opacity is not None:
        path["opacity"] = v.opacity
    return path


def _pt(p, dx: float, dy: float) -> pymupdf.Point:
    return pymupdf.Point(float(p.x) - dx, float(p.y) - dy)


def _replay_items_offset(shape: pymupdf.Shape, items: list, dx: float, dy: float) -> None:
    for item in items:
        op = item[0]
        if op == "l":
            shape.draw_line(_pt(item[1], dx, dy), _pt(item[2], dx, dy))
        elif op == "re":
            r = pymupdf.Rect(item[1])
            shape.draw_rect(
                pymupdf.Rect(r.x0 - dx, r.y0 - dy, r.x1 - dx, r.y1 - dy)
            )
        elif op == "qu":
            q = item[1]
            shape.draw_quad(
                pymupdf.Quad(
                    _pt(q.ul, dx, dy),
                    _pt(q.ur, dx, dy),
                    _pt(q.ll, dx, dy),
                    _pt(q.lr, dx, dy),
                )
            )
        elif op == "c":
            shape.draw_bezier(
                _pt(item[1], dx, dy),
                _pt(item[2], dx, dy),
                _pt(item[3], dx, dy),
                _pt(item[4], dx, dy),
            )
        else:
            raise ValueError(f"unhandled drawing op {op!r}: {item!r}")


class PngRenderer:
    """Rasterise a bounding box of vectors into a PIL image."""

    def __init__(
        self,
        *,
        dpi: int = 300,
        padding: float = 5.0,
        page: pymupdf.Page,
        min_render_px: int = MIN_RENDER_PX,
    ):
        self.dpi = dpi
        self.padding = padding
        self.page = page
        self.min_render_px = min_render_px

    def render_word_group(self, vectors: List[Vector]) -> Image.Image:
        if not vectors:
            return Image.new("RGB", (1, 1), (255, 255, 255))

        x0 = min(v.rect[0] for v in vectors)
        y0 = min(v.rect[1] for v in vectors)
        x1 = max(v.rect[2] for v in vectors)
        y1 = max(v.rect[3] for v in vectors)
        box = pymupdf.Rect(x0, y0, x1, y1)
        if box.is_empty or box.width < 0.25 or box.height < 0.25:
            return Image.new("RGB", (1, 1), (255, 255, 255))

        doc = pymupdf.open()
        try:
            page = doc.new_page(
                width=box.width + self.padding * 2,
                height=box.height + self.padding * 2,
            )
            if self.page.rotation != 0:
                page.set_rotation(self.page.rotation)
            shape = page.new_shape()
            for vec in vectors:
                path = drawing_from_vector(vec)
                _replay_items_offset(
                    shape, path["items"], box.x0 - self.padding, box.y0 - self.padding
                )
                shape.finish(**_finish_kwargs_reconstruct(path))
                shape.commit()
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(self.dpi / 72, self.dpi / 72),
                alpha=False,
            )
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            w, h = img.size
            s = max(self.min_render_px / w, self.min_render_px / h, 1.0)
            if s > 1.0:
                img = img.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)
            return img
        finally:
            doc.close()
