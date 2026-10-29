"""Render vector word groups to PIL images."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pymupdf as fitz
from PIL import Image

from raster_parser.models import Vector
from raster_parser.rendering.pdf_render.reconstruct import (
    _finish_kwargs_reconstruct,
    drawing_from_vector,
)

MIN_RENDER_PX = 64


def save_rgb(arr: np.ndarray, path: Path) -> None:
    """Write a grayscale or RGB numpy array as an image file."""
    if arr.ndim == 2:
        img = Image.fromarray(arr)
    else:
        img = Image.fromarray(arr.astype(np.uint8))
    img.save(path)


def _pt(p, dx: float, dy: float) -> fitz.Point:
    return fitz.Point(float(p.x) - dx, float(p.y) - dy)


def _replay_items_offset(shape: fitz.Shape, items: list, dx: float, dy: float) -> None:
    for item in items:
        op = item[0]
        if op == "l":
            shape.draw_line(_pt(item[1], dx, dy), _pt(item[2], dx, dy))
        elif op == "re":
            r = fitz.Rect(item[1])
            shape.draw_rect(
                fitz.Rect(r.x0 - dx, r.y0 - dy, r.x1 - dx, r.y1 - dy)
            )
        elif op == "qu":
            q = item[1]
            shape.draw_quad(
                fitz.Quad(
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
        page: fitz.Page,
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
        box = fitz.Rect(x0, y0, x1, y1)
        if box.is_empty or box.width < 0.25 or box.height < 0.25:
            return Image.new("RGB", (1, 1), (255, 255, 255))

        doc = fitz.open()
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
                matrix=fitz.Matrix(self.dpi / 72, self.dpi / 72),
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
