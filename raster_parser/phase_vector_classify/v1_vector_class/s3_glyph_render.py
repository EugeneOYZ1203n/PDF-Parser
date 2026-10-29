"""Step 3 -- render one word group's vectors, alone, to a PIL image for OCR.

Copied from `rendering/png_render/png_renderer.py` (`PngRenderer`); the replay itself is
`commons.v1_commons.render`.
"""

from __future__ import annotations

from typing import List

import pymupdf
from PIL import Image

from raster_parser.commons.helpers.pixmap import zoom_matrix
from raster_parser.models.vector import Vector
from raster_parser.commons.v1_commons.render import finish_kwargs, replay_items

# Renders smaller than this (px) on either side are upscaled to it, so the recogniser gets
# enough pixels per glyph.
MIN_RENDER_PX = 64
# Blank margin (PDF points) left around a word group's bbox when it is rendered.
DEFAULT_PADDING_PT = 5.0
# Word groups thinner than this (PDF points) on either side render as a blank pixel.
MIN_GROUP_SIDE_PT = 0.25


class PngRenderer:
    """Rasterise a bounding box of vectors into a PIL image."""

    def __init__(
        self,
        *,
        dpi: int = 300,
        padding: float = DEFAULT_PADDING_PT,
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
        if box.is_empty or box.width < MIN_GROUP_SIDE_PT or box.height < MIN_GROUP_SIDE_PT:
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
            dx, dy = box.x0 - self.padding, box.y0 - self.padding
            for vec in vectors:
                replay_items(shape, vec.items, dx, dy)
                shape.finish(**finish_kwargs(vec))
                shape.commit()
            pix = page.get_pixmap(
                matrix=zoom_matrix(self.dpi),
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
