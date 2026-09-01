"""Renderer package: turns pipeline data (vector paths, clusters, native
words, OCR results) into pixels. Not a pipeline stage.

Split by output concern:
- `png.py`  -- rasterize vector paths for OCR / FAST detection input
  (`render_vector_cluster`, `render_page_paths`, plus the
  `pixel_to_page_bbox` / `cluster_frame_size` transform helpers).
- `pdf.py`  -- `render_reconstructed_page`, the notebook's reconstruction
  preview.
- `svg.py`  -- `render_page_svg`, a thin `get_svg_image()` wrapper.
- `_shapes.py` -- `replay_drawing_paths` (per-drawing composite path replay
  with the even_odd fill rule, so filled glyph counters render as holes)
  and `path_color_hex`, shared by png/pdf.

Module-level functions, no `Renderer` class -- import what you need
straight from `rastervec.renderer`.
"""
from __future__ import annotations

from rastervec.renderer._shapes import path_color_hex, replay_drawing_paths
from rastervec.renderer.pdf import render_reconstructed_page, render_reconstructed_pdf
from rastervec.renderer.png import (
    cluster_frame_size,
    pixel_to_page_bbox,
    render_page_paths,
    render_vector_cluster,
)
from rastervec.renderer.svg import render_page_svg

__all__ = [
    "path_color_hex",
    "replay_drawing_paths",
    "render_reconstructed_page",
    "render_reconstructed_pdf",
    "render_vector_cluster",
    "render_page_paths",
    "pixel_to_page_bbox",
    "cluster_frame_size",
    "render_page_svg",
]
