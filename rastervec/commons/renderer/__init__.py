"""Renderer package: turns pipeline data (vector paths, clusters, native
words, OCR results) into pixels. Not a pipeline stage.

Split by output concern:
- `png.py`  -- rasterize vector paths for OCR / FAST detection input
  (`render_vector_cluster`, `render_page_paths`, plus the
  `pixel_to_page_bbox` / `page_points_to_pixel` transform helpers). Renders
  a cluster's bare `union_bbox` -- no padding; that lives in
  `ocr_prep.py::pad_image_uniform`.
- `ocr_prep.py` -- padding (`pad_image_uniform`) and dynamic-dpi rendering
  (`dpi_for_cluster` / `render_cluster_with_dynamic_dpi`, composing
  `png.render_vector_cluster`) shared by the P3 backends' own
  `paddle_engine.py`/`radon.py`. Rotation helpers and BGR normalization
  stay backend-local (only `FastIntoPaddle`/`LegacyRecreation` need them).
- `pdf.py`  -- `render_reconstructed_page`, the notebook's reconstruction
  preview, plus `render_boxes_pdf`, a generic colored-bbox-outline
  primitive used by the benchmark's pred-vs-GT box overlay.
- `svg.py`  -- `render_page_svg`, a thin `get_svg_image()` wrapper.
- `_shapes.py` -- `replay_drawing_paths` (per-drawing composite path replay
  with the even_odd fill rule so filled glyph counters render as holes, plus
  per-`(blendmode, opacity)`-run ExtGState wrapping so blended lines don't
  reconstruct fully opaque) and `path_color_hex`, shared by png/pdf.
- `notebook.py` -- notebook-only display plumbing (`RenderResult`,
  `visualize`, `draw_paths`/`draw_polys`/`draw_bboxes`, ...) for
  `pipeline_stage_visualization.ipynb`. Deliberately **not** re-exported
  here -- it imports matplotlib, and this package is imported by the real
  pipeline itself; import it directly
  (`from rastervec.commons.renderer.notebook import ...`) instead.

Module-level functions, no `Renderer` class -- import what you need
straight from `rastervec.commons.renderer`.
"""
from __future__ import annotations

from rastervec.commons.renderer._shapes import path_color_hex, replay_drawing_paths
from rastervec.commons.renderer.ocr_prep import (
    dpi_for_cluster,
    pad_image_uniform,
    render_cluster_with_dynamic_dpi,
)
from rastervec.commons.renderer.pdf import (
    render_boxes_pdf,
    render_reconstructed_page,
    render_reconstructed_pdf,
    render_text_pdf,
    render_vectors_pdf,
)
from rastervec.commons.renderer.png import (
    page_points_to_pixel,
    pixel_to_page_bbox,
    pixel_to_page_points,
    render_page_paths,
    render_vector_cluster,
)
from rastervec.commons.renderer.svg import render_page_svg

__all__ = [
    "path_color_hex",
    "replay_drawing_paths",
    "dpi_for_cluster",
    "pad_image_uniform",
    "render_cluster_with_dynamic_dpi",
    "render_reconstructed_page",
    "render_reconstructed_pdf",
    "render_boxes_pdf",
    "render_text_pdf",
    "render_vectors_pdf",
    "render_vector_cluster",
    "render_page_paths",
    "pixel_to_page_bbox",
    "pixel_to_page_points",
    "page_points_to_pixel",
    "render_page_svg",
]
