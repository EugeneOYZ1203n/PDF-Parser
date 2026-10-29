"""Rendering helpers shared across phases.

- `draw.py` -- the one standard drawing layer: a spec builder + a drawer per primitive kind
  (text, bbox, quad, polyline, arrow, dot, vector, raster image) and `render_specs_pdf`. Every
  debug-layer PDF and the final reconstructed page are built from these.
- `pdf.py` -- one-layer debug-PDF wrappers over `draw.py` (`render_boxes_pdf`,
  `render_quads_pdf`, `render_text_pdf`, `render_vectors_pdf`) plus `rasterize_pdf`. The final
  reconstructed page is output_assemble's (`phase_output_assemble.render_output_pdf`).
- `_shapes.py` -- `replay_drawing_paths` (per-drawing path replay with the real fill rule,
  clips and blend/opacity), `replay_item` and `path_color_hex`.
"""
