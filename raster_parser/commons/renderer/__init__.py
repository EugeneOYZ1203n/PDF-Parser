"""Rendering helpers shared across phases.

- `draw.py` -- the one standard drawing layer: a spec builder + a drawer per primitive kind
  (text, bbox, quad, polyline, arrow, dot, vector, raster image) and `render_specs_pdf`. Every
  debug-layer PDF and the final reconstructed page are built from these.
- `pdf.py` -- one-layer debug-PDF wrappers over `draw.py` (`render_boxes_pdf`,
  `render_quads_pdf`, `render_text_pdf`, `render_vectors_pdf`) plus `rasterize_pdf`. The final
  reconstructed page is Phase 4's (`P4_Output_Organization.render_output_pdf`).
- `png.py` -- `render_vector_cluster`: a cluster of vectors alone, black-and-white, as an OCR
  input image. `ocr_prep.py` -- `render_cluster_with_dynamic_dpi`: the same at a DPI raised so
  small clusters stay readable.
- `_shapes.py` -- `replay_drawing_paths` (per-drawing path replay with the real fill rule,
  clips and blend/opacity) and `path_color_hex`.
"""
