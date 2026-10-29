"""Code shared only by the v1 backends -- the original raster_parser behaviour they inherited,
kept in one place.

- `autotrace`: PNG -> SVG -> PDF -> `get_drawings()` -> shared `Vector`s.
- `render`: replay `Vector`s onto a PyMuPDF page the way the original reconstructor did.
- `ocr`: crop normalisation for PaddleOCR and the OCR-word -> `Text` conversion.

Backends still never import each other; this package sits under `commons` for that reason.
"""
