from raster_parser.main_pipeline_extract import extract
from raster_parser.models import ExtractResult, Text, Vector
from raster_parser.phase_raster_vectorize.v1_raster_to_vec.s1_clear_page import (
    delete_drawings_from_pdf,
)

# The LibreOffice DeleteDrawings macro is registered lazily, on the first
# LibreOffice call (``s1_libreoffice_macro.ensure_macro_registered``), so importing
# the package doesn't need LibreOffice installed.


__all__ = [
    "extract",
    "ExtractResult",
    "Text",
    "Vector",
    "delete_drawings_from_pdf",
]
