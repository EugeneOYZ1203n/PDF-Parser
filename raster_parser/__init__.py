from raster_parser.main_pipeline_extract import extract
from raster_parser.commons.models import ExtractResult, Text, Vector
from raster_parser.P2_Raster_To_Vec.v1RasterToVec.s1_clear_page import delete_drawings_from_pdf

# FIXME find a way to not modify path
import sys
sys.path.append(".")
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
