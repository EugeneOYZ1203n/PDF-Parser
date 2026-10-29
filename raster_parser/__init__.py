from raster_parser.main_pipeline_extract import extract
from raster_parser.serialisation.deserialization import (
    deserialize,
    deserialize_output,
    deserialize_text,
    deserialize_vector,
)
from raster_parser.models import TextDTO, NativePDFElements, VectorDTO
from raster_parser.delete_natives.delete_native import delete_drawings_from_pdf

# FIXME find a way to not modify path
import sys

sys.path.append(".")
# The LibreOffice DeleteDrawings macro is registered lazily, on the first LibreOffice call
# (``libreoffice_macro.ensure_macro_registered``), so importing the package doesn't need
# LibreOffice installed.


__all__ = [
    "extract",
    "TextDTO",
    "NativePDFElements",
    "VectorDTO",
    "deserialize",
    "deserialize_output",
    "deserialize_text",
    "deserialize_vector",
    "delete_drawings_from_pdf",
]
