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
from pathlib import Path
from raster_parser.delete_natives.libreoffice_setup.libreoffice_macro import (
    register_libreoffice_macro,
)

lo_config_dir = "~/.config/libreoffice/4"
raster_parser_dir = Path(__file__).parent.parent
register_libreoffice_macro(
    lo_config_dir, raster_parser_dir / "raster_parser/libreoffice_setup/DeleteDrawings.xba"
)


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
