from pathlib import Path

from raster_parser.helpers.geometry import PDF_POINTS_PER_INCH
from raster_parser.models.Vector_Class import Vector
import pymupdf
from raster_parser.transforms.scale_pdf_vector_items import _scale_vector
from raster_parser.vectors.vector_items_processing import _vectors_from_drawing


def pdf_to_vectors(
    pdf_path: Path,
    *,
    page_index: int = 0,
    dpi: int | None = None,
    split_strokes: bool = True,
) -> list[Vector]:
    """Read ``get_drawings(extended=True)`` from a PDF and build ``Vector`` objects (Option A).

    ``split_strokes=True``: each continuous stroke becomes its own Vector (autotrace's
    multi-``M`` path is split). Use ``False`` when the paths are *filled* shapes whose
    subpaths are winding/even-odd holes — splitting those and filling each subpath
    independently would turn outlined regions into solid blobs.
    If ``dpi`` is set, coordinates are scaled by ``72/dpi`` so they match a page that was
    rendered at that DPI (autotrace's SVG/PDF is typically 1 SVG unit = 1 pixel).
    """
    doc = pymupdf.open(str(pdf_path))
    try:
        page = doc[page_index]
        raw = page.get_drawings(extended=True)
    finally:
        doc.close()
    vectors: list[Vector] = []
    for d in raw:
        if split_strokes:
            vectors.extend(_vectors_from_drawing(d))
        else:
            vectors.append(Vector.from_drawing(d))
    if dpi is not None and dpi > 0:
        scale = PDF_POINTS_PER_INCH / float(dpi)
        vectors = [_scale_vector(v, scale) for v in vectors]
    return vectors
