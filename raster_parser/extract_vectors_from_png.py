"""Kept so `from raster_parser.extract_vectors_from_png import ...` keeps working; the code
lives in `raster_parser.png_vectors`."""
from raster_parser.png_vectors.extract import extract_vectors_from_png, get_image_polys

__all__ = ["extract_vectors_from_png", "get_image_polys"]
