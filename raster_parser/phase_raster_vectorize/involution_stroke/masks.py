"""The model's input: one color layer as a binary gray mask.

Kept apart from `adapter.py` (which imports the OCR modules), as in the
other stroke backends (`commons.raster_text.strokes.layer_image`)."""

from __future__ import annotations
from raster_parser.commons.raster_text.strokes import (  # noqa: F401 -- moved to commons; still reachable from this module
    layer_image,
)
