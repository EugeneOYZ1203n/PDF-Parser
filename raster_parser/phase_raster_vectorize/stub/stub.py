"""Do-nothing raster_vectorize backend: no raster pass. What `extract(enable_raster_pass=False)` runs."""

from __future__ import annotations

from raster_parser.models.image import Image
from raster_parser.models.page import Page
from raster_parser.models.text import Text
from raster_parser.models.vector import Vector
from raster_parser.models.settings import RunSettings


def extract(
    images: list[Image], page: Page, settings: RunSettings
) -> tuple[list[Vector], list[Text]]:
    return [], []
