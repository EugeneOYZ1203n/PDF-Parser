"""Do-nothing Phase 2 backend: no raster pass. What `extract(enable_raster_pass=False)` runs."""
from __future__ import annotations

from raster_parser.commons.models import Image, Page, Text, Vector
from raster_parser.core.settings import RunSettings


def extract(images: list[Image], page: Page, settings: RunSettings) -> tuple[list[Vector], list[Text]]:
    return [], []
