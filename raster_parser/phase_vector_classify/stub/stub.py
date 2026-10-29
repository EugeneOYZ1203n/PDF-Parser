"""Pass-through vector_classify backend: every native_read vector is drawing, no text is recognised."""

from __future__ import annotations

from raster_parser.models.page import Page
from raster_parser.models.text import Text
from raster_parser.models.vector import Vector
from raster_parser.models.settings import RunSettings


def parse(
    vectors: list[Vector], page: Page, settings: RunSettings
) -> tuple[list[Vector], list[Text]]:
    return list(vectors), []
