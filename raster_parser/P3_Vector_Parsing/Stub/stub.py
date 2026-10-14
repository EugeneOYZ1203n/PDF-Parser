"""Pass-through Phase 3 backend: every Phase 1 vector is drawing, no text is recognised."""
from __future__ import annotations

from raster_parser.commons.models import Page, Text, Vector
from raster_parser.core.settings import RunSettings


def parse(vectors: list[Vector], page: Page, settings: RunSettings) -> tuple[list[Vector], list[Text]]:
    return list(vectors), []
