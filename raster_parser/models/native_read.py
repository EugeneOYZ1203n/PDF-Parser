"""Types the native_read phase produces: its result bundle and the flattened text spans it matches words to."""

from __future__ import annotations

from dataclasses import dataclass

import pymupdf
from raster_parser.models.image import Image
from raster_parser.models.page import Page
from raster_parser.models.text import Text
from raster_parser.models.vector import Vector


@dataclass
class NativeReadResult:
    page: Page
    texts: list[Text]
    images: list[Image]
    vectors: list[Vector]


@dataclass
class NativeSpan:
    """One `get_text("dict")` span, flattened."""

    bbox: "pymupdf.Rect"
    text: str
    font: str
    font_size: float
    flags: int
    color: int | None
    origin: tuple[float, float] | None
    direction: tuple[float, float]
    ascender: float | None
    descender: float | None
    wmode: int
    raw: dict
