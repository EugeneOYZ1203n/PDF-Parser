"""`Source`: where a `Text` or `Vector` came from."""

from __future__ import annotations

from enum import Enum


class Source(str, Enum):
    """Where a `Text` / `Vector` came from. A `str` subclass: it equals its value and is written to
    JSON as it."""

    NATIVE_TEXT = "Native Text"  # a PDF text object (`get_text`)
    NATIVE_VECTOR = (
        "Native Vector"  # a PDF drawing (`get_drawings`), or text read from drawn glyphs
    )
    FROM_IMAGE = "From Image"  # traced, or read by OCR, from an embedded raster image

    def __str__(self) -> str:
        return self.value
