"""Serves both native-extracted words and OCR-recognized segments.

Orientation is stored purely as `direction` (a unit vector) -- `angle()` and
`quad()` are derived on demand (`quad()` reuses the existing
`helpers.geometry.make_oriented_quad`, which already builds an oriented quad
from an axis-aligned bbox + direction, so nothing new is invented). There is
no `rotation_used` field: an OCR result's final orientation is Radon's
precise skew angle combined with PaddleOCR's cls-detected 180-degree
correction, folded into `direction` once, before the `Text` is built (see
`OCR/ocr_backend.py::recognize_unique_segments` and
`pipelines/_steps.py::restore_segment_texts`).

`origin` is computed the same way for native and OCR text via
`helpers.geometry.compute_origin(bbox, direction)`, so the field means the
same thing regardless of `source`.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees

from rastervec.helpers.geometry import make_oriented_quad


@dataclass
class Text:
    text: str
    bbox: tuple[float, float, float, float]  # axis-aligned extent, local/page frame
    direction: tuple[float, float]  # unit vector; sole source of truth for orientation
    origin: tuple[float, float]  # baseline leading-edge point
    font: str
    font_size: float
    color: int | None
    flags: int
    ascender: float | None
    descender: float | None
    wmode: int
    block_no: int
    line_no: int
    word_no: int
    page_index: int
    seqno: int
    confidence: float = 0.0
    source: str = "native"  # "native" | "ocr"

    # Forward-compat / literally-every-get_text()-field escape hatch: the
    # verbatim source tuple/dict, alongside the normalized fields above that
    # the pipeline actually derives from them.
    raw_word: tuple | None = None  # verbatim get_text("words") tuple
    raw_span: dict | None = None  # verbatim matched get_text("dict") span

    def angle(self) -> float:
        dx, dy = self.direction
        return degrees(atan2(dy, dx))

    def quad(self) -> tuple[tuple[float, float], ...]:
        return make_oriented_quad(self.bbox, *self.direction)
