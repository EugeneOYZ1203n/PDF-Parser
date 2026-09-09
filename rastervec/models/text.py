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

`from_pymupdf`/`to_pymupdf` are the fitz boundary for *native* text (an
OCR `Text` is hand-built by `OCR/Paddle_OCR/ocr_backend.py` /
`pipelines/sub_pipelines/ocr.py::restore_segment_texts` instead, never via
`from_pymupdf`). `raw_span`, note, is not the bare `get_text("dict")` span
dict -- `direction`/`wmode` live on that span's *line*, one level up, not
on the span itself, so `native_text.py::_extract_spans` merges the line's
`dir`/`wmode` into the stored span dict so `raw_span` alone is enough to
reconstruct a `Text` (and so `from_pymupdf`/`to_pymupdf` round-trip
cleanly without needing the enclosing line as a separate argument).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees

from rastervec.helpers.geometry import compute_origin, make_oriented_quad


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

    @staticmethod
    def from_pymupdf(
        raw_word: tuple, raw_span: dict | None = None, *, page_index: int = 0, seqno: int = 0,
    ) -> "Text":
        """Build a `Text` from one `get_text("words")` row plus its
        best-overlapping `get_text("dict")` span (already merged with its
        line's `dir`/`wmode` -- see this module's docstring), or `None` if
        no span matched closely enough. `page_index`/`seqno` are pipeline
        context, not part of either raw pymupdf structure, so they default
        to 0 for standalone/round-trip use."""
        x0, y0, x1, y1, text, block_no, line_no, word_no = raw_word
        bbox = (x0, y0, x1, y1)

        if raw_span is None:
            direction = (1.0, 0.0)
            return Text(
                text=text, bbox=bbox, direction=direction, origin=compute_origin(bbox, direction),
                font="", font_size=0.0, color=None, flags=0,
                ascender=None, descender=None, wmode=0,
                block_no=block_no, line_no=line_no, word_no=word_no,
                page_index=page_index, seqno=seqno, source="native",
                raw_word=raw_word, raw_span=None,
            )

        direction = tuple(raw_span.get("dir", (1.0, 0.0)))
        return Text(
            text=text, bbox=bbox, direction=direction, origin=compute_origin(bbox, direction),
            font=raw_span.get("font", ""), font_size=raw_span.get("size", 0.0),
            color=raw_span.get("color"), flags=raw_span.get("flags", 0),
            ascender=raw_span.get("ascender"), descender=raw_span.get("descender"),
            wmode=raw_span.get("wmode", 0),
            block_no=block_no, line_no=line_no, word_no=word_no,
            page_index=page_index, seqno=seqno, source="native",
            raw_word=raw_word, raw_span=raw_span,
        )

    def to_pymupdf(self) -> tuple[tuple, dict | None]:
        """The inverse of `from_pymupdf`: the verbatim `(raw_word,
        raw_span)` pair this `Text` was built from (or would have been,
        for a non-native `Text` that happens to carry them)."""
        return (self.raw_word, self.raw_span)
