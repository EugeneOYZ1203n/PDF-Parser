"""Serves both native-extracted words and OCR-recognized segments.

Orientation is stored purely as `direction` (a unit vector) -- `angle()` and
`quad()` are derived on demand (`quad()` reuses `helpers.geometry.make_oriented_quad`,
which builds an oriented quad from an axis-aligned bbox + direction). There is
no `rotation_used` field: an OCR result's final orientation is folded into
`direction` once, before the `Text` is built. This model has no dependency
on any one backend's implementation.

`origin` is a baseline leading-edge point for both native and OCR text
(`helpers.geometry.compute_origin`), so the field means the same thing
regardless of `source`. Native text passes the matched span's real
`get_text("dict")` `origin` as the baseline sample; OCR text has none, so
`compute_origin` falls back to the bbox's normal-axis centre there.

`orientation_source` records where `direction` came from: `"span"` (a real
line `dir`), `"fallback"` (no span matched -> horizontal), or `"ocr"`.

`from_pymupdf`/`to_pymupdf` are the pymupdf boundary for *native* text (an
OCR `Text` is hand-built by the active vector_classify backend's own OCR-recognition
code instead, never via `from_pymupdf`). `raw_span`, note, is not the bare
`get_text("dict")` span
dict -- `direction`/`wmode` live on that span's *line*, one level up, not
on the span itself, so `native_text.py::_extract_spans` merges the line's
`dir`/`wmode` into the stored span dict so `raw_span` alone is enough to
reconstruct a `Text` (and so `from_pymupdf`/`to_pymupdf` round-trip
cleanly without needing the enclosing line as a separate argument).
"""

from __future__ import annotations

from math import atan2, degrees

from pydantic import BaseModel

from raster_parser.models.source import Source
from raster_parser.commons.helpers.geometry import compute_origin, make_oriented_quad
from raster_parser.models._convert import lists_to_tuples


class Text(BaseModel):
    text: str
    bbox: tuple[float, float, float, float]  # axis-aligned extent, local/page frame
    direction: tuple[float, float]  # unit vector; sole source of truth for orientation
    origin: tuple[float, float]  # baseline leading-edge point
    font: str
    font_size: float
    color: int | None  # sRGB packed as 0xRRGGBB (get_text("dict") span "color")
    flags: int  # span font flags: 1 superscript, 2 italic, 4 serif, 8 monospaced, 16 bold
    ascender: float | None
    descender: float | None
    wmode: int  # writing mode: 0 horizontal, 1 vertical
    block_no: int
    line_no: int
    word_no: int
    page_index: int
    seqno: int
    confidence: float = 0.0
    source: Source = Source.NATIVE_TEXT  # where it came from (`models/source.py`)
    # Where `direction` came from: "span" -- a real get_text("dict") line
    # `dir`; "fallback" -- no span matched, defaulted to horizontal (1, 0);
    # "ocr" -- the OCR step's own orientation, folded in before the Text was
    # built. Lets downstream/debug tell a genuine horizontal word from one
    # that silently lost its rotation.
    orientation_source: str = "span"

    # Forward-compat / literally-every-get_text()-field escape hatch: the
    # verbatim source tuple/dict, alongside the normalized fields above that
    # the pipeline actually derives from them.
    raw_word: tuple | None = None  # verbatim get_text("words") tuple
    raw_span: dict | None = None  # verbatim matched get_text("dict") span
    # The text detector's own quad, page space (4 points) -- PaddleOCR's DB
    # detector returns rotated `cv2.minAreaRect` quads, which `bbox` (their
    # axis-aligned envelope) can't represent. `None` for native text and
    # anything with no detect quad; a renderer then recovers the oriented box
    # from `bbox` + `direction`.
    quad_points: tuple[tuple[float, float], ...] | None = None

    def angle(self) -> float:
        dx, dy = self.direction
        return degrees(atan2(dy, dx))

    def quad(self) -> tuple[tuple[float, float], ...]:
        return make_oriented_quad(self.bbox, *self.direction)

    def to_dict(self) -> dict:
        """JSON-safe dict of every field (tuples become lists); `from_dict` restores it."""
        return self.model_dump()

    @staticmethod
    def from_dict(d: dict) -> "Text":
        """Inverse of `to_dict`: tuples restored; unknown keys ignored."""
        d = dict(d)
        for key in ("bbox", "direction", "origin", "raw_word"):
            if d.get(key) is not None:
                d[key] = tuple(d[key])
        if d.get("quad_points") is not None:
            d["quad_points"] = tuple(tuple(pt) for pt in d["quad_points"])
        if d.get("raw_span") is not None:
            d["raw_span"] = lists_to_tuples(d["raw_span"])
        if d.get("source") is not None:
            d["source"] = Source(d["source"])
        return Text.model_validate(d)

    @staticmethod
    def from_pymupdf(
        raw_word: tuple,
        raw_span: dict | None = None,
        *,
        page_index: int = 0,
        seqno: int = 0,
    ) -> "Text":
        """Build a `Text` from one `get_text("words")` row plus its
        best-overlapping `get_text("dict")` span (already merged with its
        line's `dir`/`wmode` -- see this module's docstring), or `raw_span=None`
        if no span matched closely enough (the word then reads as horizontal
        with `orientation_source="fallback"`). `page_index`/`seqno` are pipeline
        context, not part of either raw pymupdf structure, so they default
        to 0 for standalone/round-trip use."""
        x0, y0, x1, y1, text, block_no, line_no, word_no = raw_word
        bbox = (x0, y0, x1, y1)

        if raw_span is None:
            direction = (1.0, 0.0)
            return Text(
                text=text,
                bbox=bbox,
                direction=direction,
                origin=compute_origin(bbox, direction),
                font="",
                font_size=0.0,
                color=None,
                flags=0,
                ascender=None,
                descender=None,
                wmode=0,
                block_no=block_no,
                line_no=line_no,
                word_no=word_no,
                page_index=page_index,
                seqno=seqno,
                source=Source.NATIVE_TEXT,
                orientation_source="fallback",
                raw_word=raw_word,
                raw_span=None,
            )

        direction = tuple(raw_span.get("dir", (1.0, 0.0)))
        # `get_text("dict")`'s span-level `origin` is the real baseline start
        # of the whole span -- keep its perpendicular (normal) offset so the
        # per-word origin lands on the actual glyph baseline, not the bbox
        # centre `compute_origin` falls back to (correct for rotated/vertical
        # text too, since the offset is projected along the direction normal).
        span_origin = raw_span.get("origin")
        baseline = tuple(span_origin) if span_origin is not None else None
        return Text(
            text=text,
            bbox=bbox,
            direction=direction,
            origin=compute_origin(bbox, direction, baseline),
            font=raw_span.get("font", ""),
            font_size=raw_span.get("size", 0.0),
            color=raw_span.get("color"),
            flags=raw_span.get("flags", 0),
            ascender=raw_span.get("ascender"),
            descender=raw_span.get("descender"),
            wmode=raw_span.get("wmode", 0),
            block_no=block_no,
            line_no=line_no,
            word_no=word_no,
            page_index=page_index,
            seqno=seqno,
            source=Source.NATIVE_TEXT,
            orientation_source="span",
            raw_word=raw_word,
            raw_span=raw_span,
        )

    def to_pymupdf(self) -> tuple[tuple, dict | None]:
        """The inverse of `from_pymupdf`: the verbatim `(raw_word,
        raw_span)` pair this `Text` was built from (or would have been,
        for a non-native `Text` that happens to carry them)."""
        return (self.raw_word, self.raw_span)
