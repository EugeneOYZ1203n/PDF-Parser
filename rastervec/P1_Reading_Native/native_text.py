"""Native stage: extracts words with position, rotation, font and font-size
directly from the PDF's own text objects (no OCR/rendering involved).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

import pymupdf as fitz

from rastervec.logging_setup import get_logger
from rastervec.models import Page, Text
from rastervec.renderer.stages import render_native  # noqa: F401 -- re-exported for callers

_LOG = get_logger("native")

# A word overlapping a span by less than this fraction of the word's own
# area is treated as unmatched (garbage geometry / stray span) rather than
# inheriting that span's font metadata.
_MIN_SPAN_OVERLAP = 0.10


@dataclass
class _Span:
    """One `get_text("dict")` span, flattened."""

    bbox: "fitz.Rect"
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


def extract_native_text(page: Page) -> list[Text]:
    """One `Text` (source="native") per `get_text("words")` word, in reading
    order, with font/rotation metadata joined from the best-overlapping
    `get_text("dict")` span."""
    fitz_page = page.fitz_page
    page_index = page.meta.index

    spans = _extract_spans(fitz_page)
    words = _extract_words(fitz_page)
    _LOG.debug(
        "page %d: %d word(s), %d span(s)", page_index, len(words), len(spans),
    )

    result: list[Text] = []
    for seq, raw_word in enumerate(words):
        x0, y0, x1, y1, text, _block_no, _line_no, _word_no = raw_word
        bbox = fitz.Rect(x0, y0, x1, y1)
        span = _match_word_to_span(bbox, spans)
        if span is None:
            _LOG.warning(
                "page %d: no matching span for word %r at %s",
                page_index, text, bbox,
            )
        result.append(_to_word(page_index, seq, raw_word, span))
    return result


def _extract_spans(fitz_page: "fitz.Page") -> list[_Span]:
    """Flatten page.get_text('dict') into a list of `_Span`s."""
    spans: list[_Span] = []
    text_dict = fitz_page.get_text("dict", sort=False)

    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:  # 0 = text block (1 = image)
            continue
        for line in block.get("lines", []):
            line_dir = line.get("dir", (1.0, 0.0))
            line_wmode = line.get("wmode", 0)
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text.strip():
                    continue
                spans.append(
                    _Span(
                        bbox=fitz.Rect(span.get("bbox", (0, 0, 0, 0))),
                        text=text,
                        font=span.get("font", ""),
                        font_size=span.get("size", 0.0),
                        flags=span.get("flags", 0),
                        color=span.get("color", None),
                        origin=span.get("origin", None),
                        direction=line_dir,
                        ascender=span.get("ascender", None),
                        descender=span.get("descender", None),
                        wmode=line_wmode,
                        # `dir`/`wmode` live on the enclosing *line*, not the
                        # span itself -- merged in here so `raw` alone is
                        # enough for `Text.from_pymupdf` to reconstruct this
                        # Text without a separate line argument.
                        raw={**span, "dir": line_dir, "wmode": line_wmode},
                    )
                )
    return spans


def _extract_words(fitz_page: "fitz.Page") -> list[tuple]:
    """Words from page.get_text('words'), sorted into reading order:
    bucket by rounded y0 (rows), then left-to-right by x0 within each
    row -- a simple top-to-bottom / left-to-right first pass."""
    words = sorted(fitz_page.get_text("words", sort=False), key=lambda w: round(w[1]))
    ordered: list[tuple] = []
    for _row, group in groupby(words, key=lambda w: round(w[1])):
        for w in sorted(group, key=lambda w: w[0]):
            ordered.append((w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]))
    return ordered


def _match_word_to_span(bbox: "fitz.Rect", spans: list[_Span]) -> "_Span | None":
    """Pick the span with the highest intersection-area / word-area
    ratio, above `_MIN_SPAN_OVERLAP`. `get_text('words')` gives geometry
    but no font/rotation metadata; `get_text('dict')` gives spans with
    that metadata at a coarser (multi-word) granularity, so they must be
    joined by overlap."""
    word_area = max(bbox.get_area(), 1e-9)
    best: "_Span | None" = None
    best_score = _MIN_SPAN_OVERLAP
    for span in spans:
        intersection = bbox & span.bbox
        if intersection.is_empty:
            continue
        score = intersection.get_area() / word_area
        if score > best_score:
            best_score = score
            best = span
    return best


def _to_word(
    page_index: int,
    seq: int,
    raw_word: tuple,
    span: "_Span | None",
) -> Text:
    return Text.from_pymupdf(
        raw_word, span.raw if span is not None else None,
        page_index=page_index, seqno=seq,
    )
