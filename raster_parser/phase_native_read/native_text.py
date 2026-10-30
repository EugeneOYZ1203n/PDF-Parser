"""Native stage: extracts words with position, rotation, font and font-size
directly from the PDF's own text objects (no OCR/rendering involved).
"""

from __future__ import annotations

from itertools import groupby

import pymupdf
from raster_parser.commons.logging_setup import get_logger
from raster_parser.models.page import Page
from raster_parser.models.text import Text
from raster_parser.models.native_read import NativeSpan

_LOG = get_logger("native")

# A word overlapping a span by less than this fraction of the word's own
# area is treated as unmatched (garbage geometry / stray span) rather than
# inheriting that span's font metadata.
_MIN_SPAN_OVERLAP = 0.10


def extract_native_text(page: Page) -> list[Text]:
    """One `Text` (source=Source.NATIVE_TEXT) per `get_text("words")` word, in reading
    order, with font/rotation metadata joined from the best-overlapping
    `get_text("dict")` span."""
    pymupdf_page = page.pymupdf_page
    page_index = page.meta.index

    spans = _extract_spans(pymupdf_page)
    words = _extract_words(pymupdf_page)
    _LOG.debug(
        "page %d: %d word(s), %d span(s)",
        page_index,
        len(words),
        len(spans),
    )

    result: list[Text] = []
    for seq, raw_word in enumerate(words):
        x0, y0, x1, y1, text, _block_no, _line_no, _word_no = raw_word
        bbox = pymupdf.Rect(x0, y0, x1, y1)
        span = _match_word_to_span(bbox, spans)
        if span is None:
            _LOG.warning(
                "page %d: no matching span for word %r at %s",
                page_index,
                text,
                bbox,
            )
        result.append(_to_word(page_index, seq, raw_word, span))
    return result


def _extract_spans(pymupdf_page: "pymupdf.Page") -> list[NativeSpan]:
    """Flatten page.get_text('dict') into a list of `NativeSpan`s."""
    spans: list[NativeSpan] = []
    text_dict = pymupdf_page.get_text("dict", sort=False)

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
                    NativeSpan(
                        bbox=pymupdf.Rect(span.get("bbox", (0, 0, 0, 0))),
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


def _extract_words(pymupdf_page: "pymupdf.Page") -> list[tuple]:
    """Words from page.get_text('words') -- tuples (x0, y0, x1, y1, word, block_no,
    line_no, word_no) -- sorted into reading order:
    bucket by rounded y0 (rows), then left-to-right by x0 within each
    row -- a simple top-to-bottom / left-to-right first pass."""
    words = sorted(pymupdf_page.get_text("words", sort=False), key=lambda w: round(w[1]))
    ordered: list[tuple] = []
    for _row, group in groupby(words, key=lambda w: round(w[1])):
        for w in sorted(group, key=lambda w: w[0]):
            ordered.append((w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]))
    return ordered


def _match_word_to_span(bbox: "pymupdf.Rect", spans: list[NativeSpan]) -> "NativeSpan | None":
    """Pick the span with the highest intersection-area / word-area
    ratio, above `_MIN_SPAN_OVERLAP`. `get_text('words')` gives geometry
    but no font/rotation metadata; `get_text('dict')` gives spans with
    that metadata at a coarser (multi-word) granularity, so they must be
    joined by overlap."""
    word_area = max(bbox.get_area(), 1e-9)
    best: "NativeSpan | None" = None
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
    span: "NativeSpan | None",
) -> Text:
    return Text.from_pymupdf(
        raw_word,
        span.raw if span is not None else None,
        page_index=page_index,
        seqno=seq,
    )
