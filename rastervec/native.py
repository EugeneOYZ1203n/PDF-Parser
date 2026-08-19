"""Native stage: extracts words with position, rotation, font and font-size
directly from the PDF's own text objects (no OCR/rendering involved).
"""
from __future__ import annotations

from itertools import groupby
from math import atan2, degrees, hypot

import pymupdf as fitz

from rastervec.geometry import make_oriented_quad
from rastervec.logging_setup import get_logger
from rastervec.models import Page, TextWord

_LOG = get_logger("native")


class Native:
    """Extracts native (non-OCR) text from a page."""

    def extract_text(self, page: Page) -> list[TextWord]:
        fitz_page = page.fitz_page
        page_index = page.meta.index

        spans = self._extract_spans(fitz_page)
        words = self._extract_words(fitz_page)

        _LOG.debug(
            "page %d: %d word(s), %d span(s)",
            page_index,
            len(words),
            len(spans),
        )

        result: list[TextWord] = []
        seq = 0
        for x0, y0, x1, y1, text in words:
            bbox = fitz.Rect(x0, y0, x1, y1)
            span = self._match_word_to_span(bbox, spans)
            if span is None:
                _LOG.warning(
                    "page %d: no matching span for word %r at %s",
                    page_index,
                    text,
                    bbox,
                )
            result.append(
                self._to_text_word(
                    (bbox, text), span, page_index, seq
                )
            )
            seq += 1

        return result

    def _extract_spans(self, fitz_page: "fitz.Page") -> list[dict]:
        """Flatten page.get_text('dict') into a list of span dicts."""
        spans: list[dict] = []
        text_dict = fitz_page.get_text("dict", sort=False)

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                line_dir = line.get("dir", (1.0, 0.0))

                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text.strip():
                        continue

                    bbox = fitz.Rect(span.get("bbox", (0, 0, 0, 0)))

                    spans.append(
                        {
                            "bbox": bbox,
                            "text": text,
                            "font": span.get("font", ""),
                            "size": span.get("size", 0),
                            "flags": span.get("flags", 0),
                            "color": span.get("color", None),
                            "origin": span.get("origin", None),
                            "dir": line_dir,
                            "ascender": span.get("ascender", None),
                            "descender": span.get("descender", None),
                        }
                    )

        return spans

    def _extract_words(self, fitz_page: "fitz.Page") -> list[tuple]:
        """Words from page.get_text('words'), sorted into reading order.

        Bucket by rounded y0 (rows), then sort each row left-to-right by
        x0 -- a simple top-to-bottom/left-to-right approximation of
        reading order, good enough as a first pass.
        """
        words = fitz_page.get_text("words", sort=False)
        words = sorted(words, key=lambda w: (round(w[1]), w[0]))

        ordered: list[tuple] = []
        for _, group in groupby(words, key=lambda w: round(w[1])):
            for word in sorted(group, key=lambda w: w[0]):
                ordered.append(
                    (word[0], word[1], word[2], word[3], word[4])
                )
        return ordered

    def _match_word_to_span(
        self, bbox: "fitz.Rect", spans: list[dict]
    ) -> dict | None:
        """Pick the span with the highest intersection-area/word-area ratio.

        get_text('words') gives geometry but no font/rotation metadata;
        get_text('dict') gives spans with that metadata but at a coarser
        (multi-word) granularity, so they must be joined by overlap.
        """
        if not spans:
            return None

        best = None
        best_score = -1.0
        word_area = max(bbox.get_area(), 1e-9)

        for span in spans:
            intersection = bbox & span["bbox"]
            if intersection.is_empty:
                continue
            score = intersection.get_area() / word_area
            if score > best_score:
                best_score = score
                best = span

        return best

    def _build_oriented_quad(
        self, bbox: "fitz.Rect", dx: float, dy: float
    ) -> "fitz.Quad":
        return make_oriented_quad(bbox, dx, dy)

    def _word_origin(
        self, bbox: "fitz.Rect", span_origin: tuple[float, float] | None,
        dx: float, dy: float,
    ) -> tuple[float, float] | None:
        """A per-word insertion point on the span's baseline.

        get_text('dict')'s span-level `origin` is the baseline start of the
        *whole* span (which can hold several words) -- using it verbatim as
        every one of that span's words' origin (the previous behavior) gave
        every word in a line the exact same insertion point, which is why
        Renderer.render_reconstructed_page drew every word on a line
        stacked at the same spot instead of spread out. This instead keeps
        the span's baseline (its perpendicular offset from the direction
        line, `normal_offset`) but moves along the direction to this word's
        own leading edge (`along_min`, from its own bbox) -- reusing the
        same along/normal projection geometry.make_oriented_quad uses, so
        it's correct for rotated/vertical text too, not just horizontal."""
        if span_origin is None:
            return None
        length = hypot(dx, dy)
        if length < 1e-9:
            dx, dy = 1.0, 0.0
            length = 1.0
        dx, dy = dx / length, dy / length
        nx, ny = -dy, dx

        corners = [
            (bbox.x0, bbox.y0), (bbox.x1, bbox.y0),
            (bbox.x1, bbox.y1), (bbox.x0, bbox.y1),
        ]
        along_min = min(x * dx + y * dy for x, y in corners)
        normal_offset = span_origin[0] * nx + span_origin[1] * ny

        return (along_min * dx + normal_offset * nx, along_min * dy + normal_offset * ny)

    def _to_text_word(
        self,
        word: tuple,
        span: dict | None,
        page_index: int,
        seq: int,
    ) -> TextWord:
        bbox, text = word

        if span is None:
            return TextWord(
                text=text,
                bbox=(bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                quad=(
                    (bbox.x0, bbox.y0),
                    (bbox.x1, bbox.y0),
                    (bbox.x1, bbox.y1),
                    (bbox.x0, bbox.y1),
                ),
                angle=0.0,
                direction=(1.0, 0.0),
                font="",
                font_size=0.0,
                color=None,
                flags=0,
                origin=None,
                ascender=None,
                descender=None,
                orientation_source="fallback",
                page_index=page_index,
                seq=seq,
            )

        dx, dy = span["dir"]
        angle = degrees(atan2(dy, dx))
        quad = self._build_oriented_quad(bbox, dx, dy)

        return TextWord(
            text=text,
            bbox=(bbox.x0, bbox.y0, bbox.x1, bbox.y1),
            quad=(
                (quad.ul.x, quad.ul.y),
                (quad.ur.x, quad.ur.y),
                (quad.lr.x, quad.lr.y),
                (quad.ll.x, quad.ll.y),
            ),
            angle=angle,
            direction=(dx, dy),
            font=span["font"],
            font_size=span["size"],
            color=span["color"],
            flags=span["flags"],
            origin=self._word_origin(bbox, span["origin"], dx, dy),
            ascender=span["ascender"],
            descender=span["descender"],
            orientation_source="text-span",
            page_index=page_index,
            seq=seq,
        )
