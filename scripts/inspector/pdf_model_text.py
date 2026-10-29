"""`extract_text_items` -- native text words with span orientation metadata,
plus its own `_find_best_span`/`_make_oriented_quad` helpers."""

from __future__ import annotations

from itertools import groupby
from math import degrees, atan2

import pymupdf
from raster_parser.commons.helpers.geometry import make_oriented_quad
from scripts.inspector.layers import OverlayItem
from scripts.inspector.pdf_model_core import _rect_metadata


def extract_text_items(
    page: "pymupdf.Page",
) -> list[OverlayItem]:
    """
    Extract text words while retaining span orientation metadata.
    """

    words = page.get_text(
        "words",
        sort=False,
    )

    spans = []

    # get_text("dict") nests blocks -> lines -> spans. A block's "type" is 0 for text, 1 for
    # an image. A line's "dir" is its unit writing direction (cos, sin) in unrotated page
    # space. A span is a run of characters sharing one font, size, colour and "flags" (a
    # bit field: 1 superscript, 2 italic, 4 serif, 8 monospaced, 16 bold); its "origin" is
    # the baseline start point, and "ascender"/"descender" are the font's extent above/below
    # the baseline as fractions of the font size.
    text_dict = page.get_text(
        "dict",
        sort=False,
    )

    for block in text_dict.get(
        "blocks",
        [],
    ):
        if block.get("type") != 0:
            continue

        for line in block.get(
            "lines",
            [],
        ):
            line_dir = line.get(
                "dir",
                (1.0, 0.0),
            )

            for span in line.get(
                "spans",
                [],
            ):
                text = span.get(
                    "text",
                    "",
                )

                if not text.strip():
                    continue

                bbox = pymupdf.Rect(
                    span.get(
                        "bbox",
                        (0, 0, 0, 0),
                    )
                )

                spans.append(
                    {
                        "bbox": bbox,
                        "text": text,
                        "font": span.get(
                            "font",
                            "",
                        ),
                        "size": span.get(
                            "size",
                            0,
                        ),
                        "flags": span.get(
                            "flags",
                            0,
                        ),
                        "color": span.get(
                            "color",
                            None,
                        ),
                        "origin": span.get(
                            "origin",
                            None,
                        ),
                        "dir": line_dir,
                        "ascender": span.get(
                            "ascender",
                            None,
                        ),
                        "descender": span.get(
                            "descender",
                            None,
                        ),
                    }
                )

    words = sorted(
        words,
        key=lambda w: (
            round(w[1]),
            w[0],
        ),
    )

    items = []

    for _, group in groupby(
        words,
        key=lambda w: round(w[1]),
    ):
        for word in sorted(
            group,
            key=lambda w: w[0],
        ):
            x0, y0, x1, y1, text = (
                word[0],
                word[1],
                word[2],
                word[3],
                word[4],
            )

            bbox = pymupdf.Rect(
                x0,
                y0,
                x1,
                y1,
            )

            span = _find_best_span(
                bbox,
                spans,
            )

            if span is None:
                items.append(
                    OverlayItem(
                        bbox=bbox,
                        kind="word",
                        shape="rect",
                        label=text,
                        metadata={
                            "text": text,
                            "angle": 0.0,
                            "orientation_source": "fallback",
                        },
                    )
                )

                continue

            dx, dy = span["dir"]

            angle = degrees(
                atan2(
                    dy,
                    dx,
                )
            )

            quad = _make_oriented_quad(
                bbox,
                dx,
                dy,
            )

            metadata = {
                "text": text,
                "angle": round(angle, 3),
                "direction": (
                    round(dx, 5),
                    round(dy, 5),
                ),
                "font": span["font"],
                "font_size": round(
                    span["size"],
                    3,
                ),
                "flags": span["flags"],
                "origin": span["origin"],
                "color": span["color"],
                "ascender": span["ascender"],
                "descender": span["descender"],
                "orientation_source": "text-span",
                **_rect_metadata(bbox),
            }

            items.append(
                OverlayItem(
                    bbox=bbox,
                    quad=quad,
                    angle=angle,
                    kind="word",
                    shape="quad",
                    label=text,
                    attrs={
                        "angle": angle,
                        "font": span["font"],
                        "font_size": span["size"],
                    },
                    metadata=metadata,
                )
            )

    return items


def _find_best_span(
    bbox: pymupdf.Rect,
    spans: list[dict],
) -> dict | None:

    if not spans:
        return None

    best = None
    best_score = -1.0

    word_area = max(
        bbox.get_area(),
        1e-9,
    )

    for span in spans:
        span_bbox = span["bbox"]

        intersection = bbox & span_bbox

        if intersection.is_empty:
            continue

        score = intersection.get_area() / word_area

        if score > best_score:
            best_score = score
            best = span

    return best


def _make_oriented_quad(
    bbox: pymupdf.Rect,
    dx: float,
    dy: float,
) -> pymupdf.Quad:
    """`commons.helpers.geometry.make_oriented_quad` as a `pymupdf.Quad`."""
    ul, ur, lr, ll = make_oriented_quad(tuple(bbox), dx, dy)
    return pymupdf.Quad(ul, ur, ll, lr)
