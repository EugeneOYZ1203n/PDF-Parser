"""`extract_text_items` -- native text words with span orientation metadata,
plus its own `_find_best_span`/`_make_oriented_quad` helpers."""
from __future__ import annotations

from itertools import groupby
from math import degrees, atan2, hypot

import pymupdf as fitz

from rastervec.Evaluation.inspector.layers import OverlayItem
from rastervec.Evaluation.inspector.pdf_model_core import _rect_metadata


def extract_text_items(
    page: "fitz.Page",
) -> list[OverlayItem]:
    """
    Extract text words while retaining span orientation metadata.
    """

    words = page.get_text(
        "words",
        sort=False,
    )

    spans = []

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

                bbox = fitz.Rect(
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

            bbox = fitz.Rect(
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
    bbox: fitz.Rect,
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

        score = (
            intersection.get_area()
            / word_area
        )

        if score > best_score:
            best_score = score
            best = span

    return best


def _make_oriented_quad(
    bbox: fitz.Rect,
    dx: float,
    dy: float,
) -> fitz.Quad:
    """Build a quad around bbox, oriented along the text direction (dx, dy).

    bbox.width/height are the *axis-aligned* extents, which only match the
    text's along-direction/normal-direction extents when the text is
    horizontal. For rotated text they can be swapped or otherwise wrong, so
    the bbox corners are projected onto the (dx, dy) / normal axes instead
    to recover the correct along/normal extents regardless of orientation.
    """

    length = hypot(
        dx,
        dy,
    )

    if length < 1e-9:
        dx = 1.0
        dy = 0.0
        length = 1.0

    dx /= length
    dy /= length

    nx = -dy
    ny = dx

    corners = [
        (bbox.x0, bbox.y0),
        (bbox.x1, bbox.y0),
        (bbox.x1, bbox.y1),
        (bbox.x0, bbox.y1),
    ]

    along_vals = [x * dx + y * dy for x, y in corners]
    normal_vals = [x * nx + y * ny for x, y in corners]

    along_min, along_max = min(along_vals), max(along_vals)
    normal_min, normal_max = min(normal_vals), max(normal_vals)

    half_along = (along_max - along_min) / 2.0
    half_normal = (normal_max - normal_min) / 2.0

    along_center = (along_max + along_min) / 2.0
    normal_center = (normal_max + normal_min) / 2.0

    cx = along_center * dx + normal_center * nx
    cy = along_center * dy + normal_center * ny

    def point(
        along: float,
        normal: float,
    ) -> fitz.Point:

        return fitz.Point(
            cx
            + dx * along
            + nx * normal,
            cy
            + dy * along
            + ny * normal,
        )

    ul = point(
        -half_along,
        -half_normal,
    )

    ur = point(
        half_along,
        -half_normal,
    )

    ll = point(
        -half_along,
        half_normal,
    )

    lr = point(
        half_along,
        half_normal,
    )

    return fitz.Quad(
        ul,
        ur,
        ll,
        lr,
    )
