"""Debug layers for the pipeline report (`RunSettings.on_debug_layer` / `on_debug_image`).

Render-only: nothing here feeds back into `parse`, and nothing is rendered unless a callback is
set (`emit_layers` returns at once otherwise). Each layer is one PDF page in unrotated page space
built with `commons/renderer`. Stages, in run order:

- `fill_filter`: `glyph candidates` (s1's fill-glyph candidates) and `drawing` (everything else,
  returned unchanged as drawing).
- `word_grouping`: `word groups`, one box per s2 group, coloured by its orientation.
- `ocr`: `read` / `blank` group boxes (s4 returned a word / nothing) and `text` (the words,
  placed in their boxes with their final direction).
- `orientation`: `direction`, an arrow per word along its final direction (s5).

Debug images: `v1_ocr_input/` gets each word group's normalised crop, exactly the array s4 hands
to PaddleOCR first.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from raster_parser.commons.models import PageMeta, Text, Vector
    from raster_parser.core.settings import RunSettings

OCR_INPUT_FOLDER = "v1_ocr_input"

_CANDIDATE = "#2563eb"
_DRAWING = "#9ca3af"
_READ = "#16a34a"
_BLANK = "#dc2626"
_TEXT = "#111827"
_ARROW = "#7c3aed"
_ORIENTATION_COLOURS = {"HORIZONTAL": "#16a34a", "VERTICAL": "#f97316", "OTHER": "#a855f7"}


def _rgb(hexc: str) -> tuple[float, float, float]:
    h = hexc.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def emit_layers(
    settings: "RunSettings",
    page_meta: "PageMeta",
    *,
    candidates: "list[Vector]",
    drawings: "list[Vector]",
    word_groups: list,
    read_groups: set[int],
    texts: "list[Text]",
) -> None:
    sink = settings.on_debug_layer
    if sink is None:
        return
    from raster_parser.commons.renderer.draw import arrow_spec, render_specs_pdf
    from raster_parser.commons.renderer.pdf import render_boxes_pdf, render_text_pdf, render_vectors_pdf

    sink("fill_filter", "glyph candidates", _CANDIDATE,
         render_vectors_pdf(page_meta, candidates, color_of=lambda _v: _rgb(_CANDIDATE)))
    sink("fill_filter", "drawing", _DRAWING,
         render_vectors_pdf(page_meta, drawings, color_of=lambda _v: _rgb(_DRAWING)))

    boxes = [(wg.bbox.x0, wg.bbox.y0, wg.bbox.x1, wg.bbox.y1) for wg in word_groups]
    sink("word_grouping", "word groups", _ORIENTATION_COLOURS["HORIZONTAL"], render_boxes_pdf(
        page_meta,
        [(b, _rgb(_ORIENTATION_COLOURS.get(wg.orientation.name, _BLANK))) for b, wg in zip(boxes, word_groups)],
    ))

    sink("ocr", "read", _READ, render_boxes_pdf(
        page_meta, [(b, _rgb(_READ)) for i, b in enumerate(boxes) if i in read_groups]))
    sink("ocr", "blank", _BLANK, render_boxes_pdf(
        page_meta, [(b, _rgb(_BLANK)) for i, b in enumerate(boxes) if i not in read_groups]))
    sink("ocr", "text", _TEXT, render_text_pdf(page_meta, texts, color_of=lambda _t: _rgb(_TEXT)))

    arrows = []
    for t in texts:
        x0, y0, x1, y1 = t.bbox
        length = max(min(max(x1 - x0, y1 - y0), 40.0), 6.0)
        arrows += arrow_spec(((x0 + x1) / 2, (y0 + y1) / 2), t.angle(), length, _rgb(_ARROW))
    sink("orientation", "direction", _ARROW, render_specs_pdf(page_meta, arrows, deflate=True))


def crop_sink(settings: "RunSettings"):
    """The `on_crop(group_index, crop)` hook for `s4_word_ocr.ocr_word_groups`, or `None` when
    no image callback is set (s4 then does exactly what it did before)."""
    sink = settings.on_debug_image
    if sink is None:
        return None

    def _offer(group_index: int, crop) -> None:
        def _make():
            import numpy as np
            from PIL import Image

            return Image.fromarray(np.ascontiguousarray(crop))  # s3 renders RGB

        sink(OCR_INPUT_FOLDER, f"group{group_index:04d}.png", _make)

    return _offer
