"""Debug layers for the pipeline report (`RunSettings.on_debug_layer` / `on_debug_image`).

Render-only: nothing here feeds back into the raster pass, and `RasterDebug.for_settings` returns
`None` when no callback is set, so `extract_raster_page` then does exactly what it did before.
The files the pass writes under `dump_dir` are unchanged either way. Pixel geometry is mapped to
unrotated page space through the render's own `to_points` matrix. Stages, in run order:

- `render`: `residual page` -- the cleared page as rendered (s2).
- `split`: `text layer` / `drawing layer` -- the final text raster and drawing raster after s3/s4
  (the inputs to recognition and to autotrace).
- `detect`: `initial boxes` (s3's detection on the whole page) and `refined boxes` (s4's boxes
  on the clean text layer, the ones recognised).
- `ocr`: `text` -- the recognised words, in their boxes and directions.
- `trace`: `vectors` -- the autotraced drawing vectors in their own colours.

Debug images: `v1_raster_boxes/` gets each refined box cut from the text layer (5 px padding,
before s4's word split).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pymupdf

    from raster_parser.commons.models import PageMeta, Text, Vector
    from raster_parser.core.settings import RunSettings

BOX_FOLDER = "v1_raster_boxes"
_BOX_PAD = 5

_PAGE = "#6b7280"
_TEXT_LAYER = "#2563eb"
_DRAWING_LAYER = "#f97316"
_INITIAL = "#dc2626"
_REFINED = "#16a34a"
_TEXT = "#111827"
_TRACE = "#7c3aed"


def _rgb(hexc: str) -> tuple[float, float, float]:
    h = hexc.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


class RasterDebug:
    """Emits the raster pass's debug layers / images for one page."""

    def __init__(self, settings: "RunSettings", page_meta: "PageMeta") -> None:
        self.on_layer = settings.on_debug_layer
        self.on_image = settings.on_debug_image
        self.page_meta = page_meta
        self.to_points: "pymupdf.Matrix | None" = None
        self.size: tuple[int, int] = (0, 0)

    @classmethod
    def for_settings(cls, settings: "RunSettings", page_meta: "PageMeta") -> "RasterDebug | None":
        if settings.on_debug_layer is None and settings.on_debug_image is None:
            return None
        return cls(settings, page_meta)

    # -- geometry --------------------------------------------------------------------------
    def _page_box(self, box) -> tuple[float, float, float, float]:
        import pymupdf

        r = pymupdf.Rect(*(float(v) for v in box)) * self.to_points
        return (r.x0, r.y0, r.x1, r.y1)

    def _image_spec(self, array: "np.ndarray"):
        import pymupdf

        from raster_parser.commons.renderer.draw import image_spec

        w, h = self.size
        m = self.to_points
        corners = [pymupdf.Point(x, y) * m for x, y in ((0, 0), (w, 0), (0, h), (w, h))]
        rect = (min(p.x for p in corners), min(p.y for p in corners),
                max(p.x for p in corners), max(p.y for p in corners))
        transform = (w * m.a, w * m.b, h * m.c, h * m.d, m.e, m.f)  # unit square -> page
        return image_spec(array, rect, transform=transform)

    def _emit_image_layer(self, stage: str, label: str, hexc: str, array: "np.ndarray") -> None:
        from raster_parser.commons.renderer.draw import render_specs_pdf

        self.on_layer(stage, label, hexc, render_specs_pdf(self.page_meta, [self._image_spec(array)], deflate=True))

    def _emit_boxes(self, stage: str, label: str, hexc: str, boxes) -> None:
        from raster_parser.commons.renderer.pdf import render_boxes_pdf

        self.on_layer(stage, label, hexc, render_boxes_pdf(
            self.page_meta, [(self._page_box(b), _rgb(hexc)) for b in boxes]))

    # -- stages ----------------------------------------------------------------------------
    def rendered(self, bgr: "np.ndarray", to_points: "pymupdf.Matrix") -> None:
        self.to_points = to_points
        self.size = (bgr.shape[1], bgr.shape[0])
        if self.on_layer is not None:
            self._emit_image_layer("render", "residual page", _PAGE, bgr[:, :, ::-1])

    def split(
        self, text_gray: "np.ndarray", drawing_bgr: "np.ndarray", initial_boxes, refined_boxes,
    ) -> None:
        if self.on_layer is not None:
            self._emit_image_layer("split", "text layer", _TEXT_LAYER, text_gray)
            self._emit_image_layer("split", "drawing layer", _DRAWING_LAYER, drawing_bgr[:, :, ::-1])
            self._emit_boxes("detect", "initial boxes", _INITIAL, initial_boxes)
            self._emit_boxes("detect", "refined boxes", _REFINED, refined_boxes)
        if self.on_image is not None:
            self._offer_box_crops(text_gray, refined_boxes)

    def _offer_box_crops(self, text_gray: "np.ndarray", boxes) -> None:
        h, w = text_gray.shape[:2]
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            x0, y0 = max(0, int(x0) - _BOX_PAD), max(0, int(y0) - _BOX_PAD)
            x1, y1 = min(w, int(x1) + _BOX_PAD), min(h, int(y1) + _BOX_PAD)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = text_gray[y0:y1, x0:x1]

            def _make(crop=crop):
                import numpy as np
                from PIL import Image

                return Image.fromarray(np.ascontiguousarray(crop))

            self.on_image(BOX_FOLDER, f"box{i:04d}.png", _make)

    def output(self, texts: "list[Text]", vectors: "list[Vector]") -> None:
        if self.on_layer is None:
            return
        from raster_parser.commons.renderer.draw import render_specs_pdf, vector_spec
        from raster_parser.commons.renderer.pdf import render_text_pdf

        self.on_layer("ocr", "text", _TEXT, render_text_pdf(self.page_meta, texts, color_of=lambda _t: _rgb(_TEXT)))
        self.on_layer("trace", "vectors", _TRACE, render_specs_pdf(
            self.page_meta, [vector_spec(v) for v in vectors], deflate=True))
