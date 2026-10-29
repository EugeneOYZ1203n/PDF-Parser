"""Shared data types that flow between pipeline stages (`Text` and `Vector` are pydantic
models, the rest dataclasses).

Coordinate convention
----------------------
All geometry fields here are in the page's **unrotated space, relative to the CropBox's
top-left corner** -- the space PyMuPDF's `get_text`/`get_drawings`/`get_image_info`/`annots()`
return (`PageMeta.width`/`height` are the CropBox size). This stays the canonical space through
every stage; only code that draws for display converts to rotated *display* space (matching
`page.get_pixmap()`/`page.rect`) at the very end.

Two core models
---------------
`Vector` mirrors one whole `get_drawings()` drawing, never decomposed into
its own items. `Text` mirrors `get_text()`'s field surface and doubles as
the OCR result type. `Page`/`PageMeta` are Reader's own output; `ExtractResult` is what `extract()` returns.

Where everything lives
----------------------
Every data type the package defines is in this folder, so nothing else declares a dataclass or enum:
  - `page`, `text`, `vector`, `image`, `result`: the shared models above;
  - `source`: `Source`, where a `Text` / `Vector` came from;
  - `settings`: `RunSettings`, the options every backend receives (and its debug hooks);
  - `render`: the draw specs `commons/renderer/draw.py` builds and draws;
  - `native_read`: `NativeReadResult` and the `NativeSpan` words are matched to;
  - `raster_vectorize`: the v1_raster_to_vec step results (`OcrParseResult`, `RegionOcrResult`,
    `GlyphCc`, `ColourGroup`, `ColourFillTraceResult`);
  - `vector_classify`: `Orientation` and `WordGroup`.
Import the shared four from here; the rest from their module.
"""

from __future__ import annotations

from raster_parser.models.image import Image
from raster_parser.models.page import Page, PageMeta
from raster_parser.models.result import ExtractResult
from raster_parser.models.source import Source
from raster_parser.models.text import Text
from raster_parser.models.vector import Vector

__all__ = [
    "Page",
    "PageMeta",
    "Source",
    "Vector",
    "Text",
    "Image",
    "ExtractResult",
]
