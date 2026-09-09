"""Shared dataclasses that flow between pipeline stages.

Coordinate convention
----------------------
All geometry fields here are in PDF page **unrotated MediaBox space** -- the
same space PyMuPDF's `get_text`/`get_drawings`/`get_image_info`/`annots()`
return. This stays the pipeline's canonical space throughout every stage;
only `Renderer`/reconstruction converts to rotated *display* space at the
very end (matching `page.get_pixmap()`/`page.rect`).

Two core models
---------------
`Vector` mirrors one whole `get_drawings()` drawing, never decomposed into
its own items. `Text` mirrors `get_text()`'s field surface and doubles as
the OCR result type. `Segment`/`UniqueSegment`/`SegmentMeta` are the
Radon-segmentation-through-OCR handoff types. `Page`/`PageMeta` are Reader's
own output. See `rastervec/pipelines/PIPELINE.md` for the full data flow.
"""
from __future__ import annotations

from rastervec.models.page import Page, PageMeta
from rastervec.models.segment import Segment, SegmentMeta, UniqueSegment
from rastervec.models.text import Text
from rastervec.models.vector import Vector

__all__ = [
    "Page",
    "PageMeta",
    "Vector",
    "Text",
    "Segment",
    "UniqueSegment",
    "SegmentMeta",
]
