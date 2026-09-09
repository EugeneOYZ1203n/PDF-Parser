"""Vector stage: extracts vector drawings from a page.

extract_vectors (plus layer/color separation, in layer_color_separation.py)
is this module's concern. One `Vector` per `get_drawings()` entry -- never
decomposed into its own items; no per-kind item parsing happens here, just
`Vector.from_pymupdf` per drawing (drawing-level field copying +
`helpers.fitz_geometry.plain_item` to strip fitz objects out of `items`).
Classification of vectors into text candidates vs. drawing content lives in
rastervec/Vector_Classification/ instead.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rastervec.logging_setup import get_logger
from rastervec.models import Page, Vector
from rastervec.Vector.layer_color_separation import (  # noqa: F401 -- re-exported for callers
    separate_by_color,
    separate_by_layer,
)

if TYPE_CHECKING:
    from rastervec.pipelines.result import PipelineResult
    from rastervec.renderer.notebook import RenderResult

_LOG = get_logger("vector")


def extract_vectors(page: Page) -> list[Vector]:
    """One `Vector` per raw `get_drawings()` drawing, in page order."""
    fitz_page = page.fitz_page
    page_index = page.meta.index

    vectors = [
        Vector.from_pymupdf(drawing, page_index=page_index, seq=seq)
        for seq, drawing in enumerate(fitz_page.get_drawings())
    ]

    _LOG.debug("page %d: extracted %d vector(s)", page_index, len(vectors))
    return vectors


# --------------------------------------------------------------------------
# notebook visualization (pipeline_stage_visualization.ipynb's "Vector
# Extraction" section) -- reads a PipelineResult, never called by the real
# pipeline.
# --------------------------------------------------------------------------
def render_vectors(res: "PipelineResult") -> "RenderResult":
    """One category per distinct `Vector.type`."""
    from rastervec.renderer.notebook import RenderResult

    vectors = res.vectors_raw or []
    kinds = sorted({v.type for v in vectors})
    return RenderResult(
        categories=[
            {"name": f"type {k!r} ({sum(v.type == k for v in vectors)})",
             "vectors": [v for v in vectors if v.type == k]}
            for k in kinds
        ],
        note=f"{len(vectors)} vectors, types={kinds}",
    )
