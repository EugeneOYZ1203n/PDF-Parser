"""Vector stage: extracts vector drawings from a page.

extract_vectors (plus layer/color separation, in layer_color_separation.py)
is this module's concern. One `Vector` per `get_drawings()` entry -- never
decomposed into its own items; no per-kind item parsing happens here, just
drawing-level field copying + `helpers.fitz_geometry.plain_item` to strip
fitz objects out of `items`. Classification of vectors into text candidates
vs. drawing content lives in rastervec/Vector_Classification/ instead.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rastervec.helpers.fitz_geometry import plain_item
from rastervec.helpers.geometry import round_color
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
    vectors: list[Vector] = []

    for seq, drawing in enumerate(fitz_page.get_drawings()):
        rect = drawing.get("rect")
        scissor = drawing.get("scissor")
        line_cap = drawing.get("lineCap", 0) or 0
        if isinstance(line_cap, (list, tuple)):
            line_cap = line_cap[0] if line_cap else 0

        vectors.append(
            Vector(
                type=drawing.get("type", ""),
                items=[plain_item(item) for item in drawing.get("items", [])],
                color=round_color(drawing.get("color")),
                fill=round_color(drawing.get("fill")),
                width=drawing.get("width"),
                dashes=drawing.get("dashes"),
                closePath=drawing.get("closePath"),
                lineCap=int(line_cap),
                lineJoin=drawing.get("lineJoin", 0) or 0,
                even_odd=bool(drawing.get("even_odd", False)),
                stroke_opacity=drawing.get("stroke_opacity"),
                fill_opacity=drawing.get("fill_opacity"),
                layer=drawing.get("layer") or None,
                rect=(rect.x0, rect.y0, rect.x1, rect.y1) if rect is not None else (0.0, 0.0, 0.0, 0.0),
                scissor=(scissor.x0, scissor.y0, scissor.x1, scissor.y1) if scissor is not None else None,
                seqno=drawing.get("seqno", seq),
                blendmode=drawing.get("blendmode"),
                isolated=bool(drawing.get("isolated", False)),
                knockout=bool(drawing.get("knockout", False)),
                opacity=drawing.get("opacity"),
                page_index=page_index,
            )
        )

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
