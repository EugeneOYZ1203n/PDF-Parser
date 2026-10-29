"""Replay shared `Vector`s onto a PyMuPDF page the way the original raster_parser reconstructor did
(`rendering/pdf_render/reconstruct.py`): every item drawn, then one `Shape.finish()` per vector.

Unlike `commons.renderer._shapes.replay_drawing_paths` there is no hairline fix-up, blend-mode
wrapping or clipping, and unset fields take the old defaults (width 0, opacity 1, closePath
False). The v1 steps depend on these exact pixels: `v1_vector_class` OCRs them.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from raster_parser.models.vector import Vector
from raster_parser.commons.renderer._shapes import replay_item

# With no page size given, `draw_vectors` makes the page the vectors' extent plus this margin
# (PDF points), and at least this big.
DEFAULT_PAGE_MARGIN_PT = 20.0
MIN_PAGE_SIDE_PT = 400.0


def finish_kwargs(v: Vector) -> dict:
    """`Shape.finish()` arguments for `v` (see `models.vector.Vector` for what each field means)."""
    kwargs: dict = {
        "width": v.width or 0,
        "even_odd": bool(v.even_odd),
        "closePath": bool(v.closePath),
        "lineJoin": v.lineJoin,
        "stroke_opacity": v.stroke_opacity if v.stroke_opacity is not None else 1.0,
        "fill_opacity": v.fill_opacity if v.fill_opacity is not None else 1.0,
        "lineCap": v.line_cap,
    }
    if v.fill is not None:
        kwargs["fill"] = v.fill
    if v.color is not None:
        kwargs["color"] = v.color
    if v.dashes:
        kwargs["dashes"] = v.dashes
    return kwargs


def replay_items(shape: pymupdf.Shape, items: list, dx: float = 0.0, dy: float = 0.0) -> None:
    """Draw plain `Vector.items` into `shape`, every point moved by `(-dx, -dy)`."""
    for item in items:
        if not replay_item(shape, item, -dx, -dy):
            raise ValueError(f"unhandled drawing op {item[0]!r}: {item!r}")


def draw_vectors(
    vectors: list[Vector],
    out_pdf: Path | str,
    *,
    page_width: float | None = None,
    page_height: float | None = None,
    margin: float = DEFAULT_PAGE_MARGIN_PT,
) -> Path:
    """Replay `vectors` onto a new one-page PDF. Without a page size the page fits the vectors
    plus `margin` (at least 400 x 400)."""
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if page_width is None or page_height is None:
        max_x = max((float(v.rect[2]) for v in vectors), default=0.0)
        max_y = max((float(v.rect[3]) for v in vectors), default=0.0)
        page_width = page_width if page_width is not None else max(max_x + margin, MIN_PAGE_SIDE_PT)
        page_height = (
            page_height if page_height is not None else max(max_y + margin, MIN_PAGE_SIDE_PT)
        )

    doc = pymupdf.open()
    try:
        page = doc.new_page(width=page_width, height=page_height)
        shape = page.new_shape()
        for v in vectors:
            replay_items(shape, v.items)
            shape.finish(**finish_kwargs(v))
        shape.commit()
        doc.save(out_pdf)
    finally:
        doc.close()
    return out_pdf.resolve()
