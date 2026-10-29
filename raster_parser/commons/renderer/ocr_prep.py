"""Dynamic-DPI cluster rendering for OCR.

`dpi_for_cluster` raises (never lowers) the render DPI so a small cluster's shorter side still
reaches `min_render_side_px` -- PaddleOCR reads tiny crops poorly -- capped at
`max_render_dpi`. `render_cluster_with_dynamic_dpi` composes it with
`png.render_vector_cluster`.
"""

from __future__ import annotations

import math

from PIL import Image

from raster_parser.commons.helpers.geometry import PDF_POINTS_PER_INCH, union_bbox
from raster_parser.models.vector import Vector
from raster_parser.commons.renderer.png import render_vector_cluster


def dpi_for_cluster(
    vectors: list[Vector],
    dpi: int,
    padding: float,
    min_render_side_px: float,
    max_render_dpi: float,
) -> int:
    """`dpi`, raised so the cluster's shorter side (plus `padding` on both ends) renders at
    least `min_render_side_px` wide, capped at `max_render_dpi`."""
    x0, y0, x1, y1 = union_bbox([v.bbox for v in vectors])
    min_side_pt = min(x1 - x0, y1 - y0) + 2 * padding
    if min_side_pt <= 0:
        return dpi
    needed_dpi = math.ceil(min_render_side_px * PDF_POINTS_PER_INCH / min_side_pt)
    return min(max(dpi, needed_dpi), max_render_dpi)


def render_cluster_with_dynamic_dpi(
    vectors: list[Vector],
    base_dpi: int,
    min_render_side_px: float,
    max_render_dpi: float,
    padding: float = 0.0,
) -> "tuple[Image.Image, int]":
    """`(image, dpi_used)`: `dpi_for_cluster`, then `png.render_vector_cluster` at that DPI."""
    dpi_used = dpi_for_cluster(vectors, base_dpi, padding, min_render_side_px, max_render_dpi)
    return render_vector_cluster(vectors, dpi_used, padding), dpi_used
