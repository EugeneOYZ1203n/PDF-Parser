"""Renderer: rendering helpers, not a pipeline stage.

Turns pipeline data (vector paths, clusters, raster regions) into pixels
for OCR input (render_vector_cluster/render_raster_region -- not yet
implemented). Reconstructing pipeline output back into a PDF for
evaluation is a separate concern -- see evaluation.py, the pipeline's
actual final stage.
"""
from __future__ import annotations

from rastervec.models import Page, RasterImage, VectorPath

_DEFAULT_PATH_COLOR = "#111827"


class Renderer:
    """Rendering helpers shared by the debug app and (once built) the OCR
    input pipeline."""

    def path_color_hex(
        self, path: VectorPath, default: str = _DEFAULT_PATH_COLOR
    ) -> str:
        """A path's own stroke/fill color as a hex string -- callers
        should render the PDF's real color; any B/W-style simplification
        (e.g. Vector's background-fill heuristic) is purely an internal
        classification concern, never something substituted in its place
        for display."""
        color = path.stroke_color if path.stroke_color is not None else path.fill_color
        if color is None:
            return default
        return "#%02x%02x%02x" % tuple(min(255, max(0, round(c * 255))) for c in color)

    def render_vector_cluster(
        self, paths: list[VectorPath], page: Page, dpi: int
    ) -> "PIL.Image.Image":
        """High-resolution render of an isolated vector path cluster,
        used as OCR input."""
        raise NotImplementedError

    def render_raster_region(
        self, image: RasterImage, dpi: int
    ) -> "PIL.Image.Image":
        """High-resolution render of a raster image region, used as OCR
        input."""
        raise NotImplementedError
