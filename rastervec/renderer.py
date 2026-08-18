"""Renderer stage: interface-only stub for a future phase.

Not implemented yet -- see the "Vector / Raster / Renderer / Helpers"
section of the rastervec implementation plan. Method bodies intentionally
raise NotImplementedError. No third-party imports beyond stdlib/typing are
added here yet, to keep earlier phases' install footprint minimal.
"""
from __future__ import annotations

from rastervec.models import LineVector, Page, RasterImage, ReconstructedPage, VectorPath


class Renderer:
    """Renders isolated regions for OCR, and reconstructs PDFs for
    evaluation from the pipeline's consolidated output."""

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

    def reconstruct_page(
        self,
        page_index: int,
        texts: list,
        lines: list[LineVector],
        remainder_image: bytes | None,
    ) -> ReconstructedPage:
        """Consolidate one page's text/line/remainder-image outputs into
        a single ReconstructedPage record."""
        raise NotImplementedError

    def build_pdf(
        self, pages: list[ReconstructedPage], out_path: str
    ) -> None:
        """Evaluation step: rebuild a PDF from generated text/line
        vectors plus stored remainder images."""
        raise NotImplementedError
