"""Evaluation stage: interface-only stub for the pipeline's final phase.

Not implemented yet -- see the "Vector / Raster / Renderer / Helpers"
section of the rastervec implementation plan. Method bodies intentionally
raise NotImplementedError. Consolidates one page's text/line/remainder
outputs into a ReconstructedPage, then rebuilds a PDF from all pages'
outputs so the extraction pipeline's result can be scored against the
original -- this is the pipeline's actual last step, distinct from
renderer.py (which only renders pixels, e.g. OCR input or debug-app
overlays, and is never itself a pipeline stage).
"""
from __future__ import annotations

from rastervec.models import LineVector, ReconstructedPage


class Evaluation:
    """Reconstructs pipeline output into a PDF for evaluation against the
    original."""

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
        """Rebuild a PDF from generated text/line vectors plus stored
        remainder images, for scoring against the original."""
        raise NotImplementedError
