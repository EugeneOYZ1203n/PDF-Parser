"""RenderOCR helper: interface-only stub for a future phase.

Shared by the Vector stage (rendering + OCR'ing text-classified vector
clusters) and the Raster stage (rendering + OCR'ing raster image regions).
Not implemented yet; method bodies intentionally raise NotImplementedError.
No PIL/PaddleOCR imports yet -- deferred to when this is actually
implemented.
"""
from __future__ import annotations

from rastervec.models import TextVectorResult


class RenderOCR:
    """Multi-rotation render + OCR + confidence-voting, shared by the
    vector-text and raster-image OCR steps."""

    def render_rotations(
        self, image: "PIL.Image.Image", n: int = 8
    ) -> list["PIL.Image.Image"]:
        """Render n evenly-spaced rotations of image for OCR voting."""
        raise NotImplementedError

    def ocr(
        self, image: "PIL.Image.Image"
    ) -> tuple[str, float, list[tuple[float, float]]]:
        """Run PaddleOCR (use_angle_cls=True) on one rendered image;
        returns (text, confidence, bbox corners)."""
        raise NotImplementedError

    def combine_rotation_results(
        self, results: list[tuple[str, float, list]]
    ) -> tuple[str, float]:
        """Combine OCR results across rotations: group by text
        similarity, sum confidence per group, and return the text of the
        highest-total-confidence group -- so a majority of
        similar-but-imperfect readings can outweigh one outlier's higher
        raw per-rotation confidence."""
        raise NotImplementedError

    def ocr_cluster(
        self, paths_or_image, renderer: "rastervec.renderer.Renderer"
    ) -> TextVectorResult:
        """Shared entrypoint: render (via Renderer) + OCR across
        rotations + combine, for either a vector path cluster or a
        raster image region."""
        raise NotImplementedError
