"""Raster stage: interface-only stub for a future phase.

Not implemented yet -- see the "Vector / Raster / Renderer / Helpers"
section of the rastervec implementation plan. Method bodies intentionally
raise NotImplementedError. No third-party imports beyond stdlib/typing are
added here yet, to keep earlier phases' install footprint minimal.
"""
from __future__ import annotations

from rastervec.models import JunctionPoint, LineVector, Page, RasterImage, TextVectorResult


class Raster:
    """Extracts embedded raster images and vectorizes their line content."""

    def extract_images(self, page: Page) -> list[RasterImage]:
        """get_images()/get_image_info(xrefs=True) -> RasterImage list."""
        raise NotImplementedError

    def separate_by_color(self, image: RasterImage) -> list[RasterImage]:
        """HSV pixel clustering (unknown k, threshold-based) via
        Clustering.cluster_hsv -> per-color sub-images, each treated as
        black-and-white thereafter."""
        raise NotImplementedError

    def mask_text(
        self, image: RasterImage, ocr_results: list[TextVectorResult]
    ) -> RasterImage:
        """Mask OCR bboxes via Masking, selecting only text-colored
        pixels within each box."""
        raise NotImplementedError

    def find_junctions(self, image: RasterImage) -> list[JunctionPoint]:
        """Run the trained JunctionDetector CNN over the masked image."""
        raise NotImplementedError

    def connect_lines(
        self, junctions: list[JunctionPoint], image: RasterImage
    ) -> list[LineVector]:
        """Via JunctionDetector: 360-degree probe at each junction, then
        connect points that are closest to / almost on the same line."""
        raise NotImplementedError

    def measure_line_widths(
        self, junctions: list[JunctionPoint], image: RasterImage
    ) -> list[LineVector]:
        """Via JunctionDetector: expand a sampling radius from each
        junction until circumference samples are no longer black."""
        raise NotImplementedError

    def extract_remainder(
        self, image: RasterImage, lines: list[LineVector]
    ) -> bytes:
        """Dilate and mask out identified lines; return the remaining
        image as PNG bytes."""
        raise NotImplementedError
