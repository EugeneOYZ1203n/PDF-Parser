"""Raster stage: interface-only stub for a future phase.

Not implemented yet -- see the "Vector / Raster / Renderer / Helpers"
section of the rastervec implementation plan. Method bodies intentionally
raise NotImplementedError. No third-party imports beyond stdlib/typing are
added here yet, to keep earlier phases' install footprint minimal.

Archive -> rastervec mapping (design notes only, nothing here implements
any of this yet -- see `archive/raster_parser/` for the actual code):

- `delete_natives/delete_native.py`'s `clear_page` (LibreOffice headless
  macro strips native text/vector drawings, leaving only embedded raster
  images) + `page_is_nonempty` (decides whether the raster fallback pass
  is worth running at all) -- would gate a future `extract_remainder`-style
  entrypoint: only run this whole `Raster` stage when the LO-cleared page
  still has real ink on it, not on every page unconditionally.
- `parsing/text_drawing_parser.py`'s `OcrTextDrawingParser` (Hough-line +
  morphology line masking, inpaint-based detection cleanup, tiled
  multi-process PaddleOCR detection) -- maps onto `mask_text` above plus
  `helpers/masking.py::Masking` (also currently stubbed): line/leader-line
  removal before OCR detection, and box merge/dedup, would live there.
- `autotrace/drawing_autotrace.py` + `drawing_colour_fill_autotrace.py`
  (PNG -> SVG -> PDF -> Vector via `pyautotrace`, colour-cluster multi-mask
  tracing with page-wash detection) -- no current rastervec stub covers
  this at all. Would need a new module (e.g. `Raster_Vectorize/`) between
  `mask_text`/`find_junctions` and `extract_remainder` if ever built, and
  `pyautotrace` would need adding to `requirements.txt` (not there today).
- `scripts/testing_paper/skeletonize.py` + `chaining.py` (classical
  chamfer-distance-transform skeleton thinning + junction/endpoint
  detection + polyline chain tracing, no ML) -- a working classical-CV
  alternative to `find_junctions`/`connect_lines`/`measure_line_widths`
  below, which are scoped for a CNN (`helpers/junction.py::
  JunctionDetector`, also currently stubbed for `generate_synthetic_data`/
  `train`/`infer`). Worth revisiting as a lower-effort starting point if
  the CNN approach stalls -- archive's version is a complete, working
  implementation, not a stub.
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
