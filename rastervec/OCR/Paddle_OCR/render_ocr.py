"""RenderOCR helper: render + OCR, backend-agnostic (see ocr_backend.py in
this same package for the pluggable OcrBackend strategy pattern -- boxes
are always returned already mapped into the caller's original image pixel
space).

Vector-stage only: renders + OCRs a text-classified vector cluster via
ocr_cluster. The Raster stage (rendering + OCR'ing raster image regions)
isn't part of this project's current scope -- ocr_cluster only ever
handles `list[VectorPath]` clusters.

A cluster render is upright to begin with (there's no manual rotation
search -- ocr_cluster renders once and calls backend.detect() once);
`rotation_used` is instead read directly off the backend's own detected
`OcrDetection.rotation`.
"""
from __future__ import annotations

import math

from PIL import Image

from rastervec.helpers.geometry import union_bbox
from rastervec.models import OcrWord, Page, TextVectorResult, VectorPath
from rastervec.OCR.Paddle_OCR.ocr_backend import OcrBackend, PaddleOcrBackend

# ocr_cluster: a render whose shorter side would fall under this many pixels
# at the requested dpi gets bumped to a higher effective dpi instead --
# PaddleOCR's detector does noticeably worse on tiny crops (e.g. a single
# narrow dimension line's short text), so a small cluster's own bbox
# shouldn't be allowed to starve it of resolution.
MIN_RENDER_SIDE_PX = 50


class RenderOCR:
    """Render + detect + confidence-voting, shared by the vector-text OCR
    steps. `backend` defaults to PaddleOcrBackend -- pass any other
    OcrBackend to swap engines without touching this class."""

    def __init__(self, backend: OcrBackend | None = None) -> None:
        self.backend = backend if backend is not None else PaddleOcrBackend()

    def ocr_boxes(
        self, image: "Image.Image"
    ) -> list[tuple[str, float, list[tuple[float, float]]]]:
        """Runs the backend on one rendered image and returns every
        detected text box separately (text, confidence, quad corners),
        left as the backend found them -- unlike ocr() below, nothing is
        joined into one string. Used by ocr() (joins these into one
        reading) and the debug app's OCR inspector (`_ocr_cluster_preview`,
        via ocr()), which needs the detected-text bbox in the rendered
        image's own pixel space."""
        detection = self.backend.detect(image)
        return [(b.text, b.confidence, b.corners) for b in detection.boxes]

    def ocr(
        self, image: "Image.Image"
    ) -> tuple[str, float, list[tuple[float, float]]]:
        """Run the backend on one rendered image; returns (text,
        confidence, bbox corners). A cluster render can produce more than
        one detected text box -- these are joined left-to-right into one
        string, with confidence averaged and the bbox corners covering
        all of them."""
        boxes = self.ocr_boxes(image)
        return self._join_boxes(boxes)

    def _join_boxes(
        self, boxes: list[tuple[str, float, list[tuple[float, float]]]],
    ) -> tuple[str, float, list[tuple[float, float]]]:
        if not boxes:
            return "", 0.0, []

        order = sorted(
            range(len(boxes)),
            key=lambda i: min((x for x, _y in boxes[i][2]), default=i),
        )
        text = " ".join(boxes[i][0] for i in order)
        scores = [boxes[i][1] for i in order]
        confidence = (sum(scores) / len(scores)) if scores else 0.0

        available = [boxes[i][2] for i in order if boxes[i][2]]
        if available:
            xs = [x for poly in available for x, _y in poly]
            ys = [y for poly in available for _x, y in poly]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        else:
            corners = []

        return text, float(confidence), corners

    def ocr_cluster(
        self,
        cluster: list[VectorPath],
        page: Page,
        renderer: "rastervec.renderer.Renderer",
        dpi: int = 300,
    ) -> TextVectorResult:
        """Render (via Renderer) once, upright, and run one backend.detect()
        call over the cluster's render. `rotation_used` comes from the
        backend's own detected orientation, not a manual rotation search.
        `words` is one OcrWord per detected box, each mapped from pixel
        space to page space via Renderer.pixel_to_page_bbox -- used by
        Renderer.render_reconstructed_page to place/scale each word into
        its own bbox instead of stretching one string across the whole
        cluster bbox. `ocr_bbox` is the union of every detected box, in
        page space (only when something was detected). `dpi` is bumped
        upward (never down) if needed so the rendered image's shorter side
        is at least MIN_RENDER_SIDE_PX -- a small cluster's own bbox
        otherwise renders to a handful of pixels at the default dpi, which
        the OCR backend reads poorly. The same (possibly bumped) dpi is
        reused below for pixel_to_page_bbox, so the pixel<->page-space
        mapping always matches the image actually rendered."""
        width_pt, height_pt = renderer.cluster_frame_size(cluster)
        min_side_pt = min(width_pt, height_pt)
        if min_side_pt > 0:
            needed_dpi = math.ceil(MIN_RENDER_SIDE_PX * 72.0 / min_side_pt)
            dpi = max(dpi, needed_dpi)
        image = renderer.render_vector_cluster(cluster, dpi)
        bbox = union_bbox([p.bbox for p in cluster])

        detection = self.backend.detect(image)
        raw_boxes = [(b.text, b.confidence, b.corners) for b in detection.boxes]
        text, confidence, pixel_corners = self._join_boxes(raw_boxes)
        rotation_used = detection.rotation

        ocr_bbox = None
        words: list[OcrWord] = []
        if pixel_corners:
            ocr_bbox = renderer.pixel_to_page_bbox(cluster, dpi, pixel_corners)
        for b in detection.boxes:
            if not b.corners:
                continue
            word_bbox = renderer.pixel_to_page_bbox(cluster, dpi, b.corners)
            words.append(OcrWord(text=b.text, confidence=b.confidence, bbox=word_bbox))

        return TextVectorResult(
            paths=cluster,
            text=text,
            confidence=confidence,
            bbox=bbox,
            ocr_bbox=ocr_bbox,
            rotation_used=rotation_used,
            page_index=page.meta.index,
            words=words or None,
        )
