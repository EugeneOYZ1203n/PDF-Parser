"""RenderOCR: render a vector-text cluster, deskew + segment it (Radon),
recognise the word crops, and assemble a `TextVectorResult`.

Detection is the Radon segmentation step, not PaddleOCR -- the backend
(`ocr_backend.OcrBackend`) only recognises pre-cropped words. The 0-vs-180
flip Radon cannot resolve is settled here: every word crop is recognised
upright and rotated 180, and the higher length-weighted-confidence set
wins.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from rastervec.config import MIN_RENDER_SIDE_PX
from rastervec.helpers.geometry import PDF_POINTS_PER_INCH, union_bbox
from rastervec.models import OcrWord, Page, TextVectorResult, VectorPath
from rastervec.OCR.Paddle_OCR.ocr_backend import OcrBackend, OcrBox, PaddleRecBackend
from rastervec.pipelines.sub_pipelines.radon import ClusterSegmentation, segment_cluster
from rastervec.renderer import (
    cluster_frame_size,
    pixel_to_page_bbox,
    render_vector_cluster,
)


def render_cluster_for_ocr(
    cluster: list[VectorPath], dpi: int = 300
) -> tuple["Image.Image", int]:
    """Render `cluster` as OCR sees it: `dpi` bumped upward (never down) so
    the rendered image's shorter side is at least `MIN_RENDER_SIDE_PX`.
    Returns `(image, dpi_used)`."""
    width_pt, height_pt = cluster_frame_size(cluster)
    min_side_pt = min(width_pt, height_pt)
    if min_side_pt > 0:
        needed_dpi = math.ceil(MIN_RENDER_SIDE_PX * PDF_POINTS_PER_INCH / min_side_pt)
        dpi = max(dpi, needed_dpi)
    return render_vector_cluster(cluster, dpi), dpi


def _len_weighted_conf(boxes: list[OcrBox]) -> float:
    real = [b for b in boxes if b.text]
    total = sum(len(b.text) for b in real)
    if total == 0:
        return 0.0
    return sum(len(b.text) * b.confidence for b in real) / total


class RenderOCR:
    """Render + segment + recognise. `backend` defaults to
    `PaddleRecBackend`; pass any `OcrBackend` to swap the recogniser."""

    def __init__(self, backend: OcrBackend | None = None) -> None:
        self.backend = backend if backend is not None else PaddleRecBackend()

    # -- primitives -----------------------------------------------------
    def recognize_segmented(
        self, seg: ClusterSegmentation, cluster: list[VectorPath], page: Page,
    ) -> TextVectorResult:
        """Recognise `seg`'s word crops and map each back to page space."""
        bbox = union_bbox([p.bbox for p in cluster])
        quarter = int(round(seg.skew_deg / 90.0)) * 90

        if not seg.word_crops:
            return TextVectorResult(
                paths=cluster, text="", confidence=0.0, bbox=bbox, ocr_bbox=None,
                rotation_used=quarter % 360, page_index=page.meta.index, words=None,
            )

        up = self.backend.recognize_crops(seg.word_crops)
        flipped = self.backend.recognize_crops([np.rot90(c, 2) for c in seg.word_crops])
        if _len_weighted_conf(flipped) > _len_weighted_conf(up):
            boxes, flip = flipped, 180
        else:
            boxes, flip = up, 0

        detected: list[tuple[str, float, list[tuple[float, float]], tuple]] = []
        for box, corners in zip(boxes, seg.word_corners):
            if not box.text:
                continue
            page_bbox = pixel_to_page_bbox(cluster, seg.render_dpi, corners)
            detected.append((box.text, box.confidence, corners, page_bbox))

        detected.sort(key=lambda d: min((x for x, _y in d[2]), default=0.0))
        text = " ".join(d[0] for d in detected)
        confidence = float(np.mean([d[1] for d in detected])) if detected else 0.0
        words = [OcrWord(text=d[0], confidence=d[1], bbox=d[3]) for d in detected]
        ocr_bbox = union_bbox([d[3] for d in detected]) if detected else None

        return TextVectorResult(
            paths=cluster, text=text, confidence=confidence, bbox=bbox,
            ocr_bbox=ocr_bbox, rotation_used=(quarter + flip) % 360,
            page_index=page.meta.index, words=words or None,
        )

    def ocr_cluster(
        self, cluster: list[VectorPath], page: Page, dpi: int = 300,
    ) -> TextVectorResult:
        """Render `cluster`, Radon-segment it, recognise -- the whole
        vector-text OCR path for one cluster."""
        image, dpi_used = render_cluster_for_ocr(cluster, dpi)
        seg = segment_cluster(image, dpi_used)
        return self.recognize_segmented(seg, cluster, page)

    # -- convenience for the inspector / debug previews ----------------
    def ocr(self, image: "Image.Image") -> tuple[str, float, list[tuple[float, float]]]:
        """OCR one already-rendered image; returns `(text, confidence,
        pixel-space bbox corners)`. Segmentation + recognition, no page
        mapping."""
        seg = segment_cluster(image, 300)
        if not seg.word_crops:
            return "", 0.0, []
        up = self.backend.recognize_crops(seg.word_crops)
        flipped = self.backend.recognize_crops([np.rot90(c, 2) for c in seg.word_crops])
        boxes = flipped if _len_weighted_conf(flipped) > _len_weighted_conf(up) else up

        found = [
            (b.text, b.confidence, corners)
            for b, corners in zip(boxes, seg.word_corners)
            if b.text
        ]
        if not found:
            return "", 0.0, []
        found.sort(key=lambda d: min(x for x, _y in d[2]))
        text = " ".join(d[0] for d in found)
        confidence = float(np.mean([d[1] for d in found]))
        xs = [x for _t, _c, poly in found for x, _y in poly]
        ys = [y for _t, _c, poly in found for _x, y in poly]
        corners = [(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))]
        return text, confidence, corners
