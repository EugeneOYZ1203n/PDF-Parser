"""RenderOCR helper: multi-rotation render + OCR + confidence-voting.

Shared by the Vector stage (rendering + OCR'ing text-classified vector
clusters, via ocr_cluster) and, once built, the Raster stage (rendering +
OCR'ing raster image regions -- Renderer.render_raster_region is still a
stub, so ocr_cluster only actually works against vector clusters today).

Uses PaddleOCR (PP-OCRv6, text-detection + text-recognition only -- no doc
orientation/unwarping, since callers hand in an already-isolated, already
upright cluster render) for the underlying single-image OCR call. PaddleOCR
engines are expensive to construct (they load model weights), so one is
built lazily per (lang) and cached at module scope -- every RenderOCR
instance for the same lang shares it.
"""
from __future__ import annotations

import difflib
import os

# Must be set before paddlex reads its flags on import (which happens the
# first time _engine() lazily imports paddleocr below) -- on this project's
# dev environment, the default (mkldnn-accelerated) CPU inference path hits
# an unimplemented PIR attribute-conversion error in this paddlepaddle
# build; plain "paddle" run mode works fine and is plenty fast for
# small, pre-cropped cluster renders. Only set if the caller hasn't
# already configured this themselves.
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")

from PIL import Image

from rastervec.geometry import union_bbox
from rastervec.models import Page, RasterImage, TextVectorResult, VectorPath

_ENGINE_CACHE: dict[str, "paddleocr.PaddleOCR"] = {}

# Two readings are treated as "the same text" (grouped together) when their
# normalized similarity ratio is at least this high -- high enough that
# genuinely different text doesn't get merged, but tolerant of the kind of
# single-character misreads OCR produces across slightly different
# rotations of the same glyphs.
_SIMILARITY_THRESHOLD = 0.75


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


class RenderOCR:
    """Multi-rotation render + OCR + confidence-voting, shared by the
    vector-text and raster-image OCR steps."""

    def __init__(self, lang: str = "en") -> None:
        self.lang = lang

    def _engine(self) -> "paddleocr.PaddleOCR":
        if self.lang not in _ENGINE_CACHE:
            from paddleocr import PaddleOCR

            _ENGINE_CACHE[self.lang] = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang=self.lang,
            )
        return _ENGINE_CACHE[self.lang]

    def render_rotations(
        self, image: "Image.Image", n: int = 8
    ) -> list["Image.Image"]:
        """Render n evenly-spaced rotations of image for OCR voting."""
        if n <= 0:
            raise ValueError("n must be a positive integer")
        rgb = image.convert("RGB")
        rotations = []
        for i in range(n):
            angle = 360.0 * i / n
            if angle == 0.0:
                rotations.append(rgb)
            else:
                rotations.append(
                    rgb.rotate(
                        -angle, expand=True, fillcolor=(255, 255, 255),
                        resample=Image.BICUBIC,
                    )
                )
        return rotations

    def ocr(
        self, image: "Image.Image"
    ) -> tuple[str, float, list[tuple[float, float]]]:
        """Run PaddleOCR on one rendered image; returns (text, confidence,
        bbox corners). A cluster render can produce more than one detected
        text box (e.g. a multi-character run PaddleOCR's detector splits
        up) -- these are joined left-to-right into one string, with
        confidence averaged and the bbox corners covering all of them."""
        import numpy as np

        array = np.asarray(image.convert("RGB"))
        pages = self._engine().predict(array)
        if not pages:
            return "", 0.0, []

        page = pages[0]
        texts = list(page.get("rec_texts") or [])
        scores = list(page.get("rec_scores") or [])
        polys = list(page.get("rec_polys") or [])
        if not texts:
            return "", 0.0, []

        order = sorted(
            range(len(texts)),
            key=lambda i: float(np.min(polys[i][:, 0])) if i < len(polys) else i,
        )
        text = " ".join(texts[i] for i in order)
        confidence = (sum(scores) / len(scores)) if scores else 0.0

        available = [polys[i] for i in order if i < len(polys)]
        if available:
            all_points = np.concatenate(available)
            x0, y0 = float(all_points[:, 0].min()), float(all_points[:, 1].min())
            x1, y1 = float(all_points[:, 0].max()), float(all_points[:, 1].max())
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        else:
            corners = []

        return text, float(confidence), corners

    def combine_rotation_results(
        self, results: list[tuple[str, float, list]]
    ) -> tuple[str, float]:
        """Combine OCR results across rotations: group by text
        similarity, sum confidence per group, and return the text of the
        highest-total-confidence group (a majority of similar-but-imperfect
        readings outweighs one outlier's higher raw per-rotation
        confidence) -- with the *average* confidence within that winning
        group as the reported confidence."""
        groups: list[dict] = []

        for text, confidence, _bbox in results:
            normalized = _normalize(text)
            if not normalized:
                continue
            match = next(
                (
                    g for g in groups
                    if difflib.SequenceMatcher(None, normalized, g["normalized"]).ratio()
                    >= _SIMILARITY_THRESHOLD
                ),
                None,
            )
            if match is None:
                groups.append(
                    {
                        "normalized": normalized, "text": text,
                        "total_confidence": confidence, "best_confidence": confidence,
                        "count": 1,
                    }
                )
            else:
                match["total_confidence"] += confidence
                match["count"] += 1
                if confidence > match["best_confidence"]:
                    match["best_confidence"] = confidence
                    match["text"] = text

        if not groups:
            return "", 0.0

        best = max(groups, key=lambda g: g["total_confidence"])
        return best["text"], best["total_confidence"] / best["count"]

    def ocr_cluster(
        self,
        cluster: list[VectorPath] | RasterImage,
        page: Page,
        renderer: "rastervec.renderer.Renderer",
        dpi: int = 300,
        n_rotations: int = 8,
    ) -> TextVectorResult:
        """Shared entrypoint: render (via Renderer) + OCR across
        rotations + combine, for either a vector path cluster
        (list[VectorPath], via Renderer.render_vector_cluster -- the only
        case actually usable today) or a raster image region (RasterImage,
        via Renderer.render_raster_region, which still raises
        NotImplementedError until the Raster stage is built)."""
        if isinstance(cluster, list):
            image = renderer.render_vector_cluster(cluster, page, dpi)
            bbox = union_bbox([p.bbox for p in cluster])
        else:
            image = renderer.render_raster_region(cluster, dpi)
            bbox = cluster.bbox

        rotations = self.render_rotations(image, n_rotations)
        step = 360.0 / n_rotations

        results: list[tuple[str, float, list]] = []
        best_index, best_confidence = 0, -1.0
        for i, rotated in enumerate(rotations):
            result = self.ocr(rotated)
            results.append(result)
            if result[1] > best_confidence:
                best_index, best_confidence = i, result[1]

        text, confidence = self.combine_rotation_results(results)
        # The orientation whose own single-shot reading was most confident
        # -- not necessarily a member of the winning combined-text group,
        # but a simple, well-defined answer to "which rotation did this
        # come from".
        rotation_used = round(step * best_index) % 360

        return TextVectorResult(
            cluster_paths=cluster if isinstance(cluster, list) else [],
            text=text,
            confidence=confidence,
            bbox=bbox,
            rotation_used=rotation_used,
            page_index=page.meta.index,
        )
