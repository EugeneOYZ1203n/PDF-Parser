"""Own PaddleOCR engine wrappers -- same API surface as the other two P3
variants' own paddle_engine.py copies (`PaddleRecBackend.recognize_crops`,
`PaddleDetectBackend`), but not shared code: this is LegacyRecreation's own
independent copy, trimmed to what this variant actually needs -- no
Segment-specific batching, no `ClusterDetection`/whole-image-rotate
machinery (this backend renders each word group once and has no Radon
refine stage; `parse.py` detects and recognizes directly against that one
render)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from skimage.transform import ProjectiveTransform, warp

from rastervec.P3_Vector_Parsing.LegacyRecreation.config import OCR_LANG, OCR_VERSION

# PaddleOCR's own DB detector's det_limit_side_len -- a per-word-group render
# is rarely anywhere near this, so it's a generous ceiling rather than a
# tuned value (same constant/rationale as FastIntoPaddle's own copy).
_DETECT_LIMIT_SIDE_LEN = 4000


@dataclass
class OcrBox:
    text: str
    confidence: float
    flip_deg: int = 0


class PaddleRecBackend:
    _ENGINE_CACHE: dict[tuple[str, str], object] = {}

    def __init__(self, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        self.key = (ocr_version, lang)

    @classmethod
    def warmup(cls, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        cls(ocr_version, lang)._engine()

    def _engine(self):
        if self.key not in PaddleRecBackend._ENGINE_CACHE:
            import torch  # noqa: F401 -- must import before paddle on Windows
            from paddleocr import PaddleOCR

            ocr_version, lang = self.key
            PaddleRecBackend._ENGINE_CACHE[self.key] = PaddleOCR(
                ocr_version=ocr_version, lang=lang, use_angle_cls=True, show_log=False,
            )
        return PaddleRecBackend._ENGINE_CACHE[self.key]

    def recognize_crops(self, crops: list[np.ndarray]) -> list[OcrBox]:
        if not crops:
            return []
        bgr = [_normalize_bgr(c) for c in crops]
        engine = self._engine()
        _, cls_results, _ = engine.text_classifier(bgr)
        flips = [180 if str(label) == "180" else 0 for label, _score in cls_results]
        upright = [np.rot90(img, 2) if flip else img for img, flip in zip(bgr, flips)]
        rec = engine.text_recognizer(upright)
        rows = rec[0] if isinstance(rec, tuple) else rec
        out: list[OcrBox] = []
        for row, flip in zip(rows, flips):
            text = str(row[0] or "").strip()
            score = float(row[1] or 0.0)
            out.append(OcrBox(text=text, confidence=score if text else 0.0, flip_deg=flip))
        return out


class PaddleDetectBackend:
    """PaddleOCR's own text-DETECTION model (`engine.text_detector`, PP-OCR's
    DB detector), run against an already-rendered+padded word-group image.
    Cached separately from `PaddleRecBackend` (its own `_ENGINE_CACHE`),
    built with a generous `det_limit_side_len`. Unlike FastIntoPaddle's
    `PaddleDetectBackend.detect_on_cluster` (which renders/pads/maps to page
    space itself), this is a bare `detect` -- `parse.py` already renders and
    pads the group's own image once and only needs the raw detected quads
    (padded-render pixel space) back, since it drives its own crop+recognize
    loop directly."""

    _ENGINE_CACHE: dict[tuple[str, str], object] = {}

    def __init__(self, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        self.key = (ocr_version, lang)

    @classmethod
    def warmup(cls, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        cls(ocr_version, lang)._engine()

    def _engine(self):
        if self.key not in PaddleDetectBackend._ENGINE_CACHE:
            import torch  # noqa: F401 -- must import before paddle on Windows
            from paddleocr import PaddleOCR

            ocr_version, lang = self.key
            PaddleDetectBackend._ENGINE_CACHE[self.key] = PaddleOCR(
                ocr_version=ocr_version, lang=lang, use_angle_cls=True, show_log=False,
                det_limit_side_len=_DETECT_LIMIT_SIDE_LEN,
            )
        return PaddleDetectBackend._ENGINE_CACHE[self.key]

    def detect(self, bgr: np.ndarray) -> list[np.ndarray]:
        """Every text quad PaddleOCR's detector finds in `bgr`, each a
        `(4, 2)` array of `(x, y)` pixel points in `bgr`'s own pixel space,
        PaddleOCR's own corner order (clockwise from top-left)."""
        dt_boxes, _elapse = self._engine().text_detector(bgr)
        return [np.asarray(quad, dtype=np.float64) for quad in dt_boxes]


def _normalize_rotation(angle_deg: float) -> float:
    """Wrap to `[-90, 90)` -- text direction is a line, not an arrow, so a
    0/180 ambiguity always remains mod 180 (resolved separately by the
    cls-flip term this is added to before calling this). Own duplicated copy
    of FastIntoPaddle/VectorClassification's `paddle_engine.py::
    _normalize_rotation`."""
    return ((angle_deg + 90.0) % 180.0) - 90.0


def _quad_rotation_deg(quad: np.ndarray) -> float:
    """The quad's own dominant-edge orientation, as the rotation (degrees,
    counter-clockwise-positive) that would bring that edge to horizontal.
    `quad` is 4 `(x, y)` pixel points in PaddleOCR's own corner order
    (clockwise from top-left); the longer of the top edge (0->1) and left
    edge (0->3) is taken as the text's own baseline direction, so this is
    robust to a quad that's taller than it is wide (vertical/rotated text).
    Own duplicated copy of FastIntoPaddle/VectorClassification's
    `paddle_engine.py::_quad_rotation_deg`."""
    top = quad[1] - quad[0]
    left = quad[3] - quad[0]
    dx, dy = (top if np.hypot(*top) >= np.hypot(*left) else left)
    theta = math.degrees(math.atan2(dy, dx))
    return _normalize_rotation(-theta)


def _rotate_crop(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-correct crop of `quad` (4 `(x, y)` pixel points,
    PaddleOCR's own corner order) out of `bgr`, warped onto an axis-aligned
    rectangle sized to the quad's own edge lengths -- the standard PaddleOCR
    "get_rotate_crop_image" step, reimplemented with `skimage` (no cv2, per
    this project's convention). Own duplicated copy of FastIntoPaddle/
    VectorClassification's `paddle_engine.py::_rotate_crop`."""
    width = max(1, int(round(max(np.hypot(*(quad[1] - quad[0])), np.hypot(*(quad[2] - quad[3]))))))
    height = max(1, int(round(max(np.hypot(*(quad[3] - quad[0])), np.hypot(*(quad[2] - quad[1]))))))
    rect = np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float64)
    tf = ProjectiveTransform()
    tf.estimate(rect, quad)
    warped = warp(bgr, tf, output_shape=(height, width), cval=255.0, preserve_range=True)
    return np.clip(warped, 0, 255).astype(np.uint8)


def _normalize_bgr(crop: np.ndarray) -> np.ndarray:
    arr = np.asarray(crop, dtype=np.uint8)
    if arr.ndim == 2:
        return np.ascontiguousarray(np.repeat(arr[:, :, None], 3, axis=2))
    return np.ascontiguousarray(arr[:, :, :3][:, :, ::-1])
