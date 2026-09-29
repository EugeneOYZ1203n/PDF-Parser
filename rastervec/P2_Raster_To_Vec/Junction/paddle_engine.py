"""Junction's own PaddleOCR detect + recognize pair, plus the raster-refined
deskew used before recognition.

An independent duplicate of `P3_Vector_Parsing/VectorClassification/
paddle_engine.py` (per CLAUDE.md's "sibling backends share zero code" rule
-- P2 must not import a P3 backend): `PaddleDetectBackend.detect` returns
text quads in the given image's pixel space; `PaddleRecBackend.
recognize_crops` classifies 0/180 then recognizes; `recognize_crops_raw`
recognizes as-is (the +90/180/270 blank-retry sweep); `hough_deskew` crops
a quad axis-aligned (expanded `CROP_EXPAND_FRACTION`, bordered
`CROP_BORDER_PX`) and rotates it by a Hough + `cv2.minAreaRect` combined
angle snapped to `HOUGH_ANGLE_SNAP_DEG`. One difference from the P3 copy:
the crop border is filled with a caller-given background color, since a
scanned raster's background is rarely pure white.

Engines are cached at class scope per `(ocr_version, lang)`. The `_*_job`
functions are top-level and picklable for Pool-2 dispatch (`compute`)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.morphology import binary_dilation, disk
from skimage.transform import hough_line, hough_line_peaks, rotate

from rastervec.P2_Raster_To_Vec.Junction.config import (
    CROP_BORDER_PX,
    CROP_EXPAND_FRACTION,
    DETECT_LIMIT_SIDE_LEN,
    HOUGH_ANGLE_SNAP_DEG,
    HOUGH_DILATE_RADIUS_PX,
    HOUGH_INK_THRESHOLD,
    MINAREA_INK_THRESHOLD,
    OCR_BATCH_SIZE,
    OCR_LANG,
    OCR_VERSION,
)


@dataclass
class OcrBox:
    """One recognised crop: text, confidence, and the classifier's own 0/180
    flip decision (degrees)."""

    text: str
    confidence: float
    flip_deg: int = 0


def _rows_to_boxes(rows, flips) -> list[OcrBox]:
    out: list[OcrBox] = []
    for row, flip in zip(rows, flips):
        text = str(row[0] or "").strip()
        score = float(row[1] or 0.0)
        out.append(OcrBox(text=text, confidence=score if text else 0.0, flip_deg=flip))
    return out


class PaddleRecBackend:
    """paddleocr 2.x angle classification + recognition."""

    _ENGINE_CACHE: dict[tuple[str, str], object] = {}

    def __init__(self, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        self.key = (ocr_version, lang)

    @classmethod
    def warmup(cls, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        cls(ocr_version, lang)._engine()

    def _engine(self):
        if self.key not in PaddleRecBackend._ENGINE_CACHE:
            import torch  # noqa: F401 -- must import before paddle on Windows (shm.dll)
            from paddleocr import PaddleOCR

            ocr_version, lang = self.key
            PaddleRecBackend._ENGINE_CACHE[self.key] = PaddleOCR(
                ocr_version=ocr_version, lang=lang, use_angle_cls=True, show_log=False,
                rec_batch_num=OCR_BATCH_SIZE, cls_batch_num=OCR_BATCH_SIZE,
            )
        return PaddleRecBackend._ENGINE_CACHE[self.key]

    def recognize_crops(self, crops: list[np.ndarray]) -> list[OcrBox]:
        """Classify each BGR crop 0/180, flip the 180s, recognise the batch."""
        if not crops:
            return []
        bgr = [np.ascontiguousarray(c) for c in crops]
        engine = self._engine()
        _, cls_results, _ = engine.text_classifier(bgr)
        flips = [180 if str(label) == "180" else 0 for label, _score in cls_results]
        upright = [np.rot90(img, 2) if flip else img for img, flip in zip(bgr, flips)]
        rec = engine.text_recognizer(upright)
        return _rows_to_boxes(rec[0] if isinstance(rec, tuple) else rec, flips)

    def recognize_crops_raw(self, crops: list[np.ndarray]) -> list[OcrBox]:
        """Recognise BGR `crops` as-is (no classifier) -- the retry sweep."""
        if not crops:
            return []
        rec = self._engine().text_recognizer([np.ascontiguousarray(c) for c in crops])
        rows = rec[0] if isinstance(rec, tuple) else rec
        return _rows_to_boxes(rows, [0] * len(crops))


class PaddleDetectBackend:
    """PaddleOCR's DB text detector over a BGR image."""

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
                det_limit_side_len=DETECT_LIMIT_SIDE_LEN,
            )
        return PaddleDetectBackend._ENGINE_CACHE[self.key]

    def detect(self, bgr: np.ndarray) -> list[np.ndarray]:
        """Every text quad in `bgr`, each `(4, 2)` float pixel points
        (PaddleOCR's own clockwise-from-top-left order)."""
        dt_boxes, _elapse = self._engine().text_detector(np.ascontiguousarray(bgr))
        if dt_boxes is None:
            return []
        return [np.asarray(quad, dtype=np.float64) for quad in dt_boxes]


# --------------------------------------------------------------------------
# Picklable Pool-2 jobs (plain numpy in and out).
# --------------------------------------------------------------------------
def _detect_job(bgr: np.ndarray, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> list[np.ndarray]:
    return PaddleDetectBackend(ocr_version, lang).detect(bgr)


def _recognize_crops_job(
    crops: list[np.ndarray], ocr_version: str = OCR_VERSION, lang: str = OCR_LANG,
) -> list[OcrBox]:
    return PaddleRecBackend(ocr_version, lang).recognize_crops(crops)


def _recognize_crops_raw_job(
    crops: list[np.ndarray], ocr_version: str = OCR_VERSION, lang: str = OCR_LANG,
) -> list[OcrBox]:
    return PaddleRecBackend(ocr_version, lang).recognize_crops_raw(crops)


# --------------------------------------------------------------------------
# Deskew (duplicated from VectorClassification/paddle_engine.py).
# --------------------------------------------------------------------------
def normalize_rotation(angle_deg: float) -> float:
    """Wrap to `[-90, 90)`."""
    return ((angle_deg + 90.0) % 180.0) - 90.0


def axis_aligned_crop(
    bgr: np.ndarray, quad: np.ndarray, border_bgr: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Axis-aligned crop of `quad` expanded `CROP_EXPAND_FRACTION` about its
    centroid, clamped to `bgr`, plus a `CROP_BORDER_PX` border of
    `border_bgr` (pad 2)."""
    centroid = quad.mean(axis=0)
    expanded = centroid + (quad - centroid) * (1.0 + CROP_EXPAND_FRACTION)
    h, w = bgr.shape[:2]
    x0 = max(0, int(math.floor(expanded[:, 0].min())))
    y0 = max(0, int(math.floor(expanded[:, 1].min())))
    x1 = min(w, int(math.ceil(expanded[:, 0].max())))
    y1 = min(h, int(math.ceil(expanded[:, 1].max())))
    if x1 <= x0 or y1 <= y0:
        cropped = np.empty((1, 1, 3), dtype=np.uint8)
        cropped[:] = border_bgr
    else:
        cropped = np.asarray(bgr[y0:y1, x0:x1], dtype=np.uint8)
    return cv2.copyMakeBorder(
        cropped, CROP_BORDER_PX, CROP_BORDER_PX, CROP_BORDER_PX, CROP_BORDER_PX,
        cv2.BORDER_CONSTANT, value=tuple(int(c) for c in border_bgr),
    )


def _hough_angle_deg(mask: np.ndarray) -> "float | None":
    if not mask.any():
        return None
    thetas = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)
    _accum, angles, _dists = hough_line_peaks(*hough_line(mask, theta=thetas), num_peaks=1)
    if len(angles) == 0:
        return None
    return normalize_rotation(math.degrees(angles[0]) - 90.0)


def _minarea_angle_deg(mask: np.ndarray) -> "float | None":
    if not mask.any():
        return None
    points = cv2.findNonZero(np.asarray(mask, dtype=np.uint8))
    if points is None or len(points) < 2:
        return None
    _center, _size, angle = cv2.minAreaRect(points)
    return float(angle)


def _circular_avg_mod90(a_deg: float, b_deg: float) -> float:
    a = math.radians(a_deg * 4.0)
    b = math.radians(b_deg * 4.0)
    x = math.cos(a) + math.cos(b)
    y = math.sin(a) + math.sin(b)
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return ((a_deg + b_deg) / 2.0) % 90.0
    return round(math.degrees(math.atan2(y, x)) / 4.0, 9) % 90.0


def _combined_rotation_deg(hough: "float | None", minarea: "float | None") -> float:
    if hough is not None and minarea is not None:
        combined_mod = _circular_avg_mod90(hough % 90.0, minarea % 90.0)
    elif hough is not None:
        combined_mod = hough % 90.0
    elif minarea is not None:
        combined_mod = minarea % 90.0
    else:
        return 0.0
    signed = combined_mod - 90.0 if combined_mod > 45.0 else combined_mod
    return round(signed / HOUGH_ANGLE_SNAP_DEG) * HOUGH_ANGLE_SNAP_DEG


def hough_deskew(
    bgr: np.ndarray, quad: np.ndarray, border_bgr: tuple[int, int, int] = (255, 255, 255),
) -> "tuple[np.ndarray, float]":
    """`(rotated BGR crop, combined angle deg)` for `quad` -- see the module
    docstring. The crop is ink-thresholded on luminance, so it assumes dark
    ink on a lighter background (the common case for drawings)."""
    base = axis_aligned_crop(bgr, quad, border_bgr)
    gray = rgb2gray(base[:, :, ::-1])
    hough_mask = binary_dilation(gray < HOUGH_INK_THRESHOLD / 255.0, footprint=disk(HOUGH_DILATE_RADIUS_PX))
    minarea_mask = gray < MINAREA_INK_THRESHOLD / 255.0
    combined = _combined_rotation_deg(_hough_angle_deg(hough_mask), _minarea_angle_deg(minarea_mask))
    if combined == 0.0:
        return base, 0.0
    rotated = np.stack([
        rotate(base[:, :, ch], combined, resize=True, cval=float(border_bgr[ch]),
               order=1, preserve_range=True)
        for ch in range(3)
    ], axis=2).astype(np.uint8)
    return rotated, combined
