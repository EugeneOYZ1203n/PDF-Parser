"""The raster prelude's OCR plumbing: picklable job wrappers around the shared engine
(`commons.paddle_engine.PaddleBackend`), plus the raster-refined deskew used before
recognition. `hough_deskew` crops
a quad axis-aligned (expanded `CROP_EXPAND_FRACTION`, bordered
`CROP_BORDER_PX`) and rotates it by a Hough + `cv2.minAreaRect` combined
angle snapped to `HOUGH_ANGLE_SNAP_DEG`. The crop border is filled with a caller-given background color, since a
scanned raster's background is rarely pure white.

The `_*_job` functions are top-level and picklable for pool dispatch (`compute`)."""

from __future__ import annotations

import math

import cv2
import numpy as np
from skimage.color import rgb2gray

from raster_parser.commons.paddle_engine import OcrBox, PaddleBackend  # noqa: F401 -- re-exported
from skimage.morphology import binary_dilation, disk
from skimage.transform import hough_line, hough_line_peaks, rotate

from raster_parser.commons.raster_text.config import (
    CROP_BORDER_PX,
    CROP_EXPAND_FRACTION,
    HOUGH_ANGLE_SNAP_DEG,
    HOUGH_DILATE_RADIUS_PX,
    HOUGH_INK_THRESHOLD,
    MINAREA_INK_THRESHOLD,
    OCR_LANG,
    OCR_VERSION,
)


# --------------------------------------------------------------------------
# Picklable Pool-2 jobs (plain numpy in and out).
# --------------------------------------------------------------------------
def _detect_job(
    bgr: np.ndarray, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG
) -> list[np.ndarray]:
    return PaddleBackend(ocr_version, lang).detect(bgr)


def _recognize_crops_job(
    crops: list[np.ndarray],
    ocr_version: str = OCR_VERSION,
    lang: str = OCR_LANG,
) -> list[OcrBox]:
    return PaddleBackend(ocr_version, lang).recognize_crops(crops)


def _recognize_crops_raw_job(
    crops: list[np.ndarray],
    ocr_version: str = OCR_VERSION,
    lang: str = OCR_LANG,
) -> list[OcrBox]:
    return PaddleBackend(ocr_version, lang).recognize_crops_raw(crops)


# --------------------------------------------------------------------------
# Deskew.
# --------------------------------------------------------------------------
def normalize_rotation(angle_deg: float) -> float:
    """Wrap to `[-90, 90)`."""
    return ((angle_deg + 90.0) % 180.0) - 90.0


def axis_aligned_crop(
    bgr: np.ndarray,
    quad: np.ndarray,
    border_bgr: tuple[int, int, int] = (255, 255, 255),
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
        cropped,
        CROP_BORDER_PX,
        CROP_BORDER_PX,
        CROP_BORDER_PX,
        CROP_BORDER_PX,
        cv2.BORDER_CONSTANT,
        value=tuple(int(c) for c in border_bgr),
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


# Angles mod 90 are averaged as unit vectors at 4x the angle (one 90-degree period -> one turn).
_MOD90_TO_TURN = 4.0


def _circular_avg_mod90(a_deg: float, b_deg: float) -> float:
    a = math.radians(a_deg * _MOD90_TO_TURN)
    b = math.radians(b_deg * _MOD90_TO_TURN)
    x = math.cos(a) + math.cos(b)
    y = math.sin(a) + math.sin(b)
    if abs(x) < 1e-9 and abs(y) < 1e-9:  # opposite directions: no mean, take the plain average
        return ((a_deg + b_deg) / 2.0) % 90.0
    return round(math.degrees(math.atan2(y, x)) / _MOD90_TO_TURN, 9) % 90.0


def _combined_rotation_deg(hough: "float | None", minarea: "float | None") -> float:
    if hough is not None and minarea is not None:
        combined_mod = _circular_avg_mod90(hough % 90.0, minarea % 90.0)
    elif hough is not None:
        combined_mod = hough % 90.0
    elif minarea is not None:
        combined_mod = minarea % 90.0
    else:
        return 0.0
    signed = combined_mod - 90.0 if combined_mod > 45.0 else combined_mod  # 0..90 -> -45..45
    return round(signed / HOUGH_ANGLE_SNAP_DEG) * HOUGH_ANGLE_SNAP_DEG


def hough_deskew(
    bgr: np.ndarray,
    quad: np.ndarray,
    border_bgr: tuple[int, int, int] = (255, 255, 255),
) -> "tuple[np.ndarray, float]":
    """`(rotated BGR crop, combined angle deg)` for `quad` -- see the module
    docstring. The crop is ink-thresholded on luminance, so it assumes dark
    ink on a lighter background (the common case for drawings)."""
    base = axis_aligned_crop(bgr, quad, border_bgr)
    gray = rgb2gray(base[:, :, ::-1])
    hough_mask = binary_dilation(
        gray < HOUGH_INK_THRESHOLD / 255.0, footprint=disk(HOUGH_DILATE_RADIUS_PX)
    )
    minarea_mask = gray < MINAREA_INK_THRESHOLD / 255.0
    combined = _combined_rotation_deg(
        _hough_angle_deg(hough_mask), _minarea_angle_deg(minarea_mask)
    )
    if combined == 0.0:
        return base, 0.0
    rotated = np.stack(
        [
            rotate(
                base[:, :, ch],
                combined,
                resize=True,
                cval=float(border_bgr[ch]),
                order=1,
                preserve_range=True,
            )
            for ch in range(3)
        ],
        axis=2,
    ).astype(np.uint8)
    return rotated, combined
