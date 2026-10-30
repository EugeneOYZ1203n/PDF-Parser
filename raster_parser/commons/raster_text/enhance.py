"""Colour contrast + line enhancement for the shared P2 raster prelude, run
on the raw image after OCR and before the colour-layer split.

The image goes to LAB and only L (lightness) is touched, so hue and chroma
survive unchanged: CLAHE (contrast-limited adaptive histogram equalization)
lifts faint lines on low-contrast scans locally without blowing out the
whole page; an unsharp mask (`L + amount * (L - blur(L))`) then steepens
stroke edges. A small sigma keeps adjacent parallel lines (stair treads)
from haloing into each other. Run once on the whole image, so CLAHE's tiles
see real neighbourhoods."""

from __future__ import annotations

import cv2
import numpy as np

from raster_parser.commons.raster_text.config import CLAHE_CLIP, CLAHE_GRID, USM_AMOUNT, USM_SIGMA


def clahe(
    gray: np.ndarray, clip: float = CLAHE_CLIP, grid: tuple[int, int] = CLAHE_GRID
) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(
        np.asarray(gray, dtype=np.uint8)
    )


def unsharp_mask(
    gray: np.ndarray, sigma: float = USM_SIGMA, amount: float = USM_AMOUNT
) -> np.ndarray:
    g = np.asarray(gray, dtype=np.float32)
    blurred = cv2.GaussianBlur(g, (0, 0), sigma)
    return np.clip(g + amount * (g - blurred), 0, 255).astype(np.uint8)


def clahe_lab(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CLAHE then unsharp mask on LAB L only. Returns `(enhanced_rgb, lab)`,
    both uint8 HxWx3 -- `lab` is the *enhanced* image in OpenCV's 8-bit LAB
    (L 0..255, a/b centred on 128)."""
    lab = cv2.cvtColor(np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8), cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = unsharp_mask(clahe(lab[:, :, 0]))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), lab


def to_gray(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    return cv2.cvtColor(arr[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
