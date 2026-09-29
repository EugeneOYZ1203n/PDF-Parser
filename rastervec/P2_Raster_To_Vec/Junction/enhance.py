"""Step 8 of Junction's pre-tracing flow: contrast + line enhancement on the
text-removed grayscale image, before per-color tracing.

CLAHE (contrast-limited adaptive histogram equalization) lifts faint lines
on low-contrast scans locally without blowing out the whole page; an
unsharp mask (`g + amount * (g - blur(g))`) then steepens stroke edges. A
small sigma keeps adjacent parallel lines (stair treads) from haloing into
each other. Run once on the whole image, so CLAHE's tiles see real
neighbourhoods rather than a mostly-white per-layer image."""
from __future__ import annotations

import cv2
import numpy as np

from rastervec.P2_Raster_To_Vec.Junction.config import CLAHE_CLIP, CLAHE_GRID, USM_AMOUNT, USM_SIGMA


def clahe(gray: np.ndarray, clip: float = CLAHE_CLIP, grid: tuple[int, int] = CLAHE_GRID) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(np.asarray(gray, dtype=np.uint8))


def unsharp_mask(gray: np.ndarray, sigma: float = USM_SIGMA, amount: float = USM_AMOUNT) -> np.ndarray:
    g = np.asarray(gray, dtype=np.float32)
    blurred = cv2.GaussianBlur(g, (0, 0), sigma)
    return np.clip(g + amount * (g - blurred), 0, 255).astype(np.uint8)


def enhance(gray: np.ndarray) -> np.ndarray:
    """CLAHE then unsharp mask; same shape, uint8."""
    return unsharp_mask(clahe(gray))


def to_gray(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    return cv2.cvtColor(arr[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
