"""The model's input: one color layer as a binary gray mask.

Kept apart from `adapter.py` (which imports the OCR modules) so
`prep_dataset.py` can share it without importing any OCR code."""
from __future__ import annotations

import cv2
import numpy as np


def layer_image(labels: np.ndarray, layer: int, scale: float) -> np.ndarray:
    """One color layer as a gray mask (ink 0, everything else 255),
    resampled to the canonical scale (area-averaged when shrinking, so thin
    lines stay as gray rather than vanishing). Shared by inference and
    `prep_dataset.py` so the model always sees the same kind of input."""
    gray = np.where(labels == layer, 0, 255).astype(np.uint8)
    if abs(scale - 1.0) < 0.02:
        return gray
    h, w = gray.shape
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(gray, size, interpolation=interp)
