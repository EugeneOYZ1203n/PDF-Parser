from __future__ import annotations

import cv2
import numpy as np


def dilate(
    binary: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """
    Dilate a binary image using an elliptical kernel.

    Args:
        binary:
            Binary image. Foreground should be non-zero.

        kernel_size:
            Size of the elliptical dilation kernel.

        iterations:
            Number of dilation iterations.

    Returns:
        Dilated binary image.
    """

    if kernel_size <= 1 or iterations <= 0:
        return binary.copy()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    return cv2.dilate(
        binary,
        kernel,
        iterations=iterations,
    )