"""Masking helper: interface-only stub for a future phase.

Not implemented yet; method bodies intentionally raise NotImplementedError.
No image-processing library imports yet -- deferred to when this is
actually implemented.
"""
from __future__ import annotations

from typing import Callable


class Masking:
    """Image masking utilities used by the Raster stage's text-masking
    and line-remainder steps."""

    def mask_bboxes(
        self,
        image: "np.ndarray",
        bboxes: list[tuple[float, float, float, float]],
        color_select: Callable[["np.ndarray"], "np.ndarray"],
    ) -> "np.ndarray":
        """Mask out the given bboxes, selecting only pixels matching
        color_select within each box (e.g. the OCR'd text's color)."""
        raise NotImplementedError

    def dilate_mask(self, mask: "np.ndarray", amount: float) -> "np.ndarray":
        """Slightly expand a binary mask (e.g. to fully cover a line's
        stroke width before masking it out of the remainder image)."""
        raise NotImplementedError
