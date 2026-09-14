"""Phase 1's raster-image output, consumed by Phase 2 (raster-to-vector backends)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class Image:
    """One raster image handed to a Phase 2 backend.

    `array` is a grayscale or RGB numpy array. `bbox` locates the image in
    the page's unrotated MediaBox space (see `commons/models/__init__.py`'s
    module docstring) so a Phase 2 backend can map any geometry it derives
    back to page coordinates. `source` distinguishes a whole-page render
    (`"page"`, produced at `dpi`) from a real embedded raster (`"embedded"`,
    `xref` set, `dpi` is the render dpi used to rasterize it for this array).
    """

    array: np.ndarray
    bbox: tuple[float, float, float, float]
    dpi: float
    source: Literal["page", "embedded"]
    xref: int | None = None
