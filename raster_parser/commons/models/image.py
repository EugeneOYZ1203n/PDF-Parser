"""Phase 1's raster-image output, consumed by Phase 2 (raster-to-vector backends)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class Image:
    """One embedded raster image handed to a Phase 2 backend.

    `array` is a grayscale or RGB numpy array at the image's own native
    pixel resolution. `bbox` locates the placement in the page's unrotated
    MediaBox space (see `commons/models/__init__.py`'s module docstring) so
    a Phase 2 backend can map any geometry it derives back to page
    coordinates. `transform` is the placement's image matrix as PyMuPDF's
    `get_image_info()["transform"]` reports it -- the 6-tuple `(a, b, c, d,
    e, f)` mapping the image's unit square (`(0, 0)` = pixel top-left,
    `(1, 1)` = bottom-right) into that same page space -- `None` when
    unknown, in which case an upright placement filling `bbox` is assumed.
    `dpi` is the placement's effective resolution (pixels per inch along
    `bbox`'s width). `source` is always `"embedded"` now that Phase 1 no
    longer renders the whole page; `xref` is the image XObject.
    """

    array: np.ndarray
    bbox: tuple[float, float, float, float]
    dpi: float
    source: Literal["embedded"]
    xref: int | None = None
    transform: tuple[float, float, float, float, float, float] | None = None
