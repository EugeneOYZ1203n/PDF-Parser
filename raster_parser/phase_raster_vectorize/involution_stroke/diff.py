"""Step 10 (debug): compare the traced vectors back against the ink they
were traced from, pixel by pixel.

`diff_codes` rasterizes a component's polylines at 1 px
(`render_geometry`) and compares them with that component's binarized, text-removed ink, each
side dilated by `tol_px` so a 1-2 px width/position slop isn't flagged:

    0  background        (neither)
    1  matched           (ink covered by a vector)
    2  missed            (ink with no vector within tol -- a line the tracer lost)
    3  spurious          (vector with no ink within tol -- an invented line)

Codes 2 are the raw material for a future missed-line recovery pass (e.g.
re-running LSD/Hough on just that mask). `codes_to_rgba` colours them for
the `vector_diff` debug layers; `rendered_to_rgba` does the plain
`vector_render` layer."""

from __future__ import annotations

import numpy as np

from raster_parser.commons.raster_text.debug_tools import BACKGROUND, MATCHED, MISSED, SPURIOUS  # noqa: F401 -- shared
from raster_parser.commons.raster_text.debug_tools import (  # noqa: F401 -- moved to commons; still reachable from this module
    diff_codes,
)

_PALETTE = np.array(
    [
        (0, 0, 0, 0),  # background: transparent
        (160, 160, 160, 110),  # matched: faint gray
        (220, 38, 38, 255),  # missed: red
        (37, 99, 235, 255),  # spurious: blue
    ],
    dtype=np.uint8,
)


def codes_to_rgba(codes: np.ndarray) -> np.ndarray:
    return _PALETTE[codes]
