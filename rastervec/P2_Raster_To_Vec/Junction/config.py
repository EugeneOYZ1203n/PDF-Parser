"""Tunable thresholds for the Junction P2 backend's own stages (per-color,
per-component tracing -> vector/ink diff). OCR, colour CLAHE, colour layers
and text removal are the shared raster prelude's -- see
`RasterText/config.py`. The tracing stage's own knobs stay on
`junction_test.pipeline.Params`."""
from __future__ import annotations

# ======================================================================
# Per-color tracing (components.py)
# ======================================================================

# Ink pixels of one color layer closer than this (page points) belong to
# the same component and are traced together; components are traced one at
# a time, each on its own crop.
COMPONENT_TOLERANCE_PT = 10.0

# ======================================================================
# Vector/ink diff debug (diff.py)
# ======================================================================

# A vector pixel within this many px of ink (and vice versa) counts as
# matched rather than spurious/missed.
DIFF_TOLERANCE_PX = 2
