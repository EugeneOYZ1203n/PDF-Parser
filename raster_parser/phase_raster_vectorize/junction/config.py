"""Tunable thresholds for the Junction raster_vectorize backend's own stages
(per-color, per-component tracing -> vector/ink diff). OCR, colour CLAHE, colour layers and text
removal are the shared raster prelude's -- see `commons/raster_text/config.py`. The tracing
stage's own knobs are on `models.junction.Params`."""

from __future__ import annotations

# ======================================================================
# Per-color tracing (components.py)
# ======================================================================

# Ink pixels of one color layer closer than this (page points) belong to
# the same component and are traced together; components are traced one at
# a time, each on its own crop.
COMPONENT_TOLERANCE_PT = 10.0

# ======================================================================
# Solid fills (tracer/fills.py)
# ======================================================================

# Ink at least this wide (page points) is a solid fill: only its outline is
# kept, it isn't skeletonized. Keep it >= sqrt(2) x the widest real line --
# the centre of an X crossing is that much deeper inside the ink than either
# line, so a smaller value turns thick-line crossings into "fills".
FILL_MIN_WIDTH_PT = 3.5
# Floor in px, for low-resolution images (at 72 dpi 3.5 pt is only 3.5 px).
FILL_MIN_WIDTH_FLOOR_PX = 6.0

# ======================================================================
# Vector/ink diff debug (diff.py)
# ======================================================================

# A vector pixel within this many px of ink (and vice versa) counts as
# matched rather than spurious/missed.
DIFF_TOLERANCE_PX = 2
