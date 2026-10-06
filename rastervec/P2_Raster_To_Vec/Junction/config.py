"""Tunable thresholds for the Junction P2 backend's pre-tracing stages
(color separation -> tiled OCR -> text removal -> enhancement -> per-color,
per-component tracing -> vector/ink diff). The tracing stage's own knobs
stay on `junction_test.pipeline.Params`.

Self-contained -- the OCR padding/deskew values below are *copied* from
`P3_Vector_Parsing/OldVectorClassification/config.py` + `paddle_engine.py`
(not imported), per CLAUDE.md's "sibling backends share zero code" rule;
keep them in step by hand if you want the two backends to pad identically."""
from __future__ import annotations

# ======================================================================
# Color separation (color_separation.py) -- DBSCAN over weighted unique
# quantized colors in HSV cone space (S*V*cosH, S*V*sinH, V), V in [0, 1].
# ======================================================================

# Levels per RGB channel before de-duplicating colors (32 -> at most 32^3
# unique colors, so DBSCAN never sees more than ~33k points).
COLOR_QUANT_LEVELS = 32
# DBSCAN neighbourhood radius, in cone units (V spans 0..1).
DBSCAN_EPS = 0.08
# DBSCAN min_samples, counted in *pixels* (sample_weight = pixel count per
# unique color): a color family needs this many pixels to seed a cluster.
DBSCAN_MIN_PIXELS = 200
# A cluster whose members span more than this much Value (0..1) is split in
# two at a weighted Otsu threshold on V (repeatedly, up to
# VALUE_SPLIT_MAX_DEPTH). Guards against DBSCAN's density chaining: on a
# blurred scan the antialiased mid-grays between black ink and white paper
# are dense enough to be core points, which would otherwise fuse ink and
# paper into one layer.
VALUE_SPLIT_RANGE = 0.5
VALUE_SPLIT_MAX_DEPTH = 2

# ======================================================================
# Tiled OCR (text_ocr.py)
# ======================================================================

# 960 = PaddleOCR's own default det_limit_side_len, so a tile is never
# resized by the detector. 20% overlap (192 px) exceeds one text line's
# height at 150-300 dpi, so every line appears whole in at least one tile.
OCR_TILE_PX = 960
OCR_TILE_OVERLAP_FRAC = 0.20
# Two boxes from *different* tiles merge when they intersect or come within
# this many pixels of each other.
TILE_MERGE_GAP_PX = 2

# Pad 1 (before the second, per-box detect) -- OldVectorClassification's
# RENDER_PADDING_EXTRA_PT, in page points, converted with each image's own
# px-per-pt scale.
RENDER_PADDING_EXTRA_PT = 15.0
# The padded box is upscaled until its short side is at least this many px
# (OldVectorClassification's MIN_RENDER_SIDE_PX / dynamic-dpi analogue), but
# never by more than MAX_UPSCALE (the MAX_RENDER_DPI analogue: 4800 / 300 dpi
# would be 16x; 8x is plenty for a raster that has no more detail to give).
MIN_RENDER_SIDE_PX = 100
MAX_UPSCALE = 8.0
# Pad 2 (before recognition) -- OldVectorClassification/paddle_engine.py's
# _CROP_EXPAND_FRACTION / _CROP_BORDER_PX.
CROP_EXPAND_FRACTION = 0.05
CROP_BORDER_PX = 5

# Raster-refined rotation (paddle_engine.py::hough_deskew) -- copied values.
HOUGH_INK_THRESHOLD = 200
HOUGH_DILATE_RADIUS_PX = 2
MINAREA_INK_THRESHOLD = 200
HOUGH_ANGLE_SNAP_DEG = 10.0

OCR_VERSION = "PP-OCRv4"
OCR_LANG = "en"
OCR_BATCH_SIZE = 128
DETECT_LIMIT_SIDE_LEN = 4000

# ======================================================================
# Text removal (text_removal.py)
# ======================================================================

# Dilation (px) of a recognized box's ink-label mask before erasing, to
# catch antialiased fringe pixels.
ERASE_DILATE_PX = 1

# ======================================================================
# Enhancement (enhance.py)
# ======================================================================

CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
USM_SIGMA = 1.0
USM_AMOUNT = 1.5

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
