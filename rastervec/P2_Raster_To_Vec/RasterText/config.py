"""Tunables for the shared P2 raster prelude (`raster_text.prepare_images`):
tiled OCR -> colour CLAHE -> simple colour layers -> text removal.

The OCR padding/deskew values are *copied* from
`P3_Vector_Parsing/OldVectorClassification/config.py` + `paddle_engine.py`
(not imported -- P2 may not import P3); keep them in step by hand if you
want the two to pad identically."""
from __future__ import annotations

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
# Colour enhancement (enhance.py) -- CLAHE + unsharp mask on LAB L only,
# so hue and chroma are untouched.
# ======================================================================

CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
USM_SIGMA = 1.0
USM_AMOUNT = 1.5

# ======================================================================
# Colour layers (color_layers.py) -- chroma / hue in OpenCV 8-bit LAB
# units (a, b centred on 128, so chroma = hypot(a - 128, b - 128)).
# ======================================================================

# A pixel is ink when its L is below the Otsu threshold of L, *or* its
# chroma is above this -- the second test keeps pale coloured lines (yellow,
# light cyan) whose lightness alone reads as paper.
INK_CHROMA_MIN = 20.0
# Ink at or below this chroma is achromatic (black / gray) -> one layer.
ACHROMATIC_CHROMA_MAX = 20.0
# Hue histogram over the chromatic ink: one bin per degree, weighted by
# chroma, circularly Gaussian-smoothed before peak picking.
HUE_HIST_BINS = 360
HUE_SMOOTH_SIGMA_DEG = 6.0
# A smoothed-histogram local maximum is a layer peak when it holds at least
# this fraction of the histogram's peak height.
HUE_PEAK_MIN_FRAC = 0.02
# A layer with fewer pixels is folded into its nearest layer (by hue, or
# into the achromatic layer when there is no other chromatic one).
LAYER_MIN_PIXELS = 200

# ======================================================================
# Text removal (text_removal.py)
# ======================================================================

# Dilation (px) of a recognized box's ink-label mask before erasing, to
# catch antialiased fringe pixels.
ERASE_DILATE_PX = 1
