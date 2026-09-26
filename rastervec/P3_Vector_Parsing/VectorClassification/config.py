"""Tunable thresholds for the VectorClassification P3 backend (a reduced
2-step Vector_Classification chain + Radon word-segmentation + PaddleOCR
recognition-only OCR). Self-contained -- not shared with FastIntoPaddle's
own config.py, per the phase-isolation rule (see the repo's plan doc)."""
from __future__ import annotations

# ======================================================================
# Vector Classification -- the reduced 2-step chain in classify_vectors.py.
# ======================================================================

SEQ_OVERLAP_TOLERANCE_PX = 1.0
SPATIAL_CLUSTER_THRESHOLD = 10.0
SPATIAL_SIZE_TOLERANCE = 0.30

# ======================================================================
# fast_text_detect stage (fast_filter.py / fast_detect.py)
# ======================================================================

# The page is rendered once at FAST_PAGE_RENDER_DPI * FAST_TILE_SCALE_FACTOR
# (= 300 dpi) and cut into FAST_TILE_BLOCK_SIZE-square tiles. The block size
# must equal fast_detect._SHORT_SIDE (FAST's own 640 px test-time short
# side): each tile is then fed to the model 1:1, never resampled, so FAST
# really sees the page at 300 dpi. (The old 2048 px tiles were shrunk 3.2x
# to 640 px -- an effective ~94 dpi.)
FAST_PAGE_RENDER_DPI = 150
FAST_TILE_BLOCK_SIZE = 640
FAST_TILE_SCALE_FACTOR = 2
# Tile-selection margin around each candidate bbox, as a fraction of the
# block size (0.05 * 640 = 32 px, ~7.7 pt at 300 dpi). Only decides which
# tiles run -- a candidate's own bbox always selects its tiles, and the
# overlap (0.15 * 640 = 96 px, ~23 pt) covers text straddling a tile edge.
FAST_TILE_CANDIDATE_MARGIN_FRAC = 0.05
FAST_TILE_OVERLAP_FRAC = 0.15
# A cluster is kept as text if ANY single member vector's own FAST mask
# coverage score exceeds this (not a whole-cluster average -- one strong
# ink-looking vector is enough to save the whole cluster from being dropped
# to drawing output).
FAST_VECTOR_ANY_THRESHOLD = 0.05
# Resolution of the `fast / heatmap` debug layer's embedded PNG. Debug-only:
# the mask FAST scores clusters with stays at the 300 dpi render.
FAST_HEATMAP_DPI = 100

# ======================================================================
# Raster-refined rotation (paddle_engine.py::hough_deskew)
# ======================================================================

# Grayscale value (0-255) below which a pixel counts as "ink" when building
# the binary mask Hough line detection runs on.
HOUGH_INK_THRESHOLD = 200
# Binary-dilation disk radius (px) applied to the ink mask before Hough --
# thickens thin/broken strokes so Hough has more to find ("increase ink").
HOUGH_DILATE_RADIUS_PX = 2
# Grayscale value (0-255) below which a pixel counts as "ink" when building
# the (non-dilated) mask `cv2.minAreaRect` runs on -- independently tunable
# from HOUGH_INK_THRESHOLD since the two estimators may want different
# sensitivity in practice.
MINAREA_INK_THRESHOLD = 200
# The combined (Hough + minAreaRect) rotation correction is snapped to the
# nearest multiple of this many degrees before being applied.
HOUGH_ANGLE_SNAP_DEG = 10.0

# ======================================================================
# OCR (paddle_engine.py, wordgrouping.py, parse.py)
# ======================================================================

OCR_VERSION = "PP-OCRv4"
OCR_LANG = "en"
OCR_BATCH_SIZE = 128
MIN_RENDER_SIDE_PX = 100
MAX_RENDER_DPI = 4800

# render_cluster_with_dynamic_dpi's base dpi for a seqno-clustered word
# group's own render -- matches the value `radon.py::segment_clusters` used
# to be called with by default, before Radon deskewing was removed.
OCR_DPI = 300

# _cluster_render_padding (parse.py): page-space PDF-point margin added to a
# cluster's own render frame, on top of half its own max stroke width. This
# is now the *only* padding in the OCR render path (the old pixel-space
# post-render white border was removed), so it's intentionally larger than
# LegacyRecreation/config.py's own RENDER_PADDING_EXTRA_PT=5.0pt (which still
# matches archive's original PngRenderer.render_word_group flat padding).
RENDER_PADDING_EXTRA_PT = 15.0
