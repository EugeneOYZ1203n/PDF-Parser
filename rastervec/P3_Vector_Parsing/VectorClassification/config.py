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
# Combined FAST score threshold above which a cluster is kept as text.
FAST_COMBINED_KEEP_THRESHOLD = 0.3

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
