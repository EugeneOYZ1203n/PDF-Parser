"""Tunables for the DeepVectoriser P2 backend (Liu et al., AAAI-22,
"End-to-End Line Drawing Vectorization" -- references/00052-LiuH.pdf).

Flow: color separation -> tiled OCR (960 px tiles) -> text removal ->
per ink layer: binary layer mask -> 128 px vectorizer tiles -> learned
stroke vectorizer -> tile merge.

The pre-vectorizing constants (color separation / OCR / text removal /
diff) are *copied* from `P2_Raster_To_Vec/Junction/config.py`
(not imported), per CLAUDE.md's "sibling backends share zero code" rule.
"""
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
# Vector/ink diff debug (diff.py)
# ======================================================================

# A vector pixel within this many px of ink (and vice versa) counts as
# matched rather than spurious/missed.
DIFF_TOLERANCE_PX = 2

# ======================================================================
# Vectorizer tiling (adapter.py / inference.py / geometry.py)
# ======================================================================

# Canonical scale the model is trained and run at: 300 dpi. Every image is
# resampled to this many px per page point before vectorizer tiling, so a
# 1 pt line is ~4 px wide no matter the source image's own resolution.
TARGET_PX_PER_PT = 300.0 / 72.0
# Vectorizer tile side (px, at TARGET_PX_PER_PT) and overlap between
# neighbouring tiles. The paper's model handles 64-256 px inputs.
TILE_PX = 128
TILE_OVERLAP_PX = 16
# A tile is "saturated" when at least this fraction of the encoder's
# n_stroke queries are confident -- it is then split into 4 sub-tiles of
# TILE_PX // 2 and re-run (once; sub-tiles are never split again).
SATURATION_FRAC = 0.9
# A tile with fewer ink pixels than this is skipped (never run).
MIN_TILE_INK_PX = 8
# A layer-mask pixel is ink when its gray value is below this (the mask is
# 0/255, area-resampled to the canonical scale).
INK_GRAY_THRESHOLD = 200

# Stroke confidence below this is dropped (paper: p < 0.5).
CONFIDENCE_THRESHOLD = 0.5
# A Bezier piece whose control points are within this many px of its chord
# is emitted as an "l" item; otherwise as a "c" item.
FLAT_TOL_PX = 0.75
# Stroke endpoints from different tiles closer than this (px) are snapped
# together at their mean, joining the strokes across the tile seam.
MERGE_SNAP_PX = 3.0
# Vectorizer tiles are run through the model in batches of this many.
INFER_BATCH_TILES = 64
# torch CPU threads for inference, applied once per process when the model is
# first loaded; None leaves torch's own choice (Pool-1/2 workers already pin
# OMP/MKL to one thread each -- `core/parallel/pool.py::worker_init`).
INFER_THREADS: "int | None" = None

# ======================================================================
# Model (model/) -- the paper's own values where it gives them; the rest
# (marked "ours") are our choices. A checkpoint stores its own copy of
# these, so a trained .pth always rebuilds the architecture it was
# trained with regardless of later edits here.
# ======================================================================

MODEL_DEFAULTS = {
    "d_model": 256,          # ours
    "d_emb": 64,             # ours -- emb part of F_i (d_F = 5 + d_emb)
    "n_stroke": 64,          # ours -- encoder queries (max strokes per tile)
    "enc_layers": 8,         # paper: 8 decoder layers in the Stroke Encoder
    "vec_enc_layers": 6,     # paper: 6 encoder layers in the Stroke Vectorizer
    "vec_dec_layers": 6,     # paper: 6 decoder layers in the Stroke Vectorizer
    "n_heads": 8,            # paper
    "ff_dim": 1024,          # ours
    "unet_base": 32,         # ours -- UNet channels at full resolution
    "unet_depth": 4,         # ours -- number of 2x downsamplings
    "max_prims": 16,         # ours -- MAX_PRIMS_PER_STROKE
    "dropout": 0.1,          # ours
}

# Loss weights (paper, Eq. 3-10).
LAMBDA_E = 0.5
LAMBDA_R = 0.3
LAMBDA_C = 0.75
LAMBDA_RECON = 0.1   # lambda_R in Eq. 10
LAMBDA_PRIM = 6.0    # lambda_P in Eq. 10

# ======================================================================
# Weights
# ======================================================================

# Default trained-weights file (rastervec/weights/ is gitignored); the
# DEEPVEC_WEIGHTS_PATH environment variable overrides it.
WEIGHTS_FILENAME = "deep_vectoriser.pth"
WEIGHTS_ENV_VAR = "DEEPVEC_WEIGHTS_PATH"
