"""Tunables for the InvolutionStroke P2 backend (Gerstner, Magne &
Sorkine-Hornung, VMV 2026, "Stroke Vectorization via Involution Prediction",
https://github.com/verven/InvolutionStrokeVectorization), run with the
authors' pretrained weights.

Flow: color separation -> tiled OCR (960 px tiles) -> text removal ->
per ink layer: binary layer mask -> per 1024 px tile the full upstream chain
(centerline network -> planar line graph -> half-edge pairing network +
involutive matching -> strokes) -> clip to the tile's core -> join strokes
across tile seams -> corner split + curve fitting.

The pre-vectorizing constants (color separation / OCR / text removal /
diff) are *copied* from `P2_Raster_To_Vec/ImplicitSketchVec/config.py` (not
imported), per CLAUDE.md's "sibling backends share zero code" rule.
"upstream" marks values taken from the reference implementation; "ours"
marks our choices.
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
# The padded box is upscaled until its short side is at least this many px,
# but never by more than MAX_UPSCALE.
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
# Layer mask -> tiles (inference.py)
# ======================================================================

# Canonical scale each layer mask is resampled to before the networks (ours:
# 300 dpi, the learned siblings' scale). The networks were trained on
# sketches rasterized at 512-1024 px with strokes a few px wide; 300 dpi
# drafting lines are ~1-3 px. Lower it (e.g. 200/72) to trade detail for
# speed -- the centerline UNet costs ~35 s per 1024 px tile on CPU.
TARGET_PX_PER_PT = 300.0 / 72.0
# A layer-mask pixel is ink when its gray value is below this.
INK_GRAY_THRESHOLD = 200
# Tile side / overlap (px). 1024 = the largest size upstream benchmarks at,
# so graph-extraction and pairing see the scale they were tuned for; every
# stage runs per tile and strokes are clipped to the tile's core afterwards,
# so the overlap only has to cover the context a junction needs (ours).
TILE_PX = 1024
TILE_OVERLAP_PX = 96
# A tile with fewer ink pixels than this is skipped.
MIN_TILE_INK_PX = 8
# Upstream's `imgpreprocess.preprocess` normalization (white top-hat +
# background subtraction); off = upstream's `--skip-preprocess`, right for
# our already-clean binary masks.
PREPROCESS = False
# Centerline tiles per network batch.
INFER_BATCH_TILES = 1
# torch CPU threads, applied once per process when the models load; None
# leaves torch's own choice.
INFER_THREADS: "int | None" = None

# ======================================================================
# Graph extraction (upstream graph_extraction/build_linegraph.process)
# ======================================================================

# upstream CLI defaults: --sharp off, --deg3 off, --deg4 on (--no-deg4 off)
GRAPH_SHARP = False
GRAPH_DEG3 = False
GRAPH_DEG4 = True
# upstream --smooth: shapeutils.smooth_polyline_constrained(pl, d=0.5)
SMOOTH_LINES = False
SMOOTH_D = 0.5

# ======================================================================
# Seams + output (inference.py / adapter.py) -- ours
# ======================================================================

# Stroke ends from different tiles closer than this (px) on a core border are
# joined into one stroke.
MERGE_SNAP_PX = 2.0
# Strokes shorter than this after clipping (px) are dropped.
MIN_STROKE_PX = 2.0
# Smoothing of each final stroke: split where the stroke, RDP-simplified at
# CORNER_EPS_PX, turns by more than CORNER_DEG; then each run is one line if
# RDP at FIT_TOL_PX leaves a single segment, else "bezier" (Gaussian
# pre-smoothing of FIT_PRESMOOTH_PX along the stroke -- upstream polylines
# are 0.5 px staircases -- then Schneider fit) or "rdp" (lines).
FIT_MODE = "bezier"              # "bezier" | "rdp"
FIT_TOL_PX = 0.75
CORNER_DEG = 35.0
CORNER_EPS_PX = 2.0
FIT_PRESMOOTH_PX = 1.0
# A fitted cubic within this of its chord is emitted as an "l" item (px).
FLAT_TOL_PX = 0.75

# ======================================================================
# Weights (download_weights.py) -- the authors' released checkpoints
# ======================================================================

CENTERLINE_WEIGHTS_FILENAME = "involution_centerline.pt"
INTERSECTION_WEIGHTS_FILENAME = "involution_intersection.pt"
CENTERLINE_WEIGHTS_URL = "https://igl.ethz.ch/projects/involution-stroke-vectorization/centerline_checkpoint.pt"
INTERSECTION_WEIGHTS_URL = "https://igl.ethz.ch/projects/involution-stroke-vectorization/intersection_checkpoint.pt"
CENTERLINE_WEIGHTS_ENV_VAR = "INVOLUTIONSTROKE_CENTERLINE_WEIGHTS"
INTERSECTION_WEIGHTS_ENV_VAR = "INVOLUTIONSTROKE_INTERSECTION_WEIGHTS"
