"""Tunables for the InvolutionStroke P2 backend (Gerstner, Magne &
Sorkine-Hornung, VMV 2026, "Stroke Vectorization via Involution Prediction",
https://github.com/verven/InvolutionStrokeVectorization), run with the
authors' pretrained weights.

Flow: the shared raster prelude (`RasterText/` -- OCR, colour CLAHE,
colour layers, text removal; its tunables are in `RasterText/config.py`) ->
per ink layer: binary layer mask -> per 1024 px tile the full upstream chain
(centerline network -> planar line graph -> half-edge pairing network +
involutive matching -> strokes) -> clip to the tile's core -> join strokes
across tile seams -> corner split + curve fitting.

The diff constants are *copied* from `P2_Raster_To_Vec/Junction/config.py`
(not imported), per CLAUDE.md's "sibling backends share zero code" rule.
"upstream" marks values taken from the reference implementation; "ours"
marks our choices.
"""
from __future__ import annotations

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
