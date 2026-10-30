"""Tunables for the ImplicitSketchVec raster_vectorize backend (Yan, Li, Aneja, Fisher,
Simo-Serra, Gingold, SIGGRAPH 2024, "Deep Sketch Vectorization via Implicit
Surface Extraction" -- references/Deep Sketch Vectorization 3658197.pdf).

Flow: the shared raster prelude (`commons/raster_text/` -- OCR, colour CLAHE,
colour layers, text removal; its tunables are in `commons/raster_text/config.py`) ->
per ink layer: binary layer mask -> Distance Field Prediction network
(six UDFs at 2x super-resolution) -> Line Reconstruction network (2D Neural
Dual Contouring: edge flags + vertex map + skeleton) -> post-processing
(edge-flag refinement, dual contouring, topology refinement, DC
downsampling, line grouping, curve fitting).

The diff constants have the same values as `junction/config.py`'s (each backend keeps its own
knobs).
"paper" marks values the paper gives; "ours" marks our choices where it
leaves them open.
"""

from __future__ import annotations

from raster_parser.commons.raster_text.strokes import FLAT_TOL_PX, INFER_THREADS  # noqa: F401 -- shared

# ======================================================================
# Vector/ink diff debug (diff.py)
# ======================================================================

# A vector pixel within this many px of ink (and vice versa) counts as
# matched rather than spurious/missed.
DIFF_TOLERANCE_PX = 2

# ======================================================================
# Grids (train_data.py / inference.py)
#
# A W x H raster (pixel (i, j) covers [i, i+1] x [j, j+1]) has a dual-
# contouring grid of 0.5 px cells (paper: "sampling rate 0.5-pixel"), i.e.
# 2W x 2H cells, whose corners -- the UDF lattice -- are the
# (2W+1) x (2H+1) points (k/2, l/2).
# ======================================================================

# Canonical scale the model is trained and run at: 300 dpi -- the scale of
# the shared DeepVectoriser `prep_dataset.py` data.
TARGET_PX_PER_PT = 300.0 / 72.0
# Super-resolution factor of the UDFs / DC grid (paper: 2x).
SR = 2
# UDFs are clamped at this distance (px, ours) and predicted as udf / UDF_TRUNC_PX.
UDF_TRUNC_PX = 8.0
# Loss mask: lattice points whose GT centerline UDF is below this (paper: 4.5 px).
LOSS_MASK_PX = 4.5
# Dense sampling step along GT strokes for UDFs / DC ground truth (px, ours).
GT_SAMPLE_STEP_PX = 0.1
# Margin around a training crop whose strokes still shape its UDFs (px, ours).
GT_MARGIN_PX = UDF_TRUNC_PX

# A layer-mask pixel is ink when its gray value is below this.
INK_GRAY_THRESHOLD = 200

# ======================================================================
# Keypoint ground truth (train_data.py) -- paper Sec. 5 (after Puhachov
# et al. 2021): endpoints, sharp turns, junctions
# ======================================================================

# Stroke ends closer than this are the same point (px, ours).
KEYPOINT_MERGE_PX = 1.5
# A tangent turn sharper than this (deg) is a "sharp" keypoint (ours).
SHARP_DEG = 30.0

# ======================================================================
# Networks (model/) -- a checkpoint stores its own copy
# ======================================================================

MODEL_DEFAULTS = {
    # Distance Field Prediction network (Sec. 5): a ResNeXt FCN modelled on
    # Puhachov et al. 2021, the paper's base -- "inner channel 128 -> 256,
    # cardinality 3 -> 8 with gradually increasing dilation". The exact
    # layer layout is ours (Puhachov's isn't reproduced in the paper).
    "dfp_ch": 256,  # paper (full); --basic: 128
    "dfp_cardinality": 8,  # paper
    "dfp_dilations": [1, 1, 2, 2, 4, 4, 8, 8],  # ours ("gradually increasing")
    "dfp_hr_ch": 32,  # ours: channels at the 2x super-resolution head
    "n_udf": 6,  # paper: centerline, USM, end, sharp, junc, all
    # Line Reconstruction network (Sec. 6, Appendix A, Fig. 22)
    "ndc_ch": 64,  # ours
    "ndc_branch_layers": 3,  # paper: three consecutive residual convolutions per branch
    "ndc_dilations": [1, 2, 3],  # paper: 3x3 / 5x5 / 7x7 receptive fields
    "ndc_trunk_layers": 6,  # ours (Fig. 22 draws six)
}
BASIC_OVERRIDES = {"dfp_ch": 128, "ndc_ch": 48}  # paper's reduced "basic" variant (DFP 256 -> 128)

# UDF channel order
UDF_CHANNELS = ("centerline", "usm", "end", "sharp", "junc", "all")

# Loss weights (paper Eq. 4): L_rec = L_edge + 0.5 L_vertex + 0.01 L_skeleton
LAMBDA_VERTEX = 0.5
LAMBDA_SKELETON = 0.01
# NDC-stage input noise: Gaussian, std = this fraction of the GT UDF's std
# (Appendix C's robustness test level; training on it is ours).
NDC_INPUT_NOISE_FRAC = 0.01

# ======================================================================
# Inference (inference.py / postprocess.py) -- paper Sec. 7
# ======================================================================

# Raster tile side and overlap (px, ours). Fields are stitched from each
# tile's core, so the overlap only has to exceed the receptive field's reach.
TILE_PX = 256
TILE_OVERLAP_PX = 32
# A tile with fewer ink pixels than this is skipped.
MIN_TILE_INK_PX = 8
# Tiles per network batch.
INFER_BATCH_TILES = 4
# torch CPU threads for inference, applied once per process when the model is
# first loaded; None leaves torch's own choice (Pool-1/2 workers already pin
# OMP/MKL to one thread each -- `core/parallel/pool.py::worker_init`).
# A cell's vertex is stored only where the predicted centerline UDF at the
# cell centre is below this (px, ours) -- vertices exist only next to flagged
# edges, and flags only near strokes.
VERTEX_UDF_PX = 2.0
# Keypoints: lattice points whose predicted keypoint UDF is below this (px)
# and minimal within KEYPOINT_NMS_PX (ours: "threshold" + "local maximum").
KEYPOINT_UDF_PX = 1.0
KEYPOINT_NMS_PX = 2.0
# Under-sampled cells: predicted U_USM at the cell centre below this (px, ours).
USM_UDF_PX = 0.5
# DC downsampling factor (paper Sec. 7 / Fig. 10, e.g. 3x / 8x; ours: 2 = back
# to the native pixel grid). 1 disables.
DC_DOWNSAMPLE = 2
# Line grouping runs the paper's repeated longest-shortest-path on graph
# components up to this many vertices; larger ones (quadratic there) are
# split into maximal chains instead (ours).
GROUP_EXACT_MAX_NODES = 4000
# Grouped strokes shorter than this are dropped as noise (px, ours).
MIN_STROKE_PX = 2.0
# Smoothing (paper Sec. 7: RDP or Schneider's Bezier fitting).
FIT_MODE = "bezier"  # "bezier" | "rdp"
FIT_TOL_PX = 0.75  # Schneider max error / RDP epsilon (ours)
# A fitted cubic within this of its chord is emitted as an "l" item (px).

# ======================================================================
# Weights
# ======================================================================

# The final (joint-stage) weights the adapter loads, plus the two stage
# files the separate DFP / NDC trainings write by default (raster_parser/weights/
# is gitignored); IMPLICITSKETCHVEC_WEIGHTS_PATH overrides the final file.
WEIGHTS_FILENAME = "implicit_sketch_vec.pth"
STAGE_WEIGHTS_FILENAMES = {
    "dfp": "implicit_sketch_vec_dfp.pth",
    "ndc": "implicit_sketch_vec_ndc.pth",
    "joint": WEIGHTS_FILENAME,
}
WEIGHTS_ENV_VAR = "IMPLICITSKETCHVEC_WEIGHTS_PATH"
