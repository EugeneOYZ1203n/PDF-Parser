"""Tunables for the DeepTechVec P2 backend (Egiazarian et al., ECCV 2020,
"Deep Vectorization of Technical Drawings" --
references/Deep Vectorization of Technical Drawings 2003.05471v3.pdf).

Flow: the shared raster prelude (`RasterText/` -- OCR, colour CLAHE,
colour layers, text removal; its tunables are in `RasterText/config.py`) ->
per ink layer: binary layer mask -> 64 px patches -> primitive network
(lines or quadratic Beziers + width + confidence) -> energy refinement ->
merging.

The diff constants are *copied* from `P2_Raster_To_Vec/Junction/config.py`
(not imported), per CLAUDE.md's "sibling backends share zero code" rule.
"paper" marks values the paper gives; "ours" marks our choices where it
leaves them open.
"""
from __future__ import annotations

# ======================================================================
# Vector/ink diff debug (diff.py)
# ======================================================================

# A vector pixel within this many px of ink (and vice versa) counts as
# matched rather than spurious/missed.
DIFF_TOLERANCE_PX = 2

# ======================================================================
# Patching (inference.py / geometry.py)
# ======================================================================

# Canonical scale the model is trained and run at: 300 dpi -- the scale of
# the shared DeepVectoriser `prep_dataset.py` data. Every image is resampled
# to this many px per page point before patching.
TARGET_PX_PER_PT = 300.0 / 72.0
# Patch side (paper: 64 x 64) and overlap between neighbouring patches (ours).
PATCH_PX = 64
PATCH_OVERLAP_PX = 16
# A patch with fewer ink pixels than this is skipped (never run).
MIN_PATCH_INK_PX = 8
# A layer-mask pixel is ink when its gray value is below this.
INK_GRAY_THRESHOLD = 200
# Patches are run through the network in batches of this many.
INFER_BATCH_PATCHES = 256
# torch CPU threads for inference, applied once per process when the model is
# first loaded; None leaves torch's own choice (Pool-1/2 workers already pin
# OMP/MKL to one thread each -- `core/parallel/pool.py::worker_init`).
INFER_THREADS: "int | None" = None

# Primitive confidence below this is discarded (paper: 0.5).
CONFIDENCE_THRESHOLD = 0.5
# Primitives shorter than this after clipping to a patch's core are dropped (px, ours).
MIN_PRIM_LEN_PX = 1.0

# ======================================================================
# Primitive network (model/) -- paper Appendix A. A checkpoint stores its
# own copy, so a trained .pth always rebuilds the architecture it was
# trained with.
# ======================================================================

# The kind of primitive the adapter loads weights for when nothing else says
# (paper: lines for floor plans -- PFP -- quadratic curves for ABC).
DEFAULT_PRIM_KIND = "line"
PRIM_KINDS = ("line", "curve")
# Geometric parameters per primitive (excluding confidence): line
# (x1,y1,x2,y2,w), quadratic Bezier (x1,y1,x2,y2,x3,y3,w).
N_PARAMS = {"line": 5, "curve": 7}

MODEL_DEFAULTS = {
    "prim_kind": DEFAULT_PRIM_KIND,
    "n_prim": 10,        # paper: "more than 97% of patches ... no more than 10 primitives"
    "res_ch": 64,        # paper: c = 64 channels
    "n_res": 1,          # paper: n_res = 1 ResNet18 block
    "n_dec": 8,          # paper: n_dec = 8 Transformer blocks
    "n_heads": 4,        # paper: 4 heads
    "ff_dim": 512,       # paper: 512 neurons in the last fully-connected layer
    "dropout": 0.1,      # ours (Transformer default)
    # d_emb = parameters incl. confidence (paper: 6 lines / 8 curves) -- derived
    # from prim_kind, not stored separately.
}

# Loss (paper Eq. 2-4): BCE + (1 - LAMBDA_LOC) * L1 + LAMBDA_LOC * L2^2.
LAMBDA_LOC = 0.5     # ours (paper: "weighted sum", weight not given)

# ======================================================================
# Training data (train_data.py)
# ======================================================================

# Random rotation (any angle) and scale of each training crop (paper:
# "random rotation and scaling"; the scale range is ours).
AUG_SCALE_RANGE = (0.8, 1.25)
# A cubic is split until its controls are within this of the chord before
# becoming line primitives (px, ours).
LINE_FLAT_TOL_PX = 0.5
# Consecutive chords are merged into one line while within this angle (deg, ours).
LINE_MERGE_ANGLE_DEG = 3.0
# A cubic is split at t = 1/2 until its least-squares quadratic is within this (px, ours).
QUAD_FIT_TOL_PX = 0.5
# Crops with more than n_prim primitives are re-drawn this many times before
# keeping the n_prim longest (ours).
OVERFLOW_RETRIES = 5

# ======================================================================
# Refinement (refine.py) -- paper Sec. 3.3 + Appendix I
# ======================================================================

# Adam iterations per patch batch; 0 disables refinement (ours -- the paper
# trades iterations for accuracy, Table 4).
REFINE_ITERS = 50
# Adam step sizes per parameter kind (ours): positions / lengths / widths in
# px, angles in radians.
REFINE_LR_PX = 0.1
REFINE_LR_ANGLE = 0.01
# Patches refined together (memory bound, ours).
REFINE_BATCH = 64
# The connected-area mask c_k is recomputed every this many iterations (ours).
REFINE_MASK_EVERY = 5
# Potential phi(r) = exp(-r^2/R_c^2) + lambda_f exp(-r^2/R_f^2) (paper Eq. 47).
REFINE_R_CLOSE = 1.0
REFINE_R_FAR = 32.0
REFINE_LAMBDA_FAR = 0.02
# Radius of the convolved phi kernel (px, ours: the R_f tail is truncated so
# the convolution stays cheap on 64 px patches).
REFINE_KERNEL_RADIUS = 16
# phi truncation radius r* for the collinearity term (paper: "close phi",
# lambda_f = 0; value ours).
REFINE_RDN_RADIUS = 3
# Positional amplification inside the connected-area mask (paper Eq. 46,
# "chosen empirically"; value ours).
REFINE_LAMBDA_POS = 2.0
# Collinearity threshold angle alpha_col (paper: beta = (cos 15 deg - 1)^-2).
REFINE_COLLINEAR_DEG = 15.0
# Every this many iterations: join lined-up primitives and move collapsed
# ones onto uncovered ink (paper Sec. 3.3: "every few iterations"; ours).
REFINE_JOIN_EVERY = 20
# A primitive shorter than this (px) counts as collapsed (ours).
REFINE_COLLAPSED_PX = 1.0
# Width limits during refinement (px, ours).
REFINE_MIN_WIDTH_PX = 0.5
REFINE_MAX_WIDTH_PX = 16.0
# Curves are flattened into this many segments for distances (ours).
CURVE_FLATTEN_SEGMENTS = 8

# ======================================================================
# Merging (merge.py) -- paper Sec. 3.4 + Appendix C
# ======================================================================

MERGE = True
# Lines are linked when both endpoints of the shorter lie within this
# distance of the longer's infinite line (px) -- parallel-offset lines fail
# this test ("collinear but not almost parallel") ...
MERGE_LINE_DIST_PX = 1.5
# ... their directions differ by at most this (deg) ...
MERGE_LINE_ANGLE_DEG = 5.0
# ... and they overlap or leave a gap of at most this along the line (px).
MERGE_LINE_GAP_PX = 2.0
# Widths must agree within this ratio to merge (ours).
MERGE_WIDTH_RATIO = 2.0
# Endpoint snapping: a dangling end past an intersection shorter than this
# fraction of the primitive's length is cut (paper: "a few percent").
DANGLE_FRAC = 0.05
# Curves: the second curve's midpoint and endpoints within this distance of
# the first, and the joint fit's max point error within MERGE_CURVE_FIT_PX (ours).
MERGE_CURVE_DIST_PX = 2.0
MERGE_CURVE_FIT_PX = 1.0
# Brute-force samples for u_q1 (paper Eq. 15).
MERGE_CURVE_U_STEPS = 32
# A quadratic whose control point is within this of its chord is emitted as
# an "l" item (px).
FLAT_TOL_PX = 0.75

# ======================================================================
# Weights
# ======================================================================

# Default trained-weights file per primitive kind (rastervec/weights/ is
# gitignored); the DEEPTECHVEC_WEIGHTS_PATH environment variable overrides.
WEIGHTS_FILENAMES = {"line": "deep_tech_vec_line.pth", "curve": "deep_tech_vec_curve.pth"}
WEIGHTS_ENV_VAR = "DEEPTECHVEC_WEIGHTS_PATH"
