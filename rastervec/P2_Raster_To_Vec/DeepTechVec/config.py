"""Tunables for the DeepTechVec P2 backend (Egiazarian et al., ECCV 2020,
"Deep Vectorization of Technical Drawings" --
references/Deep Vectorization of Technical Drawings 2003.05471v3.pdf).

Flow: the shared raster prelude (`RasterText/` -- OCR, colour CLAHE,
colour layers, text removal; its tunables are in `RasterText/config.py`) ->
per ink layer: binary layer mask -> 64 px patches -> the authors' pretrained
primitive network (lines or quadratic Beziers + width + confidence;
github.com/Vahe1994/Deep-Vectorization-of-Technical-Drawings, MPL-2.0) ->
energy refinement -> merging (both ours, after the paper).

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

# Scale the model is run at (ours: 300 dpi -- the upstream models were
# trained on PFP floor-plan / ABC CAD renders, whose dpi isn't given; tune
# with `predict.py --dpi`). Every image is resampled to this many px per page
# point before patching.
TARGET_PX_PER_PT = 300.0 / 72.0
# Patch side (paper + pretrained weights: 64 x 64 -- fixed, the network's
# coordinate channels and x64 output scaling assume it) and overlap between
# neighbouring patches (ours; upstream runs without overlap).
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
# Primitive network (model/) -- the authors' released architecture, copied
# from upstream's vectorization/models/specs/*.json (lines:
# resnet18_blocks3_bn_256__c2h__trans_heads4_feat256_blocks4_ffmaps512__h2o__out512,
# curves: resnet18_blocks1_bn_64__c2h__trans_heads4_feat256_blocks8_ffmaps512__h2o__out512).
# ======================================================================

# The kind of primitive the adapter loads weights for when nothing else says
# (paper: lines for floor plans -- PFP -- quadratic curves for ABC).
DEFAULT_PRIM_KIND = "line"
PRIM_KINDS = ("line", "curve")

UPSTREAM_SPECS = {
    "line": {"in_channels": 3, "convmap_channels": 256, "blocks": 3, "feature_dim": 256, "ffn_dim": 512,
             "n_head": 4, "num_layers": 4, "in_features": 512, "out_features": 6},
    "curve": {"in_channels": 3, "convmap_channels": 64, "blocks": 1, "feature_dim": 256, "ffn_dim": 512,
              "n_head": 4, "num_layers": 8, "in_features": 512, "out_features": 8},
}
# Primitive queries per patch (upstream `--model_output_count` / `-c`, default 10).
N_PRIM_QUERIES = 10

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
# Weights -- `download_weights.py`
# ======================================================================

# Default weights file per primitive kind (rastervec/weights/ is gitignored);
# the DEEPTECHVEC_WEIGHTS_PATH environment variable overrides.
WEIGHTS_FILENAMES = {"line": "deep_tech_vec_lines.weights", "curve": "deep_tech_vec_curves.weights"}
WEIGHTS_ENV_VAR = "DEEPTECHVEC_WEIGHTS_PATH"
# The authors' public Yandex Disk links (upstream README) and the released
# files' sha256 (checked after download, and before a full unpickle).
WEIGHTS_URLS = {
    "line": "https://disk.yandex.ru/d/FKJuMvNJuy-K9g",    # model_lines.weights, 149 MB
    "curve": "https://disk.yandex.ru/d/yOZzCSrd-QSACA",   # model_curves.weights, 231 MB
}
WEIGHTS_SHA256 = {
    "line": "c92f62f1d6582efe9f51df8d991d6a561a424a46051be2f1b54d9fd326b9363e",
    "curve": "bac97280f52fa049e9f7f6a03dbc6921f2e67709b505a7139a7768f88038dc26",
}
# Yandex Disk public API: public link -> one-off download href.
YANDEX_DOWNLOAD_API = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
