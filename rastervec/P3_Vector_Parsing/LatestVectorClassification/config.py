"""Tunable thresholds for the LatestVectorClassification P3 backend --
layer/color/width buckets -> collinear drawing removal -> seqno-overlap merge
-> constrained spatial clustering -> parallel-group length outliers ->
crossed-grid removal -> PaddleOCR detect, quad-angle upright crop, 0/180
classifier + recognize, low-score retry -> quad ink-ownership text/drawing
split. Self-contained -- not shared with any sibling backend's config.py."""
from __future__ import annotations

# ======================================================================
# Classification chain (classify_vectors.py) -- seqno merge + spatial clustering.
# ======================================================================

SEQ_OVERLAP_TOLERANCE_PX = 1.0
SPATIAL_CLUSTER_THRESHOLD = 10.0
SPATIAL_SIZE_TOLERANCE = 0.30

# Page-area caps, as fractions of the page's (unrotated CropBox) area. The
# "Oversize" step drops any Vector whose bbox area is at least
# MAX_VECTOR_PAGE_AREA_FRAC of the page to drawing (a border, title-block
# frame or background fill is never text). During clustering, neither the
# seqno merge nor the spatial merge ever grows a group/cluster to a bbox of
# MAX_CLUSTER_PAGE_AREA_FRAC of the page or more -- this bounds each
# cluster's OCR render (on A0 at 300 dpi, 30 % is ~42 MP / ~125 MB as BGR;
# a whole-page cluster was ~416 MB).
MAX_VECTOR_PAGE_AREA_FRAC = 0.30
MAX_CLUSTER_PAGE_AREA_FRAC = 0.30

# ======================================================================
# Line geometry (line_geometry.py) -- straight Vectors, collinear/parallel
# grouping, proper crossings.
# ======================================================================

# Max point distance (pt) from the fitted line for a Vector to count as
# straight; only straight Vectors take part in collinear/parallel grouping.
STRAIGHT_TOL_PT = 0.25
# Single-linkage angle tolerance (deg, folded to [0, 180)) for both
# collinear and parallel grouping.
ANGLE_TOL_DEG = 2.0
# Anchored perpendicular-offset tolerance (pt) for collinear grouping.
COLLINEAR_OFFSET_TOL_PT = 1.0
# A collinear group becomes drawing when its member-length std (pt) is below
# this AND it has more than COLLINEAR_DRAWING_MIN_COUNT members -- a long
# dashed/repeated line, never text.
COLLINEAR_DRAWING_MAX_STD_PT = 5.0
COLLINEAR_DRAWING_MIN_COUNT = 50
# Minimum members for a set of same-angle straight Vectors to count as a
# "parallel group" (length-outlier pooling, `geometry` debug layers).
# Singletons don't.
MIN_PARALLEL_GROUP_SIZE = 2
# Page-wide "global potential angles" (classify_vectors.py): the mean angle
# of every collinear group with at least this many members, across every
# bucket (drawing groups included), deduped within ANGLE_TOL_DEG. A detect
# quad's long-edge angle snaps to the nearest one within
# QUAD_ANGLE_SNAP_TOL_DEG (parse.py), else stays as detected.
GLOBAL_ANGLE_MIN_GROUP_SIZE = 2
QUAD_ANGLE_SNAP_TOL_DEG = 5.0
# Pattern-lattice step (pattern_lattice.py), per bucket after collinear
# drawing: Vectors are "similar" when they have the same item kinds in order
# and the same per-item lengths, rounded to PATTERN_SIM_LENGTH_TOL_PT
# (rotation-invariant). Only similarity buckets with more than
# PATTERN_MIN_BUCKET members are searched for lattices; a lattice group
# (flood-filled from a seed along +-v1/+-v2, each step matched within
# PATTERN_LATTICE_TOL_FRAC of its own length) with more than
# PATTERN_MAX_GROUP members is dropped to drawing. Neighbours closer than
# PATTERN_MIN_STEP_PT (coincident duplicates) never define a lattice step;
# PATTERN_KNN neighbours are searched per seed for v1/v2.
PATTERN_SIM_LENGTH_TOL_PT = 0.1
PATTERN_MIN_BUCKET = 20
PATTERN_MAX_GROUP = 10
PATTERN_LATTICE_TOL_FRAC = 0.2
PATTERN_MIN_STEP_PT = 0.05
PATTERN_KNN = 16
# Per cluster, strokes in parallel groups whose length is more than this many
# (pooled) standard deviations from the pooled mean are removed to drawing.
LENGTH_OUTLIER_STD = 2.0

# Crossings step (classify_vectors.py::crossed_grid), per cluster: a Vector
# made only of "l" items is *flagged* when at least MIN_CROSSINGS distinct
# foreign pieces properly cross it -- "l" segments and "re"/"qu" edges count
# once each, every proper crossing of an (unflattened) "c" curve counts. The
# flagged Vectors' own segments are folded mod 90; the grid direction (within
# GRID_ANGLE_TOL_DEG, i.e. parallel or perpendicular) holding more than
# GRID_DOMINANCE of the flagged Vectors' total "l" length is dropped to
# drawing. A flagged Vector whose own segments span several grid directions
# never drops.
MIN_CROSSINGS = 4
GRID_ANGLE_TOL_DEG = 2.0
GRID_DOMINANCE = 0.5
# Proper-crossing epsilon (pt): an endpoint (or curve side) within this of
# the other line is touching, not crossing.
CROSS_EPS_PT = 0.01
# Memory bound for the crossing test (line_geometry.line_crossing_counts):
# at most this many (own segment, foreign segment) pairs are broadcast at
# once, ~30 MB of float64 temporaries.
CROSS_PAIR_CHUNK = 250_000

# ======================================================================
# OCR (paddle_engine.py, parse.py)
# ======================================================================

OCR_VERSION = "PP-OCRv4"
OCR_LANG = "en"
OCR_BATCH_SIZE = 128
MIN_RENDER_SIDE_PX = 100
MAX_RENDER_DPI = 4800

# render_cluster_with_dynamic_dpi's base dpi for a classification cluster's
# own render.
OCR_DPI = 300

# parse.py's render+detect stage processes classification clusters in chunks
# of this size rather than rendering the whole page's clusters before
# detecting any of them -- some clusters (e.g. a large title block/border)
# render to tens of MB as a raw array even at the base OCR_DPI, so holding
# every cluster's render simultaneously on a page with hundreds of clusters
# can exhaust memory (and has been observed to crash PaddleOCR's own
# detector with an opaque allocator error under memory pressure). Recognize
# (OCR_BATCH_SIZE) is unaffected -- crops are far smaller than full cluster
# renders, so that stage still batches across the whole page at once.
DETECT_RENDER_CHUNK_SIZE = 8

# _cluster_render_padding (parse.py): page-space PDF-point margin added to a
# cluster's own render frame, on top of half its own max stroke width.
RENDER_PADDING_EXTRA_PT = 15.0

# Upright crop (paddle_engine.py::upright_crop): the region around each
# detected quad is rotated by the quad's long-edge angle (a pure rotation --
# never a re-boxed bbox or a perspective warp), then cropped to the quad's
# own size expanded by CROP_EXPAND_FRACTION, plus a flat CROP_BORDER_PX
# white border.
CROP_EXPAND_FRACTION = 0.05
CROP_BORDER_PX = 5

# Max long:short side ratio of a detect quad sent to recognition as one
# crop (parse.py). PaddleOCR's recogniser pads a whole batch to its widest
# aspect ratio, so one very long crop inflates every crop in its batch; a
# longer quad is split along its long side into ceil(aspect / this) pieces,
# cut in the gaps between the cluster's vectors where possible (a hard cut
# otherwise). Each piece is recognised as its own Text.
MAX_CROP_ASPECT = 25.0

# Recognition retry (parse.py): pass 1 runs PaddleOCR's 0/180 angle
# classifier then recognises. A crop whose pass-1 score is below
# RETRY_CONFIDENCE_THRESHOLD is also recognised at +90/180/270 (no
# classifier) and the best-scoring attempt wins. Score = confidence, 0 for a
# blank read, and confidence * SINGLE_CHAR_PENALTY for a single-character
# read (a lone glyph is the typical misread of a sideways/upside-down word).
# The threshold only triggers retries -- a non-blank winner is always kept.
RETRY_CONFIDENCE_THRESHOLD = 0.8
SINGLE_CHAR_PENALTY = 0.5

# Retry winner selection only (english_words.py::selection_score): among a
# retried crop's attempts (pass 1 included), each attempt's score is
# multiplied by ENGLISH_WORD_MULTIPLIER ** n, n = the number of whitespace
# tokens (edge punctuation stripped) that are purely alphabetic, at least
# ENGLISH_WORD_MIN_LEN letters long, and in pyenchant's ENGLISH_DICT_LANG
# dictionary. Uncapped. The retry trigger above still uses the raw score.
ENGLISH_WORD_MULTIPLIER = 1.20
ENGLISH_WORD_MIN_LEN = 2
ENGLISH_DICT_LANG = "en_US"

# Text/drawing split (parse.py::_quad_owns): a vector in an OCR'd cluster is
# text when a non-blank detect quad (the rotated quad, not its envelope) from
# its own cluster owns it -- tiered, cheapest first: its bbox fully inside the
# quad -> owned; its bbox area larger than the quad's -> not; fewer than
# TEXT_SEGMENT_OVERLAP_FRAC of its pieces ("l" segments, "re"/"qu" edges,
# "c" chords) touching the quad -> not; otherwise owned when more than
# TEXT_INK_INSIDE_FRAC of its ink (path length) lies inside the quad. Curves
# are sampled at INK_CURVE_SAMPLES points for that length measure only.
TEXT_SEGMENT_OVERLAP_FRAC = 0.5
TEXT_INK_INSIDE_FRAC = 0.5
INK_CURVE_SAMPLES = 32
