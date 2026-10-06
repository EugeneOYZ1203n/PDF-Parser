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

# Recognition retry (parse.py): pass 1 runs PaddleOCR's 0/180 angle
# classifier then recognises. A crop whose pass-1 score is below
# RETRY_CONFIDENCE_THRESHOLD is also recognised at +90/180/270 (no
# classifier) and the best-scoring attempt wins. Score = confidence, 0 for a
# blank read, and confidence * SINGLE_CHAR_PENALTY for a single-character
# read (a lone glyph is the typical misread of a sideways/upside-down word).
# The threshold only triggers retries -- a non-blank winner is always kept.
RETRY_CONFIDENCE_THRESHOLD = 0.8
SINGLE_CHAR_PENALTY = 0.5

# Text/drawing split (parse.py::_text_vectors_by_quad): a vector in an OCR'd
# cluster is text when more than TEXT_INK_INSIDE_FRAC of its ink (path
# length) lies inside a non-blank detect quad (the rotated quad, not its
# envelope) from its own cluster. Curves are sampled at INK_CURVE_SAMPLES
# points for this length measure only.
TEXT_INK_INSIDE_FRAC = 0.5
INK_CURVE_SAMPLES = 32
