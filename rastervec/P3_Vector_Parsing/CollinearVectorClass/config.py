"""Tunable thresholds for the CollinearVectorClass P3 backend -- the
VectorClassification chain (layer/color/width buckets -> seqno-overlap merge
-> constrained spatial clustering -> PaddleOCR detect + recognize) plus
collinear drawing removal, per-cluster parallel-group length outliers,
crossing-count removal, and parallel-group-driven rotation before detect and
before recognition. Self-contained -- not shared with any sibling backend's
config.py."""
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
# collinear and parallel grouping, and for deduping global potential angles.
ANGLE_TOL_DEG = 1.0
# Anchored perpendicular-offset tolerance (pt) for collinear grouping.
COLLINEAR_OFFSET_TOL_PT = 0.5
# A collinear group becomes drawing when its member-length std (pt) is below
# this AND it has more than COLLINEAR_DRAWING_MIN_COUNT members -- a long
# dashed/repeated line, never text.
COLLINEAR_DRAWING_MAX_STD_PT = 5.0
COLLINEAR_DRAWING_MIN_COUNT = 50
# Minimum members for a set of same-angle straight Vectors to count as a
# "parallel group" (outlier pooling, rotation rules). Singletons don't.
MIN_PARALLEL_GROUP_SIZE = 2
# Per cluster, strokes in parallel groups whose length is more than this many
# (pooled) standard deviations from the pooled mean are removed to drawing.
LENGTH_OUTLIER_STD = 2.0
# Per cluster (after outlier removal), Vectors properly crossed by more than
# this many distinct foreign segments are removed to drawing.
MAX_CROSSING_SEGMENTS = 10
# Proper-crossing orientation epsilon (pt) and curve flattening steps.
CROSS_EPS_PT = 0.01
CURVE_STEPS = 8

# ======================================================================
# Hough rotation (paddle_engine.py::hough_angle_deg) -- only used for a
# detected quad over more than one connected component with zero or several
# parallel groups; the result is mod 180 and snapped (see rotation.py).
# ======================================================================

# Grayscale value (0-255) below which a pixel counts as "ink" for Hough.
HOUGH_INK_THRESHOLD = 200
# Binary-dilation disk radius (px) applied to the ink mask before Hough.
HOUGH_DILATE_RADIUS_PX = 2

# ======================================================================
# OCR (paddle_engine.py, parse.py)
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
# cluster's own render frame, on top of half its own max stroke width. This
# is now the *only* padding in the OCR render path (the old pixel-space
# post-render white border was removed), so it's intentionally larger than
# LegacyRecreation/config.py's own RENDER_PADDING_EXTRA_PT=5.0pt (which still
# matches archive's original PngRenderer.render_word_group flat padding).
RENDER_PADDING_EXTRA_PT = 15.0
