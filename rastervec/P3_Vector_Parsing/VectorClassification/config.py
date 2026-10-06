"""Tunable thresholds for the VectorClassification P3 backend (a reduced
2-step Vector_Classification chain + PaddleOCR detect+recognize, page-wide
batched, then an optional post-recognition FAST text/drawing split --
every classification cluster still goes straight to OCR). Self-contained -- not shared with LegacyRecreation's
own config.py, per the phase-isolation rule (see CLAUDE.md's "sibling
backends share zero code" rule)."""
from __future__ import annotations

# ======================================================================
# Vector Classification -- the reduced 2-step chain in classify_vectors.py.
# ======================================================================

SEQ_OVERLAP_TOLERANCE_PX = 1.0
SPATIAL_CLUSTER_THRESHOLD = 10.0
SPATIAL_SIZE_TOLERANCE = 0.30

# ======================================================================
# Raster-refined rotation (paddle_engine.py::hough_deskew)
# ======================================================================

# Grayscale value (0-255) below which a pixel counts as "ink" when building
# the binary mask Hough line detection runs on.
HOUGH_INK_THRESHOLD = 200
# Binary-dilation disk radius (px) applied to the ink mask before Hough --
# thickens thin/broken strokes so Hough has more to find ("increase ink").
HOUGH_DILATE_RADIUS_PX = 2
# Grayscale value (0-255) below which a pixel counts as "ink" when building
# the (non-dilated) mask `cv2.minAreaRect` runs on -- independently tunable
# from HOUGH_INK_THRESHOLD since the two estimators may want different
# sensitivity in practice.
MINAREA_INK_THRESHOLD = 200
# The combined (Hough + minAreaRect) rotation correction is snapped to the
# nearest multiple of this many degrees before being applied.
HOUGH_ANGLE_SNAP_DEG = 10.0
# Only trust the combined Hough+minAreaRect rotation reading if the two
# (each reduced mod 90) are within this many degrees of each other;
# otherwise the correction defaults to 0 (no rotation) rather than
# trusting either reading alone.
ROTATION_AGREEMENT_TOLERANCE_DEG = 1.0

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

# ======================================================================
# FAST text/drawing split (fast_filter.py, fast_detect.py) -- after
# recognition, which vectors under a detect box are really text.
# ======================================================================

# Off -> the old rule: every vector connected (bbox overlap, transitively)
# to a non-blank quad is text (`parse._drawing_extra_vectors`).
FAST_FILTER_ENABLED = True
# Detect quads of one cluster are grouped into one FAST crop, anchored on a
# seed quad (no chaining): a quad joins a seed only if their page bboxes
# overlap AND their centers are closer than this (pt).
FAST_GROUP_CENTER_DIST_PT = 10.0
# White margin (px, cluster-render pixels) around a group's union rect.
FAST_CROP_PADDING_PX = 8
# FAST rescales its input's short side to 640 px; a one-line label crop
# (e.g. 30x600 px) would be upscaled ~20x, far past the text sizes FAST was
# trained on. The crop is white-padded (not resized) so its long side is at
# most this many times its short side before FAST sees it.
FAST_CROP_MAX_ASPECT = 4.0
# A pixel is "highlighted" when FAST's score there is >= this.
FAST_HEAT_THRESHOLD = 0.5
# A vector stays text when at least this fraction of its own ink pixels
# (the whole vector, pixels outside the group crop count as cold) is
# highlighted; otherwise it goes to drawing.
FAST_INK_FRACTION = 0.5
# Grayscale value (0-255) below which a single-vector render pixel is ink.
FAST_INK_GRAY_THRESHOLD = 250
# Heatmap downscale cap (px, longest side) kept for the `fast / heatmap`
# debug layer -- full-size heatmaps are only kept with keep_debug_arrays.
FAST_DEBUG_HEATMAP_MAX_SIDE = 256

# ======================================================================
# Line geometry (line_geometry.py) -- debug-only collinear/parallel group
# layers per classification cluster. Same values as CollinearVectorClass.
# ======================================================================

# Max point distance (pt) from the fitted line for a Vector to be straight.
STRAIGHT_TOL_PT = 0.25
# Single-linkage angle tolerance (deg, folded to [0, 180)).
ANGLE_TOL_DEG = 2.0
# Anchored perpendicular-offset tolerance (pt) for collinear grouping.
COLLINEAR_OFFSET_TOL_PT = 1.0
# Groups smaller than this are drawn as singletons (gray).
MIN_GROUP_SIZE = 2
