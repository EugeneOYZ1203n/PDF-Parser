"""Tunable thresholds for the VectorClassification P3 backend (a reduced
2-step Vector_Classification chain + PaddleOCR detect+recognize, page-wide
batched -- there is no FAST filtering stage; every classification cluster
goes straight to OCR). Self-contained -- not shared with LegacyRecreation's
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
# Rotation-correction gating (parse.py::_quad_allows_rotation): a detected
# quad's own bbox must be at least this elongated (orientation-agnostic
# max(w,h)/min(w,h) -- a tall vertical run of text counts the same as a
# wide horizontal one) for a raster-refined rotation correction to be
# attempted at all. Below this, or if the quad's own vectors form only one
# connected component, rotation defaults to 0 without running
# Hough/minAreaRect at all.
ROTATION_MIN_ASPECT_RATIO = 3.0

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
