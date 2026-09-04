"""One home for every tunable threshold in the extraction pipeline.

Each constant below is a knob you can turn per-PDF when a page's default
classification / OCR looks wrong; there is no runtime or UI way to change
them. Grouped by the stage that reads it. Every value is pinned by a named
assertion in `tests/rastervec/test_config.py` -- change a value there and
that test tells you which behaviour you just moved.

Model-architecture constants that are faithful ports of an upstream repo
(the FAST TextNet-Tiny NAS config, ImageNet normalisation, PaddleOCR
recognition-model names) deliberately stay next to their code, not here --
they are not tuning knobs.
"""
from __future__ import annotations

# ======================================================================
# Vector Classification -- the fixed 12-step chain in
# Vector_Classification/classification.py. Step numbers below match that
# module's docstring.
# ======================================================================

# Steps 1 & 5: an item / group whose own bbox's larger side exceeds this
# fraction of the page's *smaller* side is border/frame geometry, dropped.
MAX_DIMENSION_FRACTION = 0.10

# Step 2: grid size (PDF points) that a shape signature's points are
# rounded to. Two paths that are pure translations of each other round to
# the same signature; smaller = stricter "same shape" test.
SIGNATURE_ROUND_PX = 0.5

# Step 3a: a run of this many or more consecutive same-signature items (in
# seq order) is dropped whole -- a hatching / tick strip, not text.
DUPLICATE_RUN_MIN_LENGTH = 5

# Step 3b: bbox gap tolerance (PDF points) when chain-merging the survivors
# into groups by seq order.
SEQ_OVERLAP_TOLERANCE_PX = 1.0

# Step 3c: drop a group whose aggregate bbox's larger side is under this
# many points -- a leftover speck too small to be a glyph.
MIN_GROUP_SIZE_PX = 1.0

# Step 6: bbox gap tolerance (PDF points) for the single-linkage spatial
# merge of groups into clusters.
SPATIAL_CLUSTER_THRESHOLD = 10.0

# Step 6: two groups only spatially merge if a "valid" side of one is
# within this relative difference of a parallel valid side of the other.
SPATIAL_SIZE_TOLERANCE = 0.30

# Step 9: drop a cluster if every member sits in this fraction of the
# cluster bbox's perimeter band, never touching the shrunk-in centre
# (border/ring geometry).
PERIMETER_MARGIN_FRACTION = 0.1

# Step 10: default grid cells per axis for the density check, then clamped
# so each cell's side stays within [DENSITY_MIN_CELL_PX, DENSITY_MAX_CELL_PX].
DENSITY_DEFAULT_GRID_SIZE = 4
DENSITY_MIN_CELL_PX = 5.0
DENSITY_MAX_CELL_PX = 40.0
# Step 10: drop a cluster if more than this fraction of its grid cells
# have no member touching them -- too sparse to be text.
DENSITY_MAX_EMPTY_FRACTION = 0.70

# Step 11: within a same-shape sub-group, consecutive members must sit at a
# gap whose max deviation from the mean gap (relative to the mean) is
# within this to count as "constant spacing".
PATTERN_SPACING_TOLERANCE = 0.20
# Step 11: a sub-group needs at least this many members before its spacing
# is judged at all.
PATTERN_MIN_REPEAT_COUNT = 3
# Step 11: drop the whole cluster if members of constant-spacing sub-groups
# together make up at least this fraction of it.
PATTERN_FRACTION_THRESHOLD = 0.70

# Step 12: a cluster must contain at least a log-scale-ramped number of
# distinct shape signatures for its member count. At or under
# LOW_VARIETY_MIN_MEMBER_COUNT members only LOW_VARIETY_MIN_REQUIRED are
# needed; at or over LOW_VARIETY_MAX_MEMBER_COUNT, LOW_VARIETY_MAX_REQUIRED.
LOW_VARIETY_MIN_MEMBER_COUNT = 5
LOW_VARIETY_MIN_REQUIRED = 1
LOW_VARIETY_MAX_MEMBER_COUNT = 300
LOW_VARIETY_MAX_REQUIRED = 10

# ======================================================================
# unique_clusters stage (pipeline.py) -- whole-page similarity grouping
# ======================================================================

# Two clusters count as "the same shape" if, once translation+rotation
# normalised, every corresponding point pair sits within this fraction of
# the larger cluster's own bbox max dimension.
UNIQUE_CLUSTER_TOLERANCE = 0.04

# ======================================================================
# fast_text_detect stage (pipeline.py)
# ======================================================================

# DPI the whole-page FAST render is rasterized at, before
# FastDetector.detect_tiled's own further upscale. Not full OCR resolution
# (RenderOCR's per-cluster renders use 300 DPI).
FAST_PAGE_RENDER_DPI = 150

# A text-candidate cluster passes FAST if its combined score (its own
# page-mask score, min'd across its similarity group) exceeds this.
FAST_COMBINED_KEEP_THRESHOLD = 0.2

# FastDetector.detect_tiled: FAST's own preprocessing always downsizes to a
# 640px short side, so a whole large page loses most of its resolution in
# one pass. Instead the render is upscaled by FAST_TILE_SCALE_FACTOR, cut
# into FAST_TILE_BLOCK_SIZE-square tiles, and each tile detected at
# FAST_TILE_ROTATION_COUNT evenly-spaced rotations (averaged).
FAST_TILE_BLOCK_SIZE = 2048
FAST_TILE_SCALE_FACTOR = 5
FAST_TILE_ROTATION_COUNT = 4

# ======================================================================
# spatial_regroup stage (pipeline.py)
# ======================================================================

# Two FAST-passed clusters in the same (layer, color) bucket merge before
# OCR if their aggregate bboxes are within this gap (PDF points; rect_gap
# is 0.0 for overlapping/touching boxes).
SPATIAL_REGROUP_TOLERANCE_PX = 1.0

# ======================================================================
# ocr_compare stage (pipeline.py)
# ======================================================================

# Default OCR backend: True -> LightPaddleOcrBackend (own ink-projection
# segmentation + PaddleOCR recognition-only); False -> the full PP-OCRv6
# detect+rec+orient pipeline. A PipelineContext.ocr_backend override wins.
USE_LIGHT_OCR_BACKEND = True

# RenderOCR: a cluster render whose shorter side would fall under this many
# pixels at the requested dpi is bumped to a higher effective dpi instead
# -- PaddleOCR reads tiny crops poorly.
MIN_RENDER_SIDE_PX = 50

# ======================================================================
# renderer/png.py -- OCR / FAST input rasterization
# ======================================================================

# Minimum padding (PDF points) around a cluster's bbox before rendering,
# so a thin stroke right at the edge isn't clipped.
MIN_CLUSTER_PADDING = 4.0
# OCR render-border expansion, as fractions of the cluster bbox height,
# applied asymmetrically: tight vertically (glyphs stay tall), generous
# horizontally (edge glyphs don't clip). Also used by
# OCR/Paddle_OCR/crop_normalize.py for the post-render crop pad.
OCR_VERTICAL_PADDING_FRACTION = 0.05
OCR_HORIZONTAL_PADDING_FRACTION = 0.30

# crop_normalize.py: fixed recognition line height / max width (px) the
# legacy PaddleOCR crop is resized to.
REC_LINE_HEIGHT_PX = 48
REC_LINE_MAX_WIDTH_PX = 1024

# ======================================================================
# LightPaddleOcrBackend (OCR/Paddle_OCR/light_backend.py)
# ======================================================================

# doc-orientation classifier score below this -> ignore it, fall back to
# aspect gating for the 0/90/270 decision.
DOC_ORI_MIN_CONFIDENCE = 0.7
# a crop taller than this multiple of its width is treated as vertical
# text for the aspect-gating fallback.
VERTICAL_ASPECT = 1.5
# batch size for the one recognition-only TextRecognition.predict call.
REC_BATCH_SIZE = 128
