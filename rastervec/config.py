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

# A classification cluster passes FAST if its own page-mask score exceeds
# this. FAST now runs before similarity grouping (no groups exist yet at
# this point), so this applies per cluster, independently -- kept at 0.5
# (its prior group-min value from when this threshold was last raised from
# an even older per-cluster value of 0.2) rather than reverted, pending
# real-world re-tuning now that the group-min check is gone.
FAST_COMBINED_KEEP_THRESHOLD = 0.5

# FastDetector.detect_tiled: FAST's own preprocessing always downsizes to a
# 640px short side, so a whole large page loses most of its resolution in
# one pass. Instead the render is upscaled by FAST_TILE_SCALE_FACTOR and cut
# into FAST_TILE_BLOCK_SIZE-square tiles, each detected once (no rotation).
FAST_TILE_BLOCK_SIZE = 2048
FAST_TILE_SCALE_FACTOR = 5

# FastDetector.detect_tiled: adjacent tiles overlap by this fraction of
# FAST_TILE_BLOCK_SIZE (stride = block_size * (1 - overlap)); overlapping
# regions are resolved by taking the max score, so a text line that would
# otherwise be split across a tile boundary is fully covered by at least
# one tile.
FAST_TILE_OVERLAP_FRAC = 0.15

# FastDetector.detect_tiled: a text-candidate segment's bbox is padded by
# this fraction of FAST_TILE_BLOCK_SIZE (in the same scaled-tile pixel
# space) before testing which tiles it overlaps, so a segment sitting
# right at a tile boundary isn't dropped by an off-by-one intersection.
FAST_TILE_CANDIDATE_MARGIN_FRAC = 0.05

# ======================================================================
# fast_first pipeline (pipelines/_fast_first_common.py) -- extract_vectors
# -> per-Vector FAST filter -> seqno-consecutive spatial cluster, skipping
# Vector_Classification entirely.
# ======================================================================

# filter_vectors_fast (pipelines/_steps.py): a Vector passes if its own
# per-item heatmap coverage (helpers.geometry.item_bbox per item, not the
# Vector's aggregate bbox) exceeds this.
FAST_VECTOR_KEEP_THRESHOLD = 0.5

# fast_first's spatial_cluster step: bbox-gap tolerance (PDF points) for
# Vector_Classification.group_filters.combine_overlapping_seq's
# seqno-consecutive chain-merge of FAST-surviving Vectors into
# text-candidate clusters. Kept independent of SEQ_OVERLAP_TOLERANCE_PX,
# which is tuned for the classification chain's very different
# post-item-filter merge.
FAST_FIRST_SEQ_MERGE_TOLERANCE = 10.0

# ======================================================================
# OCR (OCR/Paddle_OCR/ocr_backend.py)
# ======================================================================

# PaddleOCR model family the single OCR backend builds against. Pinned to
# PP-OCRv4 -- the last family shipped by paddleocr 2.x (see requirements.txt),
# which is also the API surface `archive/`'s raster_parser OCR targets, so
# the `legacy` benchmark variant needs no compatibility shim.
OCR_VERSION = "PP-OCRv4"

# Recognition language passed to `paddleocr.PaddleOCR(lang=...)` -- selects
# the recognition model (e.g. `en` -> `en_PP-OCRv4_rec`). Text detection is
# the Radon segmentation step, not PaddleOCR, so only the recogniser runs.
OCR_LANG = "en"

# ======================================================================
# Radon text segmentation (OCR/radon.py)
# ======================================================================

# Coarse skew full-sweep step (degrees), 0..180 -- locates the sweep regime
# (multi-line line-gap basin vs single-line `inf` band) and a rough angle
# that the fine sweep then pins. The gap-score objective is used at both
# stages; Postl/variance is only the fallback when neither structure shows.
RADON_COARSE_STEP_DEG = 3.0
# Fine skew-sweep half-range (degrees) around the rough coarse angle. The
# reported skew is NOT clamped to this -- the coarse sweep already found
# the basin/band wherever it is, so this only needs to cover the coarse
# grid's own +/-1.5-degree quantisation plus a margin.
RADON_SKEW_LIMIT_DEG = 8.0
# Max width (degrees) of the finite-objective window around the sweep
# minimum for it to count as a real multi-line line-gap basin (its centre
# is then the skew). A wider finite window means the projection keeps
# splitting into peaks at almost every angle -- a single text line -- and
# the skew comes from the narrow `inf` band instead.
RADON_MULTILINE_BASIN_MAX_DEG = 30.0
# Fine skew sweep step (degrees).
RADON_ANGLE_STEP_DEG = 0.25
# Moving-average window (px) applied to a projection profile before its
# peak / valley (gap) detection.
RADON_PROFILE_SMOOTH_PX = 3
# A smoothed-profile bin below this fraction of the profile's peak is not
# part of a peak (text-line) band.
RADON_PEAK_MIN_FRAC = 0.05
# eps in the gap-quality score
# `gap_score = 1 - ((L+R)/2 - M) / ((L+R)/2 + eps)` (L/R = the two peak
# heights, M = the valley minimum between them). Profile values are
# ink-pixel counts, so an absolute 1.0 keeps a literally-zero valley near
# gap_score 0 ("clean gap") and never divides by zero.
RADON_GAP_SCORE_EPS = 1.0
# A profile valley whose gap_score is below this is a real line/word
# boundary; above it the two peaks belong to the same text line (a shallow
# valley bridged by descenders, dotted rows, ...).
RADON_GOOD_GAP_MAX = 0.5
# A text line is split at word gaps only as far as needed to keep every
# segment's aspect ratio (segment width / line ink height) under this -- so
# "I love pineapples very much" becomes a few OCR-friendly chunks rather
# than one absurdly wide crop. A single word wider than this is never split.
RADON_MAX_SEGMENT_ASPECT = 10.0
# Cap (px) on a cluster render's long side before the Radon sweep -- angle
# estimation is scale-invariant, so a big merged bbox is downscaled to
# this first to keep the O(pixels * angles) transform fast.
RADON_MAX_RENDER_SIDE_PX = 800
# Floor (px) on the cluster-wide word-split gap threshold (1.3x the median
# inter-run gap pooled across every line in the cluster) -- guards the
# degenerate case where that median is at or near zero (very tight kerning,
# or mostly single-run lines).
RADON_MIN_GAP_PX = 2.0
# Multiplier on that pooled median gap. The median inter-run gap in a
# cluster is the typical *intra-word* letter gap (letter gaps outnumber
# word gaps), so a threshold just above it separates words from letters.
RADON_GAP_MEDIAN_MULTIPLIER = 1.3
# White border `OCR/radon.py::pad_image` adds, as a fraction of the image's
# own width (left/right) and height (top/bottom). Applied once to a whole
# cluster render before deskew/word-split, and once to each word crop before
# it becomes `Segment.image` -- the crop PaddleOCR recognizes.
RADON_PAD_FRACTION = 0.1
# Fixed extra margin (PDF points) added on top of half the cluster's own max
# stroke width when `segment_clusters` renders a cluster for Radon
# (`render_cluster_for_radon(..., padding=...)`) -- so a stroke's
# anti-aliased edge is never clipped exactly at the render frame. Distinct
# from `RADON_PAD_FRACTION`'s pixel-space post-render border.
RADON_RENDER_PADDING_EXTRA_PT = 0.5
# Minimum ink-run count (the pre-OCR proxy for character count) a split
# word must have -- a shorter word merges into a neighbor regardless of
# the gap between them, since PaddleOCR reads a too-short word's
# orientation poorly. Outranks RADON_MIN_GAP_PX's gap-based split.
RADON_MIN_WORD_CHARS = 3

# RenderOCR: a cluster render whose shorter side would fall under this many
# pixels at the requested dpi is bumped to a higher effective dpi instead
# -- PaddleOCR reads tiny crops poorly, and the Radon skew sweep needs
# enough rows to resolve the inter-line gaps of a multi-line block.
MIN_RENDER_SIDE_PX = 100
# Hard ceiling on that bump. A degenerate cluster (a sub-point bbox from
# stray CAD geometry) would otherwise demand an unbounded dpi to reach
# MIN_RENDER_SIDE_PX and rasterize to a multi-gigabyte pixmap. Content that
# small carries no readable glyph anyway -- it renders blank and is dropped
# by segment_clusters' own no-ink check.
MAX_RENDER_DPI = 4800

# batch size for both the classification (`cls_batch_num`) and
# recognition (`rec_batch_num`) PaddleOCR calls, and for
# `recognize_segments`'s own batching over `Segment`s.
OCR_BATCH_SIZE = 128
