"""Tunable thresholds for the FastIntoPaddle P3 backend (similarity grouping
-> per-Vector FAST filter -> reclassify -> layer/color/width separation ->
spatial clustering -> per-cluster PaddleOCR detect -> overlap reassignment
-> rotation refine -> PaddleOCR recognize). Self-contained -- not shared
with VectorClassification's own config.py, per the phase-isolation rule."""
from __future__ import annotations

# similarity.py (this backend's own copy) -- two normalized point clouds
# count as "the same shape" if their mean squared corresponding-point
# distance is at or under this.
SIMILARITY_MSE_THRESHOLD = 0.01

# reclassify_by_similarity: if under this fraction of a group's members
# failed FAST, the whole group passes.
RECLASSIFY_PASS_FRACTION = 0.10

# filter_vectors_fast: a Vector passes if its own per-item heatmap coverage
# exceeds this.
FAST_VECTOR_KEEP_THRESHOLD = 0.1

# filter_vectors_fast: report-only downsample factor for the verbose debug
# image, never applied to the image FastDetector actually scores.
FAST_DEBUG_IMAGE_SCALE = 0.5

FAST_PAGE_RENDER_DPI = 150
FAST_TILE_BLOCK_SIZE = 2048
FAST_TILE_SCALE_FACTOR = 2
FAST_TILE_CANDIDATE_MARGIN_FRAC = 0.05
FAST_TILE_OVERLAP_FRAC = 0.15

# cluster_buckets: base bbox-gap tolerance (PDF points) for the union-find
# spatial merge, plus its dynamic scale/cap.
FAST_PADDLE_SEQ_MERGE_TOLERANCE = 15.0
CLUSTER_TOLERANCE_SCALE = 0.05
CLUSTER_TOLERANCE_MAX = 200.0

PADDLE_DETECT_MIN_RENDER_SIDE_PX = 200
PADDLE_DETECT_MAX_RENDER_DPI = 4800

# reassign_by_overlap: minimum bbox-area coverage for a Vector to be
# reassigned to a PaddleOCR-detected box.
PADDLE_REASSIGN_MIN_COVERAGE = 0.5

PADDLE_WHITE_PAD_FRACTION = 0.1

# OCR/radon.py::sweep_rotation -- refine PaddleOCR's coarse rotation by
# sweeping this many degrees on either side of it.
VECTOR_RADON_SWEEP_RANGE_DEG = 10.0
VECTOR_RADON_SWEEP_STEP_DEG = 0.5
VECTOR_RADON_LINE_SPACING_PT = 1.0
VECTOR_RADON_MAX_LINES = 600

# cluster_render_padding: fixed extra margin (PDF points) on top of half the
# cluster's own max stroke width.
RADON_RENDER_PADDING_EXTRA_PT = 0.5

OCR_VERSION = "PP-OCRv4"
OCR_LANG = "en"
OCR_BATCH_SIZE = 128
MIN_RENDER_SIDE_PX = 100
MAX_RENDER_DPI = 4800
