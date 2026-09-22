"""Tunable thresholds for the VectorClassification P3 backend (the old
12-step Vector_Classification chain + Radon word-segmentation + PaddleOCR
recognition-only OCR). Self-contained -- not shared with FastIntoPaddle's
own config.py, per the phase-isolation rule (see the repo's plan doc)."""
from __future__ import annotations

# ======================================================================
# Vector Classification -- the fixed 12-step chain in classification.py.
# Step numbers below match that module's docstring.
# ======================================================================

MAX_DIMENSION_FRACTION = 0.10
SIGNATURE_ROUND_PX = 0.5
DUPLICATE_RUN_MIN_LENGTH = 5
SEQ_OVERLAP_TOLERANCE_PX = 1.0
MIN_GROUP_SIZE_PX = 1.0
SPATIAL_CLUSTER_THRESHOLD = 10.0
SPATIAL_SIZE_TOLERANCE = 0.30
PERIMETER_MARGIN_FRACTION = 0.1
DENSITY_DEFAULT_GRID_SIZE = 4
DENSITY_MIN_CELL_PX = 5.0
DENSITY_MAX_CELL_PX = 40.0
DENSITY_MAX_EMPTY_FRACTION = 0.70
PATTERN_SPACING_TOLERANCE = 0.20
PATTERN_MIN_REPEAT_COUNT = 3
PATTERN_FRACTION_THRESHOLD = 0.70
LOW_VARIETY_MIN_MEMBER_COUNT = 5
LOW_VARIETY_MIN_REQUIRED = 1
LOW_VARIETY_MAX_MEMBER_COUNT = 300
LOW_VARIETY_MAX_REQUIRED = 10

# ======================================================================
# fast_text_detect stage (fast_filter.py / fast_detect.py)
# ======================================================================

FAST_PAGE_RENDER_DPI = 150
FAST_TILE_BLOCK_SIZE = 2048
FAST_TILE_SCALE_FACTOR = 2
FAST_TILE_CANDIDATE_MARGIN_FRAC = 0.05
FAST_TILE_OVERLAP_FRAC = 0.15
# Combined FAST score threshold above which a cluster is kept as text.
FAST_COMBINED_KEEP_THRESHOLD = 0.5

# ======================================================================
# OCR (paddle_engine.py, wordgrouping.py, parse.py)
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

# _cluster_render_padding (parse.py): page-space PDF-point margin added to a
# word group's own render frame, on top of half its own max stroke width --
# matches LegacyRecreation/config.py's own RENDER_PADDING_EXTRA_PT (which in
# turn matches archive's original PngRenderer.render_word_group flat
# padding=5.0pt).
RENDER_PADDING_EXTRA_PT = 5.0

# ocr_prep.pad_image_uniform (parse.py): pixel-space white border added to a
# word group's render before PaddleOCR's detector sees it, sized off the
# render's own larger dimension -- matches PADDLE_WHITE_PAD_FRACTION
# everywhere else in the codebase.
RECOGNITION_PAD_FRACTION = 0.1
