"""Tunable thresholds for the LegacyRecreation P3 backend -- a from-scratch
port of archive/raster_parser's own Type-2 fill-glyph classification +
seqno word-grouping + PaddleOCR recognition algorithm
(archive/raster_parser/rendering/pdf_render/reconstruct.py::filter_text_vectors
+ archive/raster_parser/ocr/wordgrouping.py), onto commons.models types.
Self-contained -- not shared with VectorClassification/FastIntoPaddle."""
from __future__ import annotations

# filter_out_white_vectors: a vector filled exactly this color is page
# background, never glyph ink.
WHITE_FILL = (1.0, 1.0, 1.0)

# is_box_or_rule: a fill vector whose bbox aspect ratio exceeds this is a
# ruled line, not text.
BOX_RULE_ASPECT_MIN = 10.0
# ...and whose aspect ratio is under this (and not white/near-white/black)
# is a panel/box fill, not text.
BOX_SQUARE_ASPECT_MAX = 3.0
BOX_MIN_SIDE_PX = 10.0

OCR_VERSION = "PP-OCRv4"
OCR_LANG = "en"
OCR_DPI = 300
