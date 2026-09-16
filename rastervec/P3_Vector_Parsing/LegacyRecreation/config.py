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

# _cluster_render_padding (parse.py): page-space PDF-point margin added to a
# word group's own render frame, on top of half its own max stroke width --
# archive's original PngRenderer.render_word_group used a flat padding=5.0pt,
# so this is that same value rather than the smaller "extra" constant
# VectorClassification/FastIntoPaddle use for their own stroke-derived margin.
RENDER_PADDING_EXTRA_PT = 5.0

# dpi_for_cluster (paddle_engine.py): never-reduces, capped dpi bump so a
# small word group's render still reaches a usable pixel size for PaddleOCR's
# own detector to find text in -- own duplicated copy of FastIntoPaddle's
# PADDLE_DETECT_MIN_RENDER_SIDE_PX value. Unlike FastIntoPaddle, this backend
# has only one render per word group (it feeds both detection and
# recognition, no lower-resolution post-detection crop stage), so one
# constant at the detection-appropriate resolution is correct here.
MIN_RENDER_SIDE_PX = 200
MAX_RENDER_DPI = 4800

# pad_image (paddle_engine.py): pixel-space white border added to a word
# group's render before PaddleOCR's detector sees it, sized off the render's
# own larger dimension -- matches PADDLE_WHITE_PAD_FRACTION everywhere else
# in the codebase.
RECOGNITION_PAD_FRACTION = 0.1
