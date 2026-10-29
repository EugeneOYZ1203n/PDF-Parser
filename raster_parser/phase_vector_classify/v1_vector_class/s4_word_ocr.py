"""Step 4 -- OCR each word group: render it alone (step 3), pad and scale the crop to
PaddleOCR's recognition height, read it, keep the most confident word.

Long groups (aspect >= 8 or >= 20 glyphs) are also read unscaled, and the more confident of
the two readings wins.

Copied from `parsing/parser.py`: `PdfRasterParser`'s word-group OCR methods, as functions
taking the page instead of a parser instance.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np
import pymupdf

from raster_parser.models.vector import Vector
from raster_parser.commons.v1_commons.ocr import normalise_crop_for_ocr
from raster_parser.phase_vector_classify.v1_vector_class.s3_glyph_render import PngRenderer
from raster_parser.phase_vector_classify.v1_vector_class.s4_paddle_ocr import PaddleOcr

# A word group this many times wider than tall, or built from this many glyphs, is a long
# line of text. It is also recognised from its raw crop, since normalising squeezes it into
# OCR_TARGET_WIDTH_PX, and the more confident of the two readings wins.
LARGE_GROUP_MIN_ASPECT = 8
LARGE_GROUP_MIN_VECTORS = 20


@lru_cache(maxsize=None)
def engine_for(engine_cls: type, use_angle_cls: bool, lang: str, drop_score: float):
    """The PaddleOCR wrapper for these settings, built once per process: loading the models costs
    far more than reading a page's word groups, and every page of a run shares the settings."""
    return engine_cls(use_angle_cls=use_angle_cls, lang=lang, drop_score=drop_score)


def ocr_word_groups(
    page: pymupdf.Page,
    vector_groups: List[List[Vector]],
    group_boxes: List[tuple[float, float, float, float]],
    *,
    dpi: int = 300,
    min_confidence: float = 0.0,
    lang: str = "en",
    use_angle_cls: bool,
    on_crop=None,
) -> List[Tuple[int, str, float]]:
    """OCR pre-grouped vector crops; one `(group index, word, confidence)` per group that read as a word.

    `on_crop(index, crop)`, when given, is shown each group\'s normalised crop (what PaddleOCR
    reads first); it is a debug hook and changes nothing.
    """
    if not vector_groups:
        return []
    if len(vector_groups) != len(group_boxes):
        raise ValueError("vector_groups and group_boxes length mismatch")

    paddle_ocr = engine_for(PaddleOcr, use_angle_cls, lang, min_confidence)

    renderer = PngRenderer(dpi=dpi, page=page)
    out: List[Tuple[int, str, float]] = []
    for i, vectors in enumerate(vector_groups):
        if not vectors:
            continue
        crop = np.array(renderer.render_word_group(vectors))
        if crop.size == 0:
            continue
        norm = normalise_crop_for_ocr(crop)
        if on_crop is not None:
            on_crop(i, norm)
        reading = _read_word(paddle_ocr, crop, norm, len(vectors), min_confidence, use_angle_cls)
        if reading is not None:
            out.append((i, *reading))
    return out


def _read_word(
    paddle_ocr: PaddleOcr,
    crop: np.ndarray,
    norm: np.ndarray,
    n_glyphs: int,
    min_confidence: float,
    use_angle_cls: bool,
) -> tuple[str, float] | None:
    """`(word, confidence)` of the most confident word in `crop`; a long group is read scaled and
    unscaled."""
    h, w = crop.shape[:2]
    hits = paddle_ocr.ocr_image(norm, min_confidence=min_confidence, cls=use_angle_cls)
    best = max(hits, key=lambda hit: hit[2]) if hits else None
    if w > h and (w / max(h, 1) >= LARGE_GROUP_MIN_ASPECT or n_glyphs >= LARGE_GROUP_MIN_VECTORS):
        hits_raw = paddle_ocr.ocr_image(crop, min_confidence=min_confidence, cls=use_angle_cls)
        best_raw = max(hits_raw, key=lambda hit: hit[2]) if hits_raw else None
        if best_raw and (not best or best_raw[2] >= best[2]):
            best = best_raw
    return (best[1], best[2]) if best else None
