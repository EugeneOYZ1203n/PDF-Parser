"""Step 4 -- OCR each word group: render it alone (step 3), pad and scale the crop to
PaddleOCR's recognition height, read it, keep the most confident word.

Long groups (aspect >= 8 or >= 20 glyphs) are also read unscaled, and the more confident of
the two readings wins.

Copied from `parsing/parser.py`: `PdfRasterParser`'s word-group OCR methods, as functions
taking the page instead of a parser instance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import pymupdf as pymupdf

from raster_parser.commons.models import Vector
from raster_parser.commons.progress import bar
from raster_parser.P3_Vector_Parsing.v1VectorClass.s3_glyph_render import PngRenderer
from raster_parser.P3_Vector_Parsing.v1VectorClass.s2_word_grouping import Orientation
from raster_parser.P3_Vector_Parsing.v1VectorClass.s4_paddle_ocr import PaddleOcr


@dataclass
class OcrWord:
    """One OCR result: the word group's box, the text read and the group's index as `block_no`.
    `parse.py` sets `rotate` / `glyph_orientation` and converts it to the shared `Text`."""

    x0: float
    y0: float
    x1: float
    y1: float
    word: str
    block_no: int
    line_no: int
    word_no: int
    rotate: int = 0
    glyph_orientation: Orientation = Orientation.HORIZONTAL


@dataclass(frozen=True)
class _PreparedPaddleGroup:
    index: int
    box: tuple[float, float, float, float]
    norm: np.ndarray
    crop: np.ndarray
    large_group: bool


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
) -> List[Tuple[OcrWord, List[int]]]:
    """OCR pre-grouped vector crops; one `(OcrWord, seqnos)` per group that read as a word.

    Each word has the group's box and `block_no` = the group's index. `on_crop(index, crop)`,
    when given, is shown each group's normalised crop (what PaddleOCR reads first); it is a
    debug hook and changes nothing.
    """
    if not vector_groups:
        return []
    if len(vector_groups) != len(group_boxes):
        raise ValueError("vector_groups and group_boxes length mismatch")

    paddle_ocr = PaddleOcr(
        use_angle_cls=use_angle_cls,
        lang=lang,
        drop_score=min_confidence,
    )

    renderer = PngRenderer(dpi=dpi, page=page)
    out: List[Tuple[OcrWord, List[int]]] = []

    for i, (vectors, box) in enumerate(bar(zip(vector_groups, group_boxes), desc="word groups",
                                           total=len(group_boxes), unit="group")):
        prepared, vector_indices = _prepare_paddle_word_group(
            i, vectors, box, renderer
        )
        if prepared is None:
            continue
        if on_crop is not None:
            on_crop(i, prepared.norm)
        text, vector_indices = _paddle_ocr_prepared(
            paddle_ocr,
            prepared,
            vector_indices,
            min_confidence=min_confidence,
            use_angle_cls=use_angle_cls,
        )
        if text is not None:
            out.append((text, vector_indices))

    return out


def _prepare_paddle_word_group(
    index: int,
    vectors: List[Vector],
    box: tuple[float, float, float, float],
    renderer: PngRenderer,
    orientation: Orientation = Orientation.HORIZONTAL,
) -> Tuple[_PreparedPaddleGroup | None, List[int]]:
    if not vectors:
        return None, []
    vector_indices = [v.seqno for v in vectors]
    x0, y0, x1, y1 = box
    crop = rotate_crop_for_ocr(
        np.array(renderer.render_word_group(vectors)),
        orientation,
    )
    if crop.size == 0:
        return None, []
    h, w = crop.shape[:2]
    norm = normalise_crop_for_ocr(crop)
    large_group = w > h and (w / max(h, 1) >= 8 or len(vectors) >= 20)
    return _PreparedPaddleGroup(
        index=index,
        box=(x0, y0, x1, y1),
        norm=norm,
        crop=crop,
        large_group=large_group,
    ), vector_indices


def _paddle_ocr_prepared(
    paddle_ocr: PaddleOcr,
    prep: _PreparedPaddleGroup,
    vector_indices: List[int],
    *,
    min_confidence: float,
    use_angle_cls: bool,
) -> Tuple[OcrWord | None, List[int]]:
    x0, y0, x1, y1 = prep.box
    hits = paddle_ocr.ocr_image(
        prep.norm, min_confidence=min_confidence, cls=use_angle_cls
    )
    if prep.large_group:
        hits_raw = paddle_ocr.ocr_image(
            prep.crop, min_confidence=min_confidence, cls=use_angle_cls
        )
        best_norm = max(hits, key=lambda h: h[2]) if hits else None
        best_raw = max(hits_raw, key=lambda h: h[2]) if hits_raw else None
        if best_raw and (not best_norm or best_raw[2] >= best_norm[2]):
            best_word = best_raw[1]
        elif best_norm:
            best_word = best_norm[1]
        else:
            return None, vector_indices
    elif hits:
        best_word = max(hits, key=lambda h: h[2])[1]
    else:
        return None, vector_indices
    return OcrWord(x0, y0, x1, y1, best_word, prep.index, 0, 0), vector_indices


def rotate_crop_for_ocr(
    crop: np.ndarray,
    orientation: Orientation,
) -> np.ndarray:
    """Turn vertical word-group crops upright using deduced line orientation."""
    if crop.size == 0:
        return crop
    if orientation == Orientation.VERTICAL:
        return cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop


def normalise_crop_for_ocr(crop: np.ndarray, target_height: int = 48, target_width: int = 1024) -> np.ndarray:
    if crop.size == 0:
        return crop

    h, w = crop.shape[:2]
    if h <= 0 or w <= 0:
        return crop

    # 1. FIX THE IMBALANCE:
    # We want VERY tight vertical padding so the letters fill the height,
    # but GENEROUS horizontal padding so the edges don't clip.
    pad_h = max(2, int(h * 0.05))   # Only 5% vertical padding (keeps text tall)
    pad_w = max(40, int(h * 0.30))  # 30% horizontal padding cushion based on font height

    padded_crop = cv2.copyMakeBorder(
        crop,
        top=pad_h, bottom=pad_h, left=pad_w, right=pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255] if len(crop.shape) == 3 else 255
    )

    # Recalculate padded dimensions
    hp, wp = padded_crop.shape[:2]

    # 2. PROPORTIONAL ASPECT SCALE (Drop the rigid 320px width restriction!)
    # Let the width expand naturally to accommodate long strings like your image
    scale = target_height / hp
    new_w = max(1, int(wp * scale))

    # 3. CRITICAL OVERRIDE: Ensure it fits PaddleOCR's recognition shape ceiling
    # If the text is incredibly long, clamp it to your max target width parameter
    final_w = min(new_w, target_width)

    # 4. RESIZE PRESERVING SHAPE GEOMETRY
    if scale > 1.0:
        resized = cv2.resize(padded_crop, (final_w, target_height), interpolation=cv2.INTER_CUBIC)
    else:
        resized = cv2.resize(padded_crop, (final_w, target_height), interpolation=cv2.INTER_AREA)

    return resized
