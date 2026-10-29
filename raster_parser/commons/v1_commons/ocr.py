"""OCR helpers both v1 backends use: crop normalisation for PaddleOCR recognition and the
OCR word -> shared `Text` conversion."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import cv2
import numpy as np

from raster_parser.commons.helpers.geometry import compute_origin
from raster_parser.commons.helpers.text_orientation import rotate_to_direction
from raster_parser.models.text import Text

# PaddleOCR's PP-OCR recognisers take 48 px tall line images.
OCR_TARGET_HEIGHT_PX = 48
# Widest normalised crop handed to the recogniser; longer lines are squeezed to fit.
OCR_TARGET_WIDTH_PX = 1024
# Padding added around a crop before normalising: tight above/below so the glyphs fill the
# 48 px height, generous left/right so the first/last glyph is not clipped. Fractions are of
# the crop height, with a floor in pixels for tiny crops.
OCR_PAD_V_FRAC = 0.05
OCR_PAD_V_MIN_PX = 2
OCR_PAD_H_FRAC = 0.30
OCR_PAD_H_MIN_PX = 40


def normalise_crop_for_ocr(
    crop: np.ndarray,
    target_height: int = OCR_TARGET_HEIGHT_PX,
    target_width: int = OCR_TARGET_WIDTH_PX,
) -> np.ndarray:
    """Pad `crop` (tight vertically, generous horizontally) and scale it to PaddleOCR's
    recognition height, keeping the aspect ratio up to `target_width`."""
    if crop.size == 0:
        return crop

    h, w = crop.shape[:2]
    if h <= 0 or w <= 0:
        return crop

    # Very tight vertical padding so the letters fill the height, generous horizontal padding
    # so the edges don't clip.
    pad_h = max(OCR_PAD_V_MIN_PX, int(h * OCR_PAD_V_FRAC))
    pad_w = max(OCR_PAD_H_MIN_PX, int(h * OCR_PAD_H_FRAC))
    padded_crop = cv2.copyMakeBorder(
        crop,
        top=pad_h,
        bottom=pad_h,
        left=pad_w,
        right=pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255] if len(crop.shape) == 3 else 255,
    )

    hp, wp = padded_crop.shape[:2]
    scale = target_height / hp
    final_w = min(max(1, int(wp * scale)), target_width)
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(padded_crop, (final_w, target_height), interpolation=interpolation)


def ocr_text(
    word: str,
    bbox: tuple[float, float, float, float],
    rotate: int,
    *,
    page_index: int,
    block_no: int = 0,
    confidence: float = 0.0,
) -> Text:
    """The shared `Text` for one OCR word: direction from `rotate`, no font or colour, and the
    recogniser's `confidence` when it gave one."""
    bbox = tuple(float(c) for c in bbox)
    direction = rotate_to_direction(rotate)
    return Text(
        text=word,
        bbox=bbox,
        direction=direction,
        origin=compute_origin(bbox, direction),
        font="",
        font_size=0.0,
        color=None,
        flags=0,
        ascender=None,
        descender=None,
        wmode=0,
        block_no=block_no,
        line_no=0,
        word_no=0,
        page_index=page_index,
        seqno=0,
        confidence=float(confidence),
        source="ocr",
        orientation_source="ocr",
    )


def paddle_lines(raw) -> Iterator[tuple[Any, str, float]]:
    """`(box, text, confidence)` for every well-formed line of a PaddleOCR 2.x result
    (`[[[box, (text, conf)], ...], ...]`; pages may be `None`). No filtering beyond shape."""
    for page in raw or []:
        for line in page or []:
            if not line or len(line) < 2:
                continue
            box, rec = line[0], line[1]
            if not isinstance(rec, (list, tuple)) or len(rec) < 2:
                continue
            yield box, rec[0], rec[1]
