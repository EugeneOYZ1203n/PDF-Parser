"""Legacy 48px line-height crop normalization for PaddleOCR recognition.

PIL reimplementation of `archive/raster_parser/parsing/parser.py`'s
`normalise_crop_for_ocr`: asymmetric white padding (tight vertically so
glyphs stay tall, generous horizontally so edge glyphs don't clip) then a
resize to a fixed recognition line height (48px), aspect preserved, width
capped. No OpenCV -- `PIL.ImageOps.expand` + `Image.resize` only.
"""
from __future__ import annotations

from PIL import Image, ImageOps

from rastervec.config import (
    OCR_HORIZONTAL_PADDING_FRACTION,
    OCR_VERTICAL_PADDING_FRACTION,
    REC_LINE_HEIGHT_PX,
    REC_LINE_MAX_WIDTH_PX,
)


def normalize_line_crop(
    img: Image.Image,
    *,
    target_height: int = REC_LINE_HEIGHT_PX,
    target_width: int = REC_LINE_MAX_WIDTH_PX,
) -> Image.Image:
    """Pad `img` (a single word/line crop) with white -- `max(2, 5% of h)`
    top/bottom, `max(40, 30% of h)` left/right -- then resize to
    `target_height` px tall, width scaled to preserve aspect and clamped
    to `target_width`. A zero-area image is returned unchanged."""
    if img.width == 0 or img.height == 0:
        return img

    pad_h = max(2, int(img.height * OCR_VERTICAL_PADDING_FRACTION))
    pad_w = max(40, int(img.height * OCR_HORIZONTAL_PADDING_FRACTION))
    fill: int | tuple[int, int, int] = 255 if img.mode in ("L", "1", "I", "F") else (255, 255, 255)
    padded = ImageOps.expand(img, border=(pad_w, pad_h, pad_w, pad_h), fill=fill)

    scale = target_height / padded.height
    new_w = max(1, min(int(padded.width * scale), target_width))
    resample = Image.BICUBIC if scale > 1.0 else Image.LANCZOS
    return padded.resize((new_w, target_height), resample)
