"""OCR results shared by the PaddleOCR engine wrappers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OcrBox:
    """One recognised crop: text, raw confidence, and `flip_deg` -- 180 when
    the angle classifier flipped the crop before recognition, else 0
    (always 0 from `recognize_crops_raw`)."""

    text: str
    confidence: float
    flip_deg: int = 0
