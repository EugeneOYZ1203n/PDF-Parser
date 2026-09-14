"""Own, minimal PaddleOCR recognition engine wrapper -- same API surface as
the other two P3 variants' own paddle_engine.py copies (`PaddleRecBackend.
recognize_crops`), but not shared code: this is LegacyRecreation's own
independent copy, trimmed to just recognition (no detection backend, no
Segment-specific batching -- this variant OCRs one crop per word group
directly)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rastervec.P3_Vector_Parsing.LegacyRecreation.config import OCR_LANG, OCR_VERSION


@dataclass
class OcrBox:
    text: str
    confidence: float
    flip_deg: int = 0


class PaddleRecBackend:
    _ENGINE_CACHE: dict[tuple[str, str], object] = {}

    def __init__(self, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        self.key = (ocr_version, lang)

    @classmethod
    def warmup(cls, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        cls(ocr_version, lang)._engine()

    def _engine(self):
        if self.key not in PaddleRecBackend._ENGINE_CACHE:
            import torch  # noqa: F401 -- must import before paddle on Windows
            from paddleocr import PaddleOCR

            ocr_version, lang = self.key
            PaddleRecBackend._ENGINE_CACHE[self.key] = PaddleOCR(
                ocr_version=ocr_version, lang=lang, use_angle_cls=True, show_log=False,
            )
        return PaddleRecBackend._ENGINE_CACHE[self.key]

    def recognize_crops(self, crops: list[np.ndarray]) -> list[OcrBox]:
        if not crops:
            return []
        bgr = [_normalize_bgr(c) for c in crops]
        engine = self._engine()
        _, cls_results, _ = engine.text_classifier(bgr)
        flips = [180 if str(label) == "180" else 0 for label, _score in cls_results]
        upright = [np.rot90(img, 2) if flip else img for img, flip in zip(bgr, flips)]
        rec = engine.text_recognizer(upright)
        rows = rec[0] if isinstance(rec, tuple) else rec
        out: list[OcrBox] = []
        for row, flip in zip(rows, flips):
            text = str(row[0] or "").strip()
            score = float(row[1] or 0.0)
            out.append(OcrBox(text=text, confidence=score if text else 0.0, flip_deg=flip))
        return out


def _normalize_bgr(crop: np.ndarray) -> np.ndarray:
    arr = np.asarray(crop, dtype=np.uint8)
    if arr.ndim == 2:
        return np.ascontiguousarray(np.repeat(arr[:, :, None], 3, axis=2))
    return np.ascontiguousarray(arr[:, :, :3][:, :, ::-1])
