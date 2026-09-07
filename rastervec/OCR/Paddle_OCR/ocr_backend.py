"""The OCR recognition backend behind `RenderOCR`.

Text *detection* is not PaddleOCR's job in this pipeline -- the Radon
segmentation step (`pipelines/sub_pipelines/radon.py`) deskews each cluster
render and splits it into word crops. A backend only has to *recognise*
those pre-segmented crops, so the whole `OcrBackend` contract is one
method: `recognize_crops(crops) -> list[OcrBox]`, one `OcrBox` per input
crop (blank text allowed, in input order).

`PaddleRecBackend` is the only implementation: PaddleOCR's standalone
`TextRecognition` predictor (`config.OCR_REC_MODEL`), engine built lazily
and cached at class scope so a spawn pool started next finds the weights on
disk.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np

# Must be set before paddlex reads its flags on the first lazy `paddleocr`
# import below -- on this dev environment the default mkldnn CPU inference
# path hits an unimplemented PIR attribute-conversion error; plain "paddle"
# run mode is fine for small pre-cropped word renders. Only set if the
# caller hasn't already.
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")

from rastervec.config import OCR_REC_MODEL, REC_BATCH_SIZE
from rastervec.logging_setup import get_logger
from rastervec.OCR.Paddle_OCR.crop_normalize import normalize_line_crop

_LOG = get_logger("ocr.backend")


@dataclass
class OcrBox:
    """One recognised word. `corners` is filled in by `RenderOCR` from the
    Radon segmentation (the backend itself never sees page geometry), so a
    freshly recognised box carries an empty `corners`."""

    text: str
    confidence: float
    corners: list[tuple[float, float]]
    is_word: bool = True


class OcrBackend(Protocol):
    def recognize_crops(self, crops: list[np.ndarray]) -> list[OcrBox]: ...


def _rec_field(result: object, key: str):
    """Pull one field out of a paddlex predictor result (dict-like or
    attribute-style), tolerating either shape across versions."""
    try:
        return result[key]  # type: ignore[index]
    except (TypeError, KeyError, IndexError):
        return getattr(result, key, None)


class PaddleRecBackend:
    """PaddleOCR `TextRecognition` (recognition only). One engine per
    `model_name`, cached at class scope -- every instance shares it."""

    _ENGINE_CACHE: dict[str, object] = {}

    def __init__(self, model_name: str = OCR_REC_MODEL) -> None:
        self.model_name = model_name

    @classmethod
    def warmup(cls, model_name: str = OCR_REC_MODEL) -> None:
        """Force the (model-downloading on first ever call) engine build
        now, in the calling process. Safe to call repeatedly."""
        cls(model_name)._engine()

    def _engine(self):
        if self.model_name not in PaddleRecBackend._ENGINE_CACHE:
            from paddleocr import TextRecognition

            PaddleRecBackend._ENGINE_CACHE[self.model_name] = TextRecognition(
                model_name=self.model_name
            )
        return PaddleRecBackend._ENGINE_CACHE[self.model_name]

    def recognize_crops(self, crops: list[np.ndarray]) -> list[OcrBox]:
        """One `OcrBox` per crop, in input order. A crop that recognises to
        blank text still gets a box (empty text, 0.0 confidence) so callers
        can zip results back to `word_corners` positionally."""
        if not crops:
            return []
        norm = [
            np.asarray(
                normalize_line_crop(_as_pil(c)).convert("RGB")
            )
            for c in crops
        ]
        results = self._engine().predict(norm, batch_size=REC_BATCH_SIZE)
        out: list[OcrBox] = []
        for r in results:
            text = str(_rec_field(r, "rec_text") or "").strip()
            score = float(_rec_field(r, "rec_score") or 0.0)
            out.append(OcrBox(text=text, confidence=score if text else 0.0, corners=[]))
        return out


def _as_pil(arr: np.ndarray):
    from PIL import Image

    if arr.ndim == 2:
        return Image.fromarray(arr.astype(np.uint8), mode="L")
    return Image.fromarray(arr.astype(np.uint8))
