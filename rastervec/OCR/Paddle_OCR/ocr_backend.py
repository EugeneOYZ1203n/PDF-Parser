"""The OCR recognition backend behind `RenderOCR`.

Text *detection* is not PaddleOCR's job in this pipeline -- the Radon
segmentation step (`OCR/radon.py`) deskews each cluster render and splits
it into word crops. A backend only has to *recognise* those pre-segmented
crops, so the whole `OcrBackend` contract is one method:
`recognize_crops(crops) -> list[OcrBox]`, one `OcrBox` per input crop
(blank text allowed, in input order).

`PaddleRecBackend` is the only implementation: paddleocr 2.x's
recognition-only path. It builds a `paddleocr.PaddleOCR` engine
(`config.OCR_VERSION` = PP-OCRv4, `config.OCR_LANG`) and calls its batch
`text_recognizer` directly on the normalised crops -- the detector is
never invoked. The engine is built lazily and cached at class scope so a
spawn pool started next finds the weights on disk. This is the same API
surface `archive/`'s `raster_parser` OCR uses, so the `legacy` benchmark
variant needs no compatibility shim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rastervec.config import OCR_LANG, OCR_VERSION, REC_BATCH_SIZE
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


class PaddleRecBackend:
    """paddleocr 2.x recognition only. One engine per `(ocr_version, lang)`,
    cached at class scope -- every instance shares it."""

    _ENGINE_CACHE: dict[tuple[str, str], object] = {}

    def __init__(self, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        self.key = (ocr_version, lang)

    @classmethod
    def warmup(cls, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        """Force the (model-downloading on first ever call) engine build
        now, in the calling process. Safe to call repeatedly."""
        cls(ocr_version, lang)._engine()

    def _engine(self):
        if self.key not in PaddleRecBackend._ENGINE_CACHE:
            # torch must load before paddle on Windows: paddleocr 2.x pulls
            # paddle (and, via albumentations, torch) at import, and a
            # paddle-first process then fails torch's DLL load (shm.dll,
            # WinError 127 -- clashing OpenMP runtimes). FAST already relies on
            # this order; make the OCR path self-sufficient too.
            import torch  # noqa: F401
            from paddleocr import PaddleOCR

            ocr_version, lang = self.key
            PaddleRecBackend._ENGINE_CACHE[self.key] = PaddleOCR(
                ocr_version=ocr_version,
                lang=lang,
                use_angle_cls=False,  # the Radon step resolves the 180-degree flip
                show_log=False,
                rec_batch_num=REC_BATCH_SIZE,
            )
        return PaddleRecBackend._ENGINE_CACHE[self.key]

    def recognize_crops(self, crops: list[np.ndarray]) -> list[OcrBox]:
        """One `OcrBox` per crop, in input order. A crop that recognises to
        blank text still gets a box (empty text, 0.0 confidence) so callers
        can zip results back to `word_corners` positionally."""
        if not crops:
            return []
        # normalise -> RGB -> BGR (paddleocr 2.x's TextRecognizer is cv2/BGR).
        bgr = [
            np.ascontiguousarray(
                np.asarray(normalize_line_crop(_as_pil(c)).convert("RGB"))[:, :, ::-1]
            )
            for c in crops
        ]
        rec = self._engine().text_recognizer(bgr)
        rows = rec[0] if isinstance(rec, tuple) else rec
        out: list[OcrBox] = []
        for row in rows:
            text = str(row[0] or "").strip()
            score = float(row[1] or 0.0)
            out.append(OcrBox(text=text, confidence=score if text else 0.0, corners=[]))
        return out


def _as_pil(arr: np.ndarray):
    from PIL import Image

    if arr.ndim == 2:
        return Image.fromarray(arr.astype(np.uint8), mode="L")
    return Image.fromarray(arr.astype(np.uint8))


def _recognize_crops_job(
    crops: list[np.ndarray],
    ocr_version: str = OCR_VERSION,
    lang: str = OCR_LANG,
) -> list[OcrBox]:
    """Top-level, picklable Pool-2 job: recognise `crops` with a
    `PaddleRecBackend` cached per Pool-2 worker process by `(ocr_version,
    lang)` (`PaddleRecBackend._ENGINE_CACHE` is keyed the same way for local
    calls). Fitz-free -- plain numpy crops in, dataclasses out."""
    return PaddleRecBackend(ocr_version, lang).recognize_crops(crops)
