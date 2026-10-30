"""The PaddleOCR 2.x engine every caller shares: `v2_vector_class` (cluster renders) and the raster
prelude `commons.raster_text` (embedded images).

`PaddleBackend.detect` returns text quads in the given image's pixel space;
`recognize_crops` classifies each crop 0/180 then recognises the batch; `recognize_crops_raw`
recognises crops as given (a caller's own quarter-turn retry sweep). One `PaddleOCR` per
`(ocr_version, lang)` per process (`_shared_engine`).
"""

from __future__ import annotations

import contextlib
import sys

import numpy as np
from raster_parser.models.ocr import OcrBox

OCR_VERSION = "PP-OCRv4"
OCR_LANG = "en"
OCR_BATCH_SIZE = 128

# PaddleOCR's own DB detector's det_limit_side_len -- renders are rarely anywhere near this, so
# it's a generous ceiling rather than a tuned value.
_DETECT_LIMIT_SIDE_LEN = 4000


def _rows_to_boxes(rows, flips) -> list[OcrBox]:
    out: list[OcrBox] = []
    for row, flip in zip(rows, flips):
        text = str(row[0] or "").strip()
        conf = float(row[1] or 0.0)
        out.append(OcrBox(text=text, confidence=conf if text else 0.0, flip_deg=flip))
    return out


_ENGINES: dict[tuple[str, str], object] = {}


def _shared_engine(ocr_version: str, lang: str):
    """The one `PaddleOCR` instance per `(ocr_version, lang)` in this process,
    used for detection and recognition alike, by every caller -- each `PaddleOCR`
    loads det + cls + rec, so separate instances held every model twice. The merged settings are the union of what each backend
    needed (`det_limit_side_len` only affects `text_detector`; the batch
    sizes only `text_classifier`/`text_recognizer`), so outputs are
    unchanged. Every pool worker is its own process with
    its own cache, and within a process detect/recognise run sequentially on one
    thread, so sharing is safe."""
    key = (ocr_version, lang)
    if key not in _ENGINES:
        if sys.platform == "win32":
            with contextlib.suppress(ImportError):
                import torch  # noqa: F401 -- must load before paddle on Windows (shm.dll)
        from paddleocr import PaddleOCR

        _ENGINES[key] = PaddleOCR(
            ocr_version=ocr_version,
            lang=lang,
            use_angle_cls=True,  # PaddleOCR's own classifier resolves the 0/180 flip
            show_log=False,
            det_limit_side_len=_DETECT_LIMIT_SIDE_LEN,
            rec_batch_num=OCR_BATCH_SIZE,
            cls_batch_num=OCR_BATCH_SIZE,
        )
    return _ENGINES[key]


class PaddleBackend:
    """paddleocr 2.x text detection, angle classification and recognition, all on the process's
    shared engine (`_shared_engine`, built with a generous `det_limit_side_len`)."""

    def __init__(self, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        self.key = (ocr_version, lang)

    @classmethod
    def warmup(cls, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        """Force the (model-downloading on first ever call) engine build
        now, in the calling process. Safe to call repeatedly."""
        cls(ocr_version, lang)._engine()

    def _engine(self):
        return _shared_engine(*self.key)

    def detect(self, bgr: np.ndarray) -> list[np.ndarray]:
        """Every text quad PaddleOCR's DB detector (`engine.text_detector`) finds in `bgr`, each
        a `(4, 2)` array of `(x, y)` pixel points in `bgr`'s own pixel space, PaddleOCR's own
        corner order (clockwise from top-left)."""
        dt_boxes, _elapse = self._engine().text_detector(bgr)
        if dt_boxes is None:
            return []
        return [np.asarray(quad, dtype=np.float64) for quad in dt_boxes]

    def recognize_crops(self, crops: list[np.ndarray]) -> list[OcrBox]:
        """One `OcrBox` per BGR crop, in input order: PaddleOCR's angle
        classifier labels each (already upright-rotated) crop 0 or 180, the
        180 ones are flipped, then the whole batch is recognised in one
        `text_recognizer` call. `flip_deg` records the flip."""
        if not crops:
            return []
        bgr = [_as_bgr(c) for c in crops]
        engine = self._engine()
        # TextClassifier deep-copies its input itself (predict_cls.py).
        _, cls_results, _ = engine.text_classifier(bgr)
        flips = [180 if str(label) == "180" else 0 for label, _score in cls_results]
        upright = [
            np.ascontiguousarray(np.rot90(img, 2)) if flip else img for img, flip in zip(bgr, flips)
        ]
        rec = engine.text_recognizer(upright)
        rows = rec[0] if isinstance(rec, tuple) else rec
        return _rows_to_boxes(rows, flips)

    def recognize_crops_raw(self, crops: list[np.ndarray]) -> list[OcrBox]:
        """Recognise BGR `crops` as-is, no `text_classifier` call --
        `parse.py`'s retry sweep, which has already rotated each crop to the
        quarter-turn it wants tried. `flip_deg` is always 0."""
        if not crops:
            return []
        bgr = [_as_bgr(c) for c in crops]
        rec = self._engine().text_recognizer(bgr)
        rows = rec[0] if isinstance(rec, tuple) else rec
        return _rows_to_boxes(rows, [0] * len(bgr))


def _as_bgr(crop: np.ndarray) -> np.ndarray:
    """An already-BGR crop -> a contiguous 3-channel uint8 array, copying
    only when it isn't one already (grayscale, extra channels, a strided
    view)."""
    arr = np.asarray(crop, dtype=np.uint8)
    if arr.ndim == 2:
        return np.ascontiguousarray(np.repeat(arr[:, :, None], 3, axis=2))
    return np.ascontiguousarray(arr[:, :, :3])


def rgb_to_bgr(crop: np.ndarray) -> np.ndarray:
    """An RGB image (e.g. a PIL render) -> a 3-channel BGR array (paddleocr
    2.x's TextClassifier/TextRecognizer/TextDetector are cv2/BGR)."""
    arr = np.asarray(crop, dtype=np.uint8)
    if arr.ndim == 2:  # grayscale -- gray RGB and gray BGR are identical
        return np.ascontiguousarray(np.repeat(arr[:, :, None], 3, axis=2))
    return np.ascontiguousarray(arr[:, :, :3][:, :, ::-1])
