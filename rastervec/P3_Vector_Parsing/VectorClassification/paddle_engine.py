"""The OCR backend: PaddleOCR detection + recognition over seqno-clustered
word groups (`wordgrouping.py::cluster_by_seqno`).

`PaddleDetectBackend.detect` runs PaddleOCR's own text-DETECTION model
against a word group's own rendered+padded image, returning every text quad
it finds in that image's own pixel space; `PaddleRecBackend.recognize_crops`
then recognises each detected quad's own perspective-cropped region
(`_rotate_crop`). This mirrors `LegacyRecreation/paddle_engine.py`'s own
independent copy of the same detect-then-recognize pair (itself a port of
archive/raster_parser's `PaddleOcr.ocr_image`, which called PaddleOCR's full
`ocr()` pipeline per word group) -- own duplicated copy here too, per the
"sibling P3 backends share zero code" rule.

`PaddleRecBackend`/`PaddleDetectBackend` each build one `paddleocr.PaddleOCR`
engine (`config.OCR_VERSION` = PP-OCRv4, `config.OCR_LANG`), cached at class
scope by `(ocr_version, lang)` so a spawn pool started next finds the
weights on disk. This is the same API surface `archive/`'s `raster_parser`
OCR uses, so the `legacy` benchmark variant needs no compatibility shim.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from skimage.transform import ProjectiveTransform, warp

from rastervec.P3_Vector_Parsing.VectorClassification.config import (
    OCR_BATCH_SIZE,
    OCR_LANG,
    OCR_VERSION,
)

# PaddleOCR's own DB detector's det_limit_side_len -- a per-word-group render
# is rarely anywhere near this, so it's a generous ceiling rather than a
# tuned value (same constant/rationale as FastIntoPaddle's/LegacyRecreation's
# own copies).
_DETECT_LIMIT_SIDE_LEN = 4000

# _rotate_crop's own crop-shaping knobs: how much larger than the raw
# detected quad to crop (scaled about the quad's own centroid, so a tight
# detection still keeps full glyph strokes/ascenders/descenders), and the
# flat white border added around the result.
_CROP_EXPAND_FRACTION = 0.05
_CROP_BORDER_PX = 5


@dataclass
class OcrBox:
    """One recognised crop: text, confidence, and the classifier's own
    0/180 flip decision (degrees, 0 or 180)."""

    text: str
    confidence: float
    flip_deg: int = 0


class PaddleRecBackend:
    """paddleocr 2.x recognition + angle classification. One engine per
    `(ocr_version, lang)`, cached at class scope -- every instance shares
    it."""

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
                use_angle_cls=True,  # PaddleOCR's own classifier resolves the 0/180 flip
                show_log=False,
                rec_batch_num=OCR_BATCH_SIZE,
                cls_batch_num=OCR_BATCH_SIZE,
            )
        return PaddleRecBackend._ENGINE_CACHE[self.key]

    def recognize_crops(self, crops: list[np.ndarray]) -> list[OcrBox]:
        """One `OcrBox` per crop, in input order: classify each crop's
        orientation (0/180), rotate the ones flagged 180, then recognise
        the whole (now-upright) batch in a single `text_recognizer` pass --
        one recognition call total, not the old upright-and-flipped double
        call. Each crop is used as handed over, only converted to BGR --
        `text_recognizer` does its own resize."""
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


class PaddleDetectBackend:
    """PaddleOCR's own text-DETECTION model (`engine.text_detector`, PP-OCR's
    DB detector), run against an already-rendered+padded word-group image.
    Cached separately from `PaddleRecBackend` (its own `_ENGINE_CACHE`),
    built with a generous `det_limit_side_len`. A bare `detect` -- the
    caller (`parse.py`) already renders and pads the group's own image once
    and only needs the raw detected quads (padded-render pixel space) back,
    since it drives its own crop+recognize loop directly. Own duplicated
    copy of `LegacyRecreation/paddle_engine.py::PaddleDetectBackend`."""

    _ENGINE_CACHE: dict[tuple[str, str], object] = {}

    def __init__(self, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        self.key = (ocr_version, lang)

    @classmethod
    def warmup(cls, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        cls(ocr_version, lang)._engine()

    def _engine(self):
        if self.key not in PaddleDetectBackend._ENGINE_CACHE:
            import torch  # noqa: F401 -- must import before paddle on Windows
            from paddleocr import PaddleOCR

            ocr_version, lang = self.key
            PaddleDetectBackend._ENGINE_CACHE[self.key] = PaddleOCR(
                ocr_version=ocr_version, lang=lang, use_angle_cls=True, show_log=False,
                det_limit_side_len=_DETECT_LIMIT_SIDE_LEN,
            )
        return PaddleDetectBackend._ENGINE_CACHE[self.key]

    def detect(self, bgr: np.ndarray) -> list[np.ndarray]:
        """Every text quad PaddleOCR's detector finds in `bgr`, each a
        `(4, 2)` array of `(x, y)` pixel points in `bgr`'s own pixel space,
        PaddleOCR's own corner order (clockwise from top-left)."""
        dt_boxes, _elapse = self._engine().text_detector(bgr)
        return [np.asarray(quad, dtype=np.float64) for quad in dt_boxes]


def _normalize_rotation(angle_deg: float) -> float:
    """Wrap to `[-90, 90)` -- text direction is a line, not an arrow, so a
    0/180 ambiguity always remains mod 180 (resolved separately by the
    cls-flip term this is added to before calling this). Own duplicated copy
    of `FastIntoPaddle`/`LegacyRecreation`'s `paddle_engine.py::
    _normalize_rotation`."""
    return ((angle_deg + 90.0) % 180.0) - 90.0


def _quad_rotation_deg(quad: np.ndarray) -> float:
    """The quad's own dominant-edge orientation, as the rotation (degrees,
    counter-clockwise-positive) that would bring that edge to horizontal.
    `quad` is 4 `(x, y)` pixel points in PaddleOCR's own corner order
    (clockwise from top-left); the longer of the top edge (0->1) and left
    edge (0->3) is taken as the text's own baseline direction, so this is
    robust to a quad that's taller than it is wide (vertical/rotated text).
    Own duplicated copy of `FastIntoPaddle`/`LegacyRecreation`'s
    `paddle_engine.py::_quad_rotation_deg`."""
    top = quad[1] - quad[0]
    left = quad[3] - quad[0]
    dx, dy = (top if np.hypot(*top) >= np.hypot(*left) else left)
    theta = math.degrees(math.atan2(dy, dx))
    return _normalize_rotation(-theta)


def _rotate_crop(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-correct crop of `quad` (4 `(x, y)` pixel points,
    PaddleOCR's own corner order), expanded `_CROP_EXPAND_FRACTION` larger
    about its own centroid before cropping (so a tight detection still keeps
    full glyph strokes/ascenders/descenders rather than clipping them) and
    warped onto an axis-aligned rectangle sized to that expanded quad's own
    edge lengths -- the standard PaddleOCR "get_rotate_crop_image" step,
    reimplemented with `skimage` (no cv2, per this project's convention) --
    plus a flat `_CROP_BORDER_PX` white border added around the result.
    `warp`'s own `cval=255.0` already fills white for any part of the
    expanded quad that samples outside `bgr` (e.g. a detection near the
    render's own edge), so no separate bounds clamping is needed. Was
    previously an unexpanded, unbordered, byte-identical copy of
    `FastIntoPaddle`/`LegacyRecreation`'s `paddle_engine.py::_rotate_crop`;
    those two still crop tight with no border -- this is now
    VectorClassification's own divergent copy."""
    centroid = quad.mean(axis=0)
    quad = centroid + (quad - centroid) * (1.0 + _CROP_EXPAND_FRACTION)
    width = max(1, int(round(max(np.hypot(*(quad[1] - quad[0])), np.hypot(*(quad[2] - quad[3]))))))
    height = max(1, int(round(max(np.hypot(*(quad[3] - quad[0])), np.hypot(*(quad[2] - quad[1]))))))
    rect = np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float64)
    tf = ProjectiveTransform()
    tf.estimate(rect, quad)
    warped = warp(bgr, tf, output_shape=(height, width), cval=255.0, preserve_range=True)
    cropped = np.clip(warped, 0, 255).astype(np.uint8)
    return np.pad(
        cropped,
        ((_CROP_BORDER_PX, _CROP_BORDER_PX), (_CROP_BORDER_PX, _CROP_BORDER_PX), (0, 0)),
        mode="constant", constant_values=255,
    )


def _normalize_bgr(crop: np.ndarray) -> np.ndarray:
    """A crop -> a 3-channel BGR array (paddleocr 2.x's
    TextClassifier/TextRecognizer/TextDetector are cv2/BGR)."""
    arr = np.asarray(crop, dtype=np.uint8)
    if arr.ndim == 2:  # grayscale -- gray RGB and gray BGR are identical
        return np.ascontiguousarray(np.repeat(arr[:, :, None], 3, axis=2))
    return np.ascontiguousarray(arr[:, :, :3][:, :, ::-1])
