"""Light OCR backend -- own ink-projection segmentation + PaddleOCR
recognition-only, an `OcrBackend` (see `ocr_backend.py`) drop-in for
`RenderOCR` that skips the full PP-OCRv6 detection + doc-unwarping +
textline-orientation pipeline the heavy `PaddleOcrBackend` runs per
cluster.

Per `detect(image)`:

1. Grayscale the render. Blank -> empty detection.
2. Rotation: ask PaddleOCR's standalone `DocImgOrientationClassification`
   (`PP-LCNet_x1_0_doc_ori` -- a light 0/90/180/270 image classifier) for
   the render's orientation. If it loads and is confident, rotate the
   render upright by that angle. Otherwise fall back to aspect-gated
   retry: a crop taller than `VERTICAL_ASPECT x` its width is tried at 90
   and 270 and the higher length-weighted-confidence reading wins (the
   archive's `recognize_boxes` CW/CCW approach). `180` is only reachable
   via the classifier, never via aspect-gating.
3. Split the upright render into line boxes (`split_lines_by_ink`) then
   per line into word boxes (`split_words_by_ink`).
4. Normalize each word crop to the legacy 48px recognition line height
   (`normalize_line_crop`) and run one batched `TextRecognition.predict`.
5. Map each word box's corners back to the *original* render's pixel
   space (`_unrotate_box`) so `RenderOCR.ocr_cluster` /
   `renderer.pixel_to_page_bbox` place them correctly. Boxes are
   word-level (`OcrBox.is_word=True`).

Engines (`TextRecognition`, `DocImgOrientationClassification`) load model
weights on first construction, so each is built lazily and cached at
class scope -- mirrors `PaddleOcrBackend._ENGINE_CACHE` / `warmup()`.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

# Must be set before paddlex reads its flags on the first lazy `paddleocr`
# import below -- same reason as ocr_backend.py (mkldnn CPU path hits a PIR
# error in this paddlepaddle build). Only set if the caller hasn't.
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")

from rastervec.config import DOC_ORI_MIN_CONFIDENCE, REC_BATCH_SIZE, VERTICAL_ASPECT
from rastervec.logging_setup import get_logger
from rastervec.OCR.Paddle_OCR.crop_normalize import normalize_line_crop
from rastervec.OCR.Paddle_OCR.ink_segment import (
    INK_LEVEL,
    split_lines_by_ink,
    split_words_by_ink,
)
from rastervec.OCR.Paddle_OCR.ocr_backend import OcrBox, OcrDetection

_LOG = get_logger("ocr.light")

# PP-OCRv6 rec model to use (small is ~2x lighter than the medium default
# TextRecognition ships; PP-OCRv6_tiny_rec is lighter still,
# PP-OCRv6_medium_rec is the accuracy fallback). Verified present in this
# paddleocr==3.7.0 install. Kept here, not in config.py: it's a model
# identifier, not a tuning knob.
LIGHT_REC_MODEL_NAME = "PP-OCRv6_small_rec"


def _rec_field(result: object, key: str):
    """Pull one field out of a paddlex predictor result (dict-like or
    attribute-style), tolerating either shape across versions."""
    try:
        return result[key]  # type: ignore[index]
    except (TypeError, KeyError, IndexError):
        return getattr(result, key, None)


class LightPaddleOcrBackend:
    """`OcrBackend` implementation: ink-projection word segmentation +
    PaddleOCR recognition-only. Word-level boxes (`OcrBox.is_word=True`),
    all corners already in the passed image's own pixel space."""

    _REC_ENGINE_CACHE: dict[tuple[str, str], object] = {}
    _ORI_ENGINE_CACHE: dict[str, object | None] = {}

    def __init__(self, lang: str = "en", model_name: str = LIGHT_REC_MODEL_NAME) -> None:
        self.lang = lang
        self.model_name = model_name

    @classmethod
    def warmup(cls, lang: str = "en", model_name: str = LIGHT_REC_MODEL_NAME) -> None:
        """Build + cache both engines now, in the calling process, so a
        spawn pool started next finds the weights on disk. Best effort."""
        backend = cls(lang, model_name)
        backend._rec_engine()
        backend._ori_engine()

    # -- engines -----------------------------------------------------------
    def _rec_engine(self):
        key = (self.lang, self.model_name)
        if key not in LightPaddleOcrBackend._REC_ENGINE_CACHE:
            from paddleocr import TextRecognition

            LightPaddleOcrBackend._REC_ENGINE_CACHE[key] = TextRecognition(
                model_name=self.model_name
            )
        return LightPaddleOcrBackend._REC_ENGINE_CACHE[key]

    def _ori_engine(self):
        """`DocImgOrientationClassification`, or `None` if it can't be
        built (missing weights / offline first run) -- callers then fall
        back to aspect gating."""
        key = self.lang
        if key not in LightPaddleOcrBackend._ORI_ENGINE_CACHE:
            try:
                from paddleocr import DocImgOrientationClassification

                LightPaddleOcrBackend._ORI_ENGINE_CACHE[key] = (
                    DocImgOrientationClassification()
                )
            except Exception as exc:  # noqa: BLE001 -- optional, degrade gracefully
                _LOG.warning(
                    "DocImgOrientationClassification unavailable (%s); "
                    "using aspect-gating for rotation", exc,
                )
                LightPaddleOcrBackend._ORI_ENGINE_CACHE[key] = None
        return LightPaddleOcrBackend._ORI_ENGINE_CACHE[key]

    # -- rotation --------------------------------------------------------
    def _detect_page_angle(self, image: "Image.Image") -> tuple[int | None, float]:
        engine = self._ori_engine()
        if engine is None:
            return None, 0.0
        try:
            results = engine.predict([np.asarray(image.convert("RGB"))])
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("doc-orientation predict failed (%s)", exc)
            return None, 0.0
        if not results:
            return None, 0.0
        r = results[0]
        names = _rec_field(r, "label_names") or _rec_field(r, "labels")
        scores = _rec_field(r, "scores")
        try:
            angle = int(str(names[0]).strip())
            score = float(scores[0]) if scores is not None else 1.0
        except (TypeError, ValueError, IndexError):
            return None, 0.0
        if angle not in (0, 90, 180, 270):
            return None, 0.0
        return angle, score

    @staticmethod
    def _candidate_angles(gray: np.ndarray) -> list[int]:
        h, w = gray.shape[:2]
        return [90, 270] if h > VERTICAL_ASPECT * max(1, w) else [0]

    # -- rotation geometry ---------------------------------------------
    @staticmethod
    def _rotate_gray(gray: np.ndarray, angle: int) -> np.ndarray:
        """Rotate `gray` counter-clockwise by `angle` (0/90/180/270)."""
        return np.rot90(gray, k=(angle // 90) % 4)

    @staticmethod
    def _unrotate_point(
        xp: float, yp: float, k: int, orig_w: int, orig_h: int,
    ) -> tuple[float, float]:
        """Map a point from `np.rot90(.., k)` space back to the original
        `(orig_w, orig_h)` image's pixel space."""
        k %= 4
        if k == 0:
            return xp, yp
        if k == 1:            # forward: x' = y, y' = orig_w-1 - x
            return orig_w - 1 - yp, xp
        if k == 2:
            return orig_w - 1 - xp, orig_h - 1 - yp
        return yp, orig_h - 1 - xp  # k == 3

    @classmethod
    def _unrotate_box(
        cls, box: tuple[int, int, int, int], angle: int, rot_w: int, rot_h: int,
    ) -> list[tuple[float, float]]:
        """The 4 corners of an axis-aligned box in rotated space
        (`rot_w` x `rot_h`), mapped back to the original render's pixel
        space. Corner order need not stay axis-aligned -- downstream takes
        min/max."""
        k = (angle // 90) % 4
        orig_w, orig_h = (rot_h, rot_w) if k in (1, 3) else (rot_w, rot_h)
        x0, y0, x1, y1 = box
        return [
            cls._unrotate_point(px, py, k, orig_w, orig_h)
            for px, py in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        ]

    @staticmethod
    def _len_weighted_mean_conf(boxes: list[OcrBox]) -> float:
        total_len = sum(len(b.text) for b in boxes)
        if total_len == 0:
            return float(np.mean([b.confidence for b in boxes])) if boxes else 0.0
        return sum(len(b.text) * b.confidence for b in boxes) / total_len

    # -- segmentation + recognition ----------------------------------
    @staticmethod
    def _word_boxes(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Line then word ink-projection split; word boxes translated back
        to `gray`'s own pixel coordinates."""
        boxes: list[tuple[int, int, int, int]] = []
        for lx0, ly0, lx1, ly1 in split_lines_by_ink(gray):
            line = gray[ly0:ly1, lx0:lx1]
            for wx0, wy0, wx1, wy1 in split_words_by_ink(line):
                boxes.append((lx0 + wx0, ly0 + wy0, lx0 + wx1, ly0 + wy1))
        return boxes

    def _recognize_at(self, gray: np.ndarray, angle: int) -> list[OcrBox]:
        rot = self._rotate_gray(gray, angle)
        boxes_px = self._word_boxes(rot)
        if not boxes_px:
            return []

        crops = [
            np.asarray(
                normalize_line_crop(Image.fromarray(rot[y0:y1, x0:x1])).convert("RGB")
            )
            for x0, y0, x1, y1 in boxes_px
        ]
        results = self._rec_engine().predict(crops, batch_size=REC_BATCH_SIZE)

        rot_h, rot_w = rot.shape[:2]
        out: list[OcrBox] = []
        for (x0, y0, x1, y1), r in zip(boxes_px, results):
            text = str(_rec_field(r, "rec_text") or "").strip()
            if not text:
                continue
            score = float(_rec_field(r, "rec_score") or 0.0)
            corners = self._unrotate_box((x0, y0, x1, y1), angle, rot_w, rot_h)
            out.append(OcrBox(text=text, confidence=score, corners=corners, is_word=True))
        return out

    def detect(self, image: "Image.Image") -> OcrDetection:
        gray = np.asarray(image.convert("L"))
        if gray.size == 0 or not (gray < INK_LEVEL).any():
            return OcrDetection(boxes=[], rotation=0)

        angle, conf = self._detect_page_angle(image)
        if angle is not None and conf >= DOC_ORI_MIN_CONFIDENCE:
            boxes = self._recognize_at(gray, angle)
            if boxes:
                return OcrDetection(boxes=boxes, rotation=angle)

        best: tuple[float, int, list[OcrBox]] | None = None
        for candidate in self._candidate_angles(gray):
            boxes = self._recognize_at(gray, candidate)
            if not boxes:
                continue
            score = self._len_weighted_mean_conf(boxes)
            if best is None or score > best[0]:
                best = (score, candidate, boxes)

        if best is None:
            return OcrDetection(boxes=[], rotation=0)
        return OcrDetection(boxes=best[2], rotation=best[1])
