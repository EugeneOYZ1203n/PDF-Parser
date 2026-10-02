"""The OCR backend for CollinearVectorClass: PaddleOCR detection +
recognition, plus the image-rotation helpers `parse.py` uses.

`PaddleDetectBackend.detect` runs PaddleOCR's text detector over a cluster
render that `parse.py` has already rotated (`rotate_image`) by the
cluster's dominant parallel-group direction; detected quads are mapped back
to the unrotated render with `unrotate_points`. Each quad's crop
(`axis_aligned_crop`, out of the rotated render) is rotated by whatever
extra correction `rotation.py` decides, then `PaddleRecBackend.
recognize_crops_raw` recognises it with no angle classifier; a crop whose
`score` is low is retried at +90/180/270 and the best-scoring read wins. `hough_angle_deg` is the only
raster angle estimate left -- a plain Hough line angle mod 180 (no
minAreaRect agreement, no 10-degree grid), snapped to a vector-derived
angle by `rotation.py`.

Engines are cached at class scope by `(ocr_version, lang)`. Own duplicated
copy, per the "sibling P3 backends share zero code" rule. Imports `cv2`
(affine rotation) and `skimage` (Hough, dilation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.morphology import binary_dilation, disk
from skimage.transform import hough_line, hough_line_peaks

from rastervec.P3_Vector_Parsing.CollinearVectorClass.config import (
    HOUGH_DILATE_RADIUS_PX,
    HOUGH_INK_THRESHOLD,
    OCR_BATCH_SIZE,
    OCR_LANG,
    OCR_VERSION,
    SINGLE_CHAR_PENALTY,
)

# PaddleOCR's own DB detector's det_limit_side_len -- a per-word-group render
# is rarely anywhere near this, so it's a generous ceiling rather than a
# tuned value (same constant/rationale as LegacyRecreation's own copy).
_DETECT_LIMIT_SIDE_LEN = 4000

# axis_aligned_crop's crop-shaping knobs: how much larger than the raw
# detected quad to crop (scaled about the quad's own centroid, so a tight
# detection still keeps full glyph strokes/ascenders/descenders), and the
# flat white border added around the result.
_CROP_EXPAND_FRACTION = 0.05
_CROP_BORDER_PX = 5


@dataclass
class OcrBox:
    """One recognised crop: text, raw confidence, and `flip_deg` (always 0
    here -- there is no angle classifier; kept for shape parity)."""

    text: str
    confidence: float
    flip_deg: int = 0


def score(box: OcrBox) -> float:
    """Retry/selection score: 0 for a blank read, `confidence *
    SINGLE_CHAR_PENALTY` for a single-character read, else `confidence`."""
    if not box.text:
        return 0.0
    if len(box.text) == 1:
        return box.confidence * SINGLE_CHAR_PENALTY
    return box.confidence


class PaddleRecBackend:
    """paddleocr 2.x recognition (no angle classifier). One engine per
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
                use_angle_cls=False,  # no 0/180 classifier -- parse.py's retry sweep covers flips
                show_log=False,
                rec_batch_num=OCR_BATCH_SIZE,
            )
        return PaddleRecBackend._ENGINE_CACHE[self.key]

    def recognize_crops_raw(self, crops: list[np.ndarray]) -> list[OcrBox]:
        """Recognise `crops` as-is, one `OcrBox` per crop in input order --
        no `text_classifier` call. `parse.py` uses it for pass 1 and for
        every +90/180/270 retry, tracking the rotation it applied itself;
        `flip_deg` is always 0."""
        if not crops:
            return []
        bgr = [_normalize_bgr(c) for c in crops]
        rec = self._engine().text_recognizer(bgr)
        rows = rec[0] if isinstance(rec, tuple) else rec
        out: list[OcrBox] = []
        for row in rows:
            text = str(row[0] or "").strip()
            score = float(row[1] or 0.0)
            out.append(OcrBox(text=text, confidence=score if text else 0.0, flip_deg=0))
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


# ---------------------------------------------------------------------------
# Top-level, picklable Pool-2 jobs -- fitz-free, plain numpy/dataclasses in
# and out, each building/caching its own engine per Pool-2 worker process via
# the classes' own `_ENGINE_CACHE` (keyed by `(ocr_version, lang)`, same as a
# local call). `parse.py` dispatches these via `compute.starmap`/`compute.apply` when a caller passes
# a Pool-2 `compute` proxy, and calls the backend directly otherwise.
# ---------------------------------------------------------------------------
def _detect_job(
    bgr: np.ndarray, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG,
) -> list[np.ndarray]:
    """One cluster's `PaddleDetectBackend.detect` call as a Pool-2 job --
    every cluster's render is independent, so a page's clusters fan out
    across Pool-2 workers instead of running one at a time in the calling
    process."""
    return PaddleDetectBackend(ocr_version, lang).detect(bgr)


def _recognize_crops_raw_job(
    crops: list[np.ndarray], ocr_version: str = OCR_VERSION, lang: str = OCR_LANG,
) -> list[OcrBox]:
    """One page-wide `recognize_crops_raw` batch as a Pool-2 job (pass 1
    and every retry rotation in `parse.py`)."""
    return PaddleRecBackend(ocr_version, lang).recognize_crops_raw(crops)


def normalize_rotation(angle_deg: float) -> float:
    """Wrap to `[-90, 90)` -- a text direction from a line angle is axial
    (0/180 ambiguous); the classifier flip / retry sweep resolves that."""
    return ((angle_deg + 90.0) % 180.0) - 90.0


def axis_aligned_crop(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Axis-aligned bbox crop of `quad`'s region straight out of `bgr`,
    expanded `_CROP_EXPAND_FRACTION` about the quad's centroid (keeps full
    glyph strokes), clamped to `bgr`'s bounds, plus a flat `_CROP_BORDER_PX`
    white border."""
    centroid = quad.mean(axis=0)
    expanded = centroid + (quad - centroid) * (1.0 + _CROP_EXPAND_FRACTION)
    h, w = bgr.shape[:2]
    x0 = max(0, int(math.floor(expanded[:, 0].min())))
    y0 = max(0, int(math.floor(expanded[:, 1].min())))
    x1 = min(w, int(math.ceil(expanded[:, 0].max())))
    y1 = min(h, int(math.ceil(expanded[:, 1].max())))
    if x1 <= x0 or y1 <= y0:
        cropped = np.full((1, 1, 3), 255, dtype=np.uint8)
    else:
        cropped = np.asarray(bgr[y0:y1, x0:x1], dtype=np.uint8)
    return np.pad(
        cropped,
        ((_CROP_BORDER_PX, _CROP_BORDER_PX), (_CROP_BORDER_PX, _CROP_BORDER_PX), (0, 0)),
        mode="constant", constant_values=255,
    )


def rotate_image(bgr: np.ndarray, angle_deg: float) -> "tuple[np.ndarray, np.ndarray]":
    """`bgr` rotated by `angle_deg` -- visually counter-clockwise, so a
    line whose page-space (y-down) direction angle is `angle_deg` comes out
    horizontal -- on a canvas expanded to fit, background white. Returns
    `(rotated, M)` with `M` the 2x3 affine mapping original pixel coords to
    rotated ones (invert with `cv2.invertAffineTransform`). An angle of 0
    returns `bgr` itself and the identity."""
    if angle_deg % 360.0 == 0.0:
        return bgr, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    h, w = bgr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w = int(math.ceil(h * sin + w * cos))
    new_h = int(math.ceil(h * cos + w * sin))
    m[0, 2] += new_w / 2.0 - w / 2.0
    m[1, 2] += new_h / 2.0 - h / 2.0
    rotated = cv2.warpAffine(
        np.ascontiguousarray(bgr), m, (new_w, new_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
    return rotated, m


def unrotate_points(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Map `(n, 2)` pixel points of a `rotate_image` output back into the
    original image's pixel space."""
    inv = cv2.invertAffineTransform(m)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return pts @ inv[:, :2].T + inv[:, 2]


def _hough_ink_mask(crop: np.ndarray) -> np.ndarray:
    """Binary ink mask of a BGR crop (gray < `HOUGH_INK_THRESHOLD`),
    thickened by `binary_dilation` so thin/broken strokes form continuous
    lines for Hough."""
    gray = rgb2gray(np.asarray(crop, dtype=np.uint8)[:, :, ::-1])
    ink = gray < (HOUGH_INK_THRESHOLD / 255.0)
    return binary_dilation(ink, footprint=disk(HOUGH_DILATE_RADIUS_PX))


def hough_angle_deg(crop: np.ndarray) -> "tuple[float | None, np.ndarray]":
    """The dominant line's direction angle in `crop` (degrees, image y-down,
    folded to [0, 180) -- mod 180, not mod 90, so horizontal and vertical
    stay distinct while up/down are the same) from a single-peak Hough line
    transform over the dilated ink mask, plus that mask (for debug images).
    `None` when the mask is empty or no peak is found."""
    mask = _hough_ink_mask(crop)
    if not mask.any():
        return None, mask
    thetas = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)
    _acc, angles, _dists = hough_line_peaks(*hough_line(mask, theta=thetas), num_peaks=1)
    if len(angles) == 0:
        return None, mask
    # skimage's theta is the line's NORMAL angle (x cos t + y sin t = r);
    # the line itself runs at t + 90.
    return (math.degrees(angles[0]) + 90.0) % 180.0, mask


def _normalize_bgr(crop: np.ndarray) -> np.ndarray:
    """A crop -> a 3-channel BGR array (paddleocr 2.x's
    TextClassifier/TextRecognizer/TextDetector are cv2/BGR)."""
    arr = np.asarray(crop, dtype=np.uint8)
    if arr.ndim == 2:  # grayscale -- gray RGB and gray BGR are identical
        return np.ascontiguousarray(np.repeat(arr[:, :, None], 3, axis=2))
    return np.ascontiguousarray(arr[:, :, :3][:, :, ::-1])
