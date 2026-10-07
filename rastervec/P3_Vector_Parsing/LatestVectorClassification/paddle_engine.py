"""The OCR backend for LatestVectorClassification: PaddleOCR detection,
quad-driven upright crops, the 0/180 angle classifier and recognition.

Rotation comes from the PaddleOCR detect quad alone -- there is no Hough
line, minAreaRect or vector-direction estimate:

1. `PaddleDetectBackend.detect` runs on the *unrotated* cluster render and
   returns PaddleOCR's own (rotated, minAreaRect-shaped) quads.
2. `quad_long_edge_angle` takes the direction of the quad's longer edge;
   `upright_crop` **rotates** the image by that angle about the quad's
   centre (`cv2.warpAffine`, a pure rotation -- never a re-boxed
   axis-aligned bbox and never a perspective warp), warping straight into
   the now-axis-aligned quad's own crop size.
3. `PaddleRecBackend.recognize_crops` runs PaddleOCR's own angle
   classifier (0/180 flip) then recognises; `recognize_crops_raw` (no
   classifier) is `parse.py`'s +90/180/270 low-score retry. Both take BGR
   crops.

One `PaddleOCR` engine per `(ocr_version, lang)` per process, shared by
both backends (`_shared_engine`). Own duplicated copy, per the "sibling P3
backends share zero code" rule. Imports `cv2`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from rastervec.P3_Vector_Parsing.LatestVectorClassification.config import (
    CROP_BORDER_PX,
    CROP_EXPAND_FRACTION,
    OCR_BATCH_SIZE,
    OCR_LANG,
    OCR_VERSION,
    SINGLE_CHAR_PENALTY,
)

# PaddleOCR's own DB detector's det_limit_side_len -- a cluster render is
# rarely anywhere near this, so it's a generous ceiling rather than a tuned
# value (same constant/rationale as LegacyRecreation's own copy).
_DETECT_LIMIT_SIDE_LEN = 4000


@dataclass
class OcrBox:
    """One recognised crop: text, raw confidence, and `flip_deg` -- 180 when
    the angle classifier flipped the crop before recognition, else 0
    (always 0 from `recognize_crops_raw`)."""

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
    shared by `PaddleDetectBackend` and `PaddleRecBackend` -- each `PaddleOCR`
    loads det + cls + rec, so a separate instance per backend held every
    model twice. The merged settings are the union of what each backend
    needed (`det_limit_side_len` only affects `text_detector`; the batch
    sizes only `text_classifier`/`text_recognizer`), so outputs are
    unchanged. Every Pool-1/Pool-2 worker is its own process with its own
    cache, and within a process detect/recognise run sequentially on one
    thread, so sharing is safe."""
    key = (ocr_version, lang)
    if key not in _ENGINES:
        # torch must load before paddle on Windows: paddleocr 2.x pulls
        # paddle (and, via albumentations, torch) at import, and a
        # paddle-first process then fails torch's DLL load (shm.dll,
        # WinError 127 -- clashing OpenMP runtimes).
        import torch  # noqa: F401
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


class PaddleRecBackend:
    """paddleocr 2.x angle classification + recognition, on the process's
    shared engine (`_shared_engine`)."""

    def __init__(self, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        self.key = (ocr_version, lang)

    @classmethod
    def warmup(cls, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        """Force the (model-downloading on first ever call) engine build
        now, in the calling process. Safe to call repeatedly."""
        cls(ocr_version, lang)._engine()

    def _engine(self):
        return _shared_engine(*self.key)

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
        upright = [np.ascontiguousarray(np.rot90(img, 2)) if flip else img for img, flip in zip(bgr, flips)]
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


class PaddleDetectBackend:
    """PaddleOCR's own text-DETECTION model (`engine.text_detector`, PP-OCR's
    DB detector), run against an already-rendered cluster image, on the
    process's shared engine (`_shared_engine`, built with a generous
    `det_limit_side_len`)."""

    def __init__(self, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        self.key = (ocr_version, lang)

    @classmethod
    def warmup(cls, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        cls(ocr_version, lang)._engine()

    def _engine(self):
        return _shared_engine(*self.key)

    def detect(self, bgr: np.ndarray) -> list[np.ndarray]:
        """Every text quad PaddleOCR's detector finds in `bgr`, each a
        `(4, 2)` array of `(x, y)` pixel points in `bgr`'s own pixel space,
        PaddleOCR's own corner order (clockwise from top-left)."""
        dt_boxes, _elapse = self._engine().text_detector(bgr)
        return [np.asarray(quad, dtype=np.float64) for quad in dt_boxes]


# ---------------------------------------------------------------------------
# Top-level, picklable Pool-2 jobs -- fitz-free, plain numpy/dataclasses in
# and out, each building/caching its own engine per Pool-2 worker process via
# the classes' own `_ENGINE_CACHE`. `parse.py` dispatches these via
# `compute.starmap`/`compute.apply` when a caller passes a Pool-2 `compute`
# proxy, and calls the backend directly otherwise.
# ---------------------------------------------------------------------------
def _detect_job(
    bgr: np.ndarray, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG,
) -> list[np.ndarray]:
    """One cluster's `PaddleDetectBackend.detect` call as a Pool-2 job."""
    return PaddleDetectBackend(ocr_version, lang).detect(bgr)


def _recognize_crops_job(
    crops: list[np.ndarray], ocr_version: str = OCR_VERSION, lang: str = OCR_LANG,
) -> list[OcrBox]:
    """One page-wide classify+recognise batch (pass 1) as a Pool-2 job."""
    return PaddleRecBackend(ocr_version, lang).recognize_crops(crops)


def _recognize_crops_raw_job(
    crops: list[np.ndarray], ocr_version: str = OCR_VERSION, lang: str = OCR_LANG,
) -> list[OcrBox]:
    """One page-wide `recognize_crops_raw` batch (a retry rotation) as a
    Pool-2 job."""
    return PaddleRecBackend(ocr_version, lang).recognize_crops_raw(crops)


# ---------------------------------------------------------------------------
# Quad geometry + upright crop
# ---------------------------------------------------------------------------
def normalize_rotation(angle_deg: float) -> float:
    """Wrap to `[-90, 90)` -- a line direction is axial (0/180 ambiguous);
    the angle classifier / retry sweep resolves that."""
    return ((angle_deg + 90.0) % 180.0) - 90.0


def quad_long_edge_angle(quad) -> float:
    """Direction (deg, y-down, `[-90, 90)`) of the quad's longer side --
    edge p0->p1 vs p1->p2 (a detect quad is a rotated rectangle, so these
    are its two side directions)."""
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    e1, e2 = q[1] - q[0], q[2] - q[1]
    edge = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
    return normalize_rotation(math.degrees(math.atan2(edge[1], edge[0])))


def quad_size(quad) -> tuple[float, float]:
    """`(long, short)` side lengths of a (rotated-rectangle) quad -- the
    mean of each pair of opposite sides."""
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    a = (np.hypot(*(q[1] - q[0])) + np.hypot(*(q[3] - q[2]))) / 2.0
    b = (np.hypot(*(q[2] - q[1])) + np.hypot(*(q[0] - q[3]))) / 2.0
    return (max(a, b), min(a, b))


def quad_region(
    bgr: np.ndarray, quad, *, expand: float = CROP_EXPAND_FRACTION, border: int = CROP_BORDER_PX,
) -> "tuple[np.ndarray, tuple[float, float]]":
    """`(region, centre)`: the square, axis-aligned region of `bgr` centred
    on the quad's centre, big enough to hold the whole (expanded) quad at
    any rotation, white wherever it falls outside `bgr` -- `upright_crop`'s
    rotation frame (`upright_crop` itself no longer allocates it; kept for
    debug images) -- plus the quad's centre in `region`'s pixel space."""
    half, cx, cy, rc = _region_geometry(quad, expand, border)
    h, w = bgr.shape[:2]
    region = np.full((2 * half + 1, 2 * half + 1, 3), 255, dtype=np.uint8)
    x0, y0 = cx - half, cy - half
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, cx + half + 1), min(h, cy + half + 1)
    if sx1 > sx0 and sy1 > sy0:
        region[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = np.asarray(bgr[sy0:sy1, sx0:sx1], dtype=np.uint8)[:, :, :3]
    return region, rc


def _region_geometry(
    quad, expand: float, border: int,
) -> "tuple[int, int, int, tuple[float, float]]":
    """`quad_region`'s geometry without allocating the region:
    `(half, cx, cy, rc)` -- the region spans `[cx - half, cx + half]` x
    `[cy - half, cy + half]` of the source image (side `2 * half + 1`) and
    `rc` is the quad's centre in region pixel space."""
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    centre = q.mean(axis=0)
    half = int(math.ceil(max(np.hypot(*(q - centre).T)) * (1.0 + expand))) + border + 2
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    return half, cx, cy, (half + (centre[0] - cx), half + (centre[1] - cy))


def upright_crop(
    bgr: np.ndarray, quad, angle_deg: float,
    *, expand: float = CROP_EXPAND_FRACTION, border: int = CROP_BORDER_PX,
    size: "tuple[float, float] | None" = None,
) -> np.ndarray:
    """The quad's text, rotated upright: the image is rotated by `angle_deg`
    about the quad's centre -- visually counter-clockwise, so a quad whose
    long edge runs at `angle_deg` (y-down) comes out horizontal -- and the
    quad's own `long x short` rectangle (expanded by `expand`) is cropped
    from its centre, plus a flat `border` px white border.

    Pixel-for-pixel what rotating `quad_region`'s square region and cropping
    it would give (same window arithmetic, same sampling positions, white
    outside the image), but done as one `cv2.warpAffine` straight from `bgr`
    into the crop's own size -- the square region (side ~2x the quad's
    diagonal) is never allocated, so a long text line no longer costs two
    diagonal^2 buffers.

    `size` = `(along, across)` overrides `quad_size`'s `(long, short)` -- for
    a piece of a split quad, which can be shorter along the reading axis
    than across it."""
    long_side, short_side = size if size is not None else quad_size(quad)
    half, cx, cy, rc = _region_geometry(quad, expand, border)
    side = 2 * half + 1
    cw = long_side * (1.0 + expand)
    ch = short_side * (1.0 + expand)
    rx0 = max(0, int(math.floor(rc[0] - cw / 2.0)))
    ry0 = max(0, int(math.floor(rc[1] - ch / 2.0)))
    rx1 = min(side, int(math.ceil(rc[0] + cw / 2.0)))
    ry1 = min(side, int(math.ceil(rc[1] + ch / 2.0)))
    if rx1 <= rx0 or ry1 <= ry0:
        cropped = np.full((1, 1, 3), 255, dtype=np.uint8)
    else:
        # Region pixel (x, y) is source pixel (x + ox, y + oy); the crop
        # window's (0, 0) is region pixel (rx0, ry0).
        ox, oy = cx - half, cy - half
        src = np.asarray(bgr, dtype=np.uint8)[:, :, :3]
        if angle_deg % 360.0 != 0.0:
            m = cv2.getRotationMatrix2D(rc, angle_deg, 1.0)
            # Region-space rotation, conjugated into source -> crop space:
            # crop = M (src - o) - (rx0, ry0).
            m[:, 2] -= m[:, :2] @ np.array([ox, oy], dtype=np.float64) + np.array([rx0, ry0], dtype=np.float64)
            cropped = cv2.warpAffine(
                np.ascontiguousarray(src), m, (rx1 - rx0, ry1 - ry0),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
            )
        else:
            cropped = _white_window(src, ox + rx0, oy + ry0, rx1 - rx0, ry1 - ry0)
    return np.pad(
        cropped, ((border, border), (border, border), (0, 0)),
        mode="constant", constant_values=255,
    )


def _white_window(src: np.ndarray, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    """The `w x h` window of `src` at `(x0, y0)`, white wherever it falls
    outside `src`."""
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    H, W = src.shape[:2]
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = src[sy0:sy1, sx0:sx1]
    return out


def reorder_quad_reading(quad, angle_deg: float) -> tuple:
    """The quad's 4 corners reordered so p0 -> p1 runs along the reading
    direction `angle_deg` (y-down): p0 top-left, p1 top-right, p2
    bottom-right, p3 bottom-left in the text's own frame."""
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    rad = math.radians(angle_deg)
    u = np.array([math.cos(rad), math.sin(rad)])
    n = np.array([-math.sin(rad), math.cos(rad)])
    along, across = q @ u, q @ n
    top = sorted(np.argsort(across)[:2], key=lambda i: along[i])
    bottom = sorted(np.argsort(across)[2:], key=lambda i: along[i])
    order = [top[0], top[1], bottom[1], bottom[0]]
    return tuple((float(q[i][0]), float(q[i][1])) for i in order)


def _as_bgr(crop: np.ndarray) -> np.ndarray:
    """An already-BGR crop -> a contiguous 3-channel uint8 array, copying
    only when it isn't one already (grayscale, extra channels, a strided
    view)."""
    arr = np.asarray(crop, dtype=np.uint8)
    if arr.ndim == 2:
        return np.ascontiguousarray(np.repeat(arr[:, :, None], 3, axis=2))
    return np.ascontiguousarray(arr[:, :, :3])


def _normalize_bgr(crop: np.ndarray) -> np.ndarray:
    """An RGB image (e.g. a PIL render) -> a 3-channel BGR array (paddleocr
    2.x's TextClassifier/TextRecognizer/TextDetector are cv2/BGR)."""
    arr = np.asarray(crop, dtype=np.uint8)
    if arr.ndim == 2:  # grayscale -- gray RGB and gray BGR are identical
        return np.ascontiguousarray(np.repeat(arr[:, :, None], 3, axis=2))
    return np.ascontiguousarray(arr[:, :, :3][:, :, ::-1])
