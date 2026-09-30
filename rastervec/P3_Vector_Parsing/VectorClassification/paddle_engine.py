"""The OCR backend: PaddleOCR detection + recognition over seqno-clustered
word groups (`group_filters.py::combine_overlapping_seq`).

`PaddleDetectBackend.detect` runs PaddleOCR's own text-DETECTION model
against a word group's own rendered+padded image, returning every text quad
it finds in that image's own pixel space; `PaddleRecBackend.recognize_crops`
then recognises each detected quad's own crop, built by `hough_deskew` --
an axis-aligned crop of the quad's region, rotated by a raster-refined angle
(a Hough-line reading and a `cv2.minAreaRect` reading, each mod 90,
circular-averaged and snapped to the nearest `config.HOUGH_ANGLE_SNAP_DEG`)
rather than the quad's own perspective warp or corner geometry (the quad's
own dominant-edge angle was tried first and dropped -- too inaccurate in
practice). If that recognition comes back blank, `parse.py` retries the same
crop rotated a further +90/180/270 degrees via `recognize_crops_raw` before
giving up. This mirrors `LegacyRecreation/paddle_engine.py`'s own independent
copy of the same detect-then-recognize pair (itself a port of
archive/raster_parser's `PaddleOcr.ocr_image`, which called PaddleOCR's full
`ocr()` pipeline per word group) -- own duplicated copy here too, per the
"sibling P3 backends share zero code" rule.

`PaddleRecBackend`/`PaddleDetectBackend` each build one `paddleocr.PaddleOCR`
engine (`config.OCR_VERSION` = PP-OCRv4, `config.OCR_LANG`), cached at class
scope by `(ocr_version, lang)` so a spawn pool started next finds the
weights on disk. This is the same API surface `archive/`'s `raster_parser`
OCR uses, so the `legacy` benchmark variant needs no compatibility shim.

This module imports `cv2` (for `_minarea_angle_deg`'s `cv2.minAreaRect`) --
`skimage` still handles everything else (Hough, dilation, rotation) per this
backend's usual convention, but `cv2` is already an indirect dependency via
`paddleocr`, and the repo has no blanket "no cv2" rule (`Junction`'s P2
backend already imports it).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.morphology import binary_dilation, disk
from skimage.transform import hough_line, hough_line_peaks, rotate

from rastervec.P3_Vector_Parsing.VectorClassification.config import (
    HOUGH_ANGLE_SNAP_DEG,
    HOUGH_DILATE_RADIUS_PX,
    HOUGH_INK_THRESHOLD,
    MINAREA_INK_THRESHOLD,
    OCR_BATCH_SIZE,
    OCR_LANG,
    OCR_VERSION,
    ROTATION_AGREEMENT_TOLERANCE_DEG,
)

# PaddleOCR's own DB detector's det_limit_side_len -- a per-word-group render
# is rarely anywhere near this, so it's a generous ceiling rather than a
# tuned value (same constant/rationale as LegacyRecreation's own copy).
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

    def recognize_crops_raw(self, crops: list[np.ndarray]) -> list[OcrBox]:
        """Recognise `crops` as-is, with no `text_classifier` call -- used by
        `parse.py`'s blank-recognition retry sweep, where the caller has
        already rotated each crop to a specific quarter-turn it wants tried
        directly (there is no 0/180 decision left to make). `flip_deg` is
        always 0 on the returned boxes; the caller tracks whatever extra
        rotation it applied itself."""
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
# local call). Mirrors `FastIntoPaddle/paddle_engine.py`'s identical pattern
# (`_recognize_crops_job`) before that module was removed -- `parse.py`
# dispatches these via `compute.starmap`/`compute.apply` when a caller passes
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


def _recognize_crops_job(
    crops: list[np.ndarray], ocr_version: str = OCR_VERSION, lang: str = OCR_LANG,
) -> list[OcrBox]:
    """One page-wide recognize batch as a Pool-2 job (see `parse.py`'s
    `OCR_BATCH_SIZE`-chunked dispatch)."""
    return PaddleRecBackend(ocr_version, lang).recognize_crops(crops)


def _recognize_crops_raw_job(
    crops: list[np.ndarray], ocr_version: str = OCR_VERSION, lang: str = OCR_LANG,
) -> list[OcrBox]:
    """`recognize_crops_raw`'s counterpart to `_recognize_crops_job`, for
    `parse.py`'s page-wide blank-recognition retry sweep."""
    return PaddleRecBackend(ocr_version, lang).recognize_crops_raw(crops)


def _normalize_rotation(angle_deg: float) -> float:
    """Wrap to `[-90, 90)` -- text direction is a line, not an arrow, so a
    0/180 ambiguity always remains mod 180 (resolved separately by the
    cls-flip term this is added to before calling this). Own duplicated copy
    of `LegacyRecreation`'s `paddle_engine.py::_normalize_rotation`."""
    return ((angle_deg + 90.0) % 180.0) - 90.0


def _axis_aligned_crop(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Axis-aligned bbox crop of `quad`'s region straight out of `bgr` -- no
    perspective warp, unlike the old `_rotate_crop` this replaces for
    rotation purposes (see `hough_deskew`). Expanded `_CROP_EXPAND_FRACTION`
    about the quad's own centroid before cropping (so a tight detection
    still keeps full glyph strokes/ascenders/descenders), clamped to
    `bgr`'s own bounds, plus a flat `_CROP_BORDER_PX` white border."""
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


def _grayscale(crop: np.ndarray) -> np.ndarray:
    """`crop` (BGR uint8) -> single-channel 0..1 grayscale, computed once
    per quad in `hough_deskew` and shared by both `_hough_ink_mask` and
    `_minarea_ink_mask` -- they used to each call this independently on the
    same crop (differing only in the threshold applied afterward), doubling
    this conversion's cost per quad for no reason."""
    return rgb2gray(np.asarray(crop, dtype=np.uint8)[:, :, ::-1])  # BGR -> RGB -> gray, 0..1


def _ink_mask(gray: np.ndarray, threshold: int) -> np.ndarray:
    """Binary ink mask (dark-on-light text/line-art strokes) below
    `threshold` (0-255 grayscale) from an already-grayscaled `gray` array
    (see `_grayscale`). Shared thresholding step for both `_hough_ink_mask`
    (dilated) and `_minarea_ink_mask` (not)."""
    return gray < (threshold / 255.0)


def _hough_ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary ink mask thickened by `binary_dilation` ("increase ink colors"
    before Hough gets a look) so thin or broken strokes still form a
    continuous line for Hough to find. No cv2 here, per this module's usual
    convention -- `skimage.morphology` instead."""
    ink = _ink_mask(gray, HOUGH_INK_THRESHOLD)
    return binary_dilation(ink, footprint=disk(HOUGH_DILATE_RADIUS_PX))


def _minarea_ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary ink mask for `_minarea_angle_deg`, with its own independently
    tunable `MINAREA_INK_THRESHOLD` and (unlike Hough's) no dilation --
    thickening ink is a Hough-specific trick to help a broken stroke form
    one continuous line; a bounding-rect fit doesn't need it."""
    return _ink_mask(gray, MINAREA_INK_THRESHOLD)


def _hough_angle_deg(mask: np.ndarray) -> "float | None":
    """The dominant line's rotation (degrees, the rotation that would bring
    that line to horizontal) from a single-peak Hough line transform over a
    binary ink `mask`. `None` if the mask is empty or no usable peak is
    found (a Hough reading isn't always available -- `_combined_rotation_deg`
    falls back to the minAreaRect reading alone, or to 0, in that case)."""
    if not mask.any():
        return None
    thetas = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)
    accum, angles, _dists = hough_line_peaks(*hough_line(mask, theta=thetas), num_peaks=1)
    if len(angles) == 0:
        return None
    # skimage's hough theta is the line's own NORMAL angle from the x-axis;
    # a horizontal line has theta = +-90 deg. Converting to "rotation needed
    # to bring the line to horizontal" (theta - 90, then wrapped) is the
    # convention `_minarea_angle_deg` also lands on, so the two angles are
    # directly comparable/averageable.
    theta_deg = math.degrees(angles[0])
    return _normalize_rotation(theta_deg - 90.0)


def _minarea_angle_deg(mask: np.ndarray) -> "float | None":
    """The dominant orientation (degrees) of `cv2.minAreaRect`'s minimum-area
    bounding box over `mask`'s own ink pixels, reduced mod 90. `None` if the
    mask is empty. `cv2.minAreaRect`'s own returned angle already lands in
    the same mod-90 convention `_hough_angle_deg` uses (the rotation that
    would bring the dominant edge to horizontal) -- verified empirically
    against synthetic tilted masks at a range of angles (see
    `tests/.../test_paddle_engine.py`), so no sign conversion is needed."""
    if not mask.any():
        return None
    points = cv2.findNonZero(np.asarray(mask, dtype=np.uint8))
    if points is None or len(points) < 2:
        return None
    _center, _size, angle = cv2.minAreaRect(points)
    return float(angle)


def _circular_avg_mod90(a_deg: float, b_deg: float) -> float:
    """Circular mean of two angles on a period-90 circle (0 deg and 90 deg
    are the same point), e.g. (89, 1) -> 0, (45, 43) -> 44, (3, 4) -> 3.5.
    Represents each angle as a point on the unit circle at 4x its value (so
    the period-90 circle maps onto the ordinary period-360 one), averages
    the two unit vectors, and maps back. Falls back to a plain arithmetic
    mean for the degenerate case where the two angles are exactly 45 deg
    apart on this circle (antipodal -- no unique average)."""
    a = math.radians(a_deg * 4.0)
    b = math.radians(b_deg * 4.0)
    x = math.cos(a) + math.cos(b)
    y = math.sin(a) + math.sin(b)
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return ((a_deg + b_deg) / 2.0) % 90.0
    # Round before the final mod: atan2 on near-exact-zero components (e.g.
    # averaging 89 and 1, which should land exactly on 0) can come out a
    # tiny epsilon on the negative side of zero due to floating point, which
    # `% 90.0` would otherwise wrap to ~90 instead of ~0.
    return round(math.degrees(math.atan2(y, x)) / 4.0, 9) % 90.0


def _to_signed_small_angle(mod90_deg: float) -> float:
    """A `[0, 90)` mod-90 angle -> its nearest signed representative in
    `(-45, 45]`, e.g. 88 -> -2, 2 -> 2 -- the applicable rotation a mod-90
    value like `_circular_avg_mod90`'s output actually represents."""
    v = mod90_deg % 90.0
    return v - 90.0 if v > 45.0 else v


def _snap_to_angle_grid(angle_deg: float, step_deg: float = HOUGH_ANGLE_SNAP_DEG) -> float:
    return round(angle_deg / step_deg) * step_deg


def _mod90_circular_distance(a_deg: float, b_deg: float) -> float:
    """Smallest separation between two angles already reduced mod 90 (each
    in `[0, 90)`), on the periodic mod-90 circle where 0 deg and 90 deg are
    the same point (e.g. 89 and 1 are 2 deg apart, not 88)."""
    diff = abs(a_deg - b_deg) % 90.0
    return min(diff, 90.0 - diff)


def _combined_rotation_deg(
    hough_angle_deg: "float | None", minarea_angle_deg: "float | None",
) -> float:
    """The final rotation to apply: ONLY when both `hough_angle_deg` and
    `minarea_angle_deg` are available AND agree -- their mod-90 reductions
    within `ROTATION_AGREEMENT_TOLERANCE_DEG` of each other -- circular-
    average them, convert back to a signed small-angle correction, then
    snap to the nearest `HOUGH_ANGLE_SNAP_DEG`. Either reading missing, or
    the two disagreeing beyond tolerance, returns `0.0` -- a single
    uncorroborated raster reading is no longer trusted alone."""
    if hough_angle_deg is None or minarea_angle_deg is None:
        return 0.0
    hough_mod = hough_angle_deg % 90.0
    minarea_mod = minarea_angle_deg % 90.0
    if _mod90_circular_distance(hough_mod, minarea_mod) > ROTATION_AGREEMENT_TOLERANCE_DEG:
        return 0.0
    combined_mod = _circular_avg_mod90(hough_mod, minarea_mod)
    return _snap_to_angle_grid(_to_signed_small_angle(combined_mod))


@dataclass
class RotationDebug:
    """One detection's rotation provenance, for the rotation/retry debug
    layers and image folders (`parse.py`). `base_crop`/`dilated_ink_mask`/
    `minarea_mask` are the pre-rotation axis-aligned crop and the two ink
    masks Hough/minAreaRect actually ran on -- kept only for debug-image
    rendering (`scripts/debug_image_savers.py`), never consumed by the
    pipeline itself, and populated only when `hough_deskew` is called with
    `keep_debug=True` (`None` otherwise -- see that function); `dilated_
    ink_mask`/`minarea_mask` additionally stay `None` under `keep_debug=True`
    whenever `allow_rotation=False`, since they're never built at all in
    that case (`base_crop` still is -- it's just the axis-aligned crop,
    needed regardless of whether rotation is attempted). `hough_angle_deg`/
    `minarea_angle_deg` also stay `None` (and `combined_angle_deg` stays
    `0.0`) whenever the caller passed `allow_rotation=False` -- the
    estimators are never even run for a quad the caller decided has no
    reliable baseline direction (see `parse.py::_quad_allows_rotation`).
    `flip_angle_deg`/`retry_count`/`best_angle_deg` are filled in by
    `parse.py` after recognition (and any blank-retry passes) runs, not by
    `hough_deskew` itself -- see `parse.py`'s per-quad loop."""

    hough_angle_deg: "float | None"
    minarea_angle_deg: "float | None"
    combined_angle_deg: float
    base_crop: "np.ndarray | None" = None
    dilated_ink_mask: "np.ndarray | None" = None
    minarea_mask: "np.ndarray | None" = None
    flip_angle_deg: "float | None" = None
    retry_count: "int | None" = None
    best_angle_deg: "float | None" = None


def hough_deskew(
    bgr: np.ndarray, quad: np.ndarray, *, keep_debug: bool = False, allow_rotation: bool = True,
) -> "tuple[np.ndarray, RotationDebug]":
    """Builds the axis-aligned crop for `quad` (`_axis_aligned_crop`), then
    rotates it by the raster-refined combined angle (a Hough-line reading
    and a `cv2.minAreaRect` reading, each mod 90, circular-averaged and
    snapped to the nearest `HOUGH_ANGLE_SNAP_DEG`, and only trusted at all
    when the two agree -- see `_combined_rotation_deg`) so the crop's
    baseline lands on a quarter-turn boundary -- the crop `recognize_crops`'s
    own 0/180 classifier then resolves the final flip on (see `parse.py`).
    Replaces the old `_rotate_crop` perspective-warp approach (and the quad's
    own corner-geometry angle, dropped for being too inaccurate) entirely:
    this crop's rotation comes solely from `_combined_rotation_deg`.

    `allow_rotation=False` (set by the caller, `parse.py::
    _quad_allows_rotation`, when the quad's underlying vectors collapse to a
    single connected component -- no reliable baseline direction) skips the
    grayscale/ink-mask/Hough/minAreaRect estimation entirely: both angles stay `None`, the
    combined correction is `0.0`, and the crop is returned unrotated. This
    also means the estimators are never run for a quad that was going to be
    forced to `0.0` anyway.

    `keep_debug` controls only whether the pre-rotation crop and the two ink
    masks are *retained* on the returned `RotationDebug` (`base_crop`/
    `dilated_ink_mask`/`minarea_mask` -- `None` otherwise, the default):
    `base_crop` is always *built* either way (it's just the axis-aligned
    crop `_axis_aligned_crop` produces, needed regardless of whether
    rotation is attempted) and so is retained under `keep_debug=True` even
    when `allow_rotation=False`; `dilated_ink_mask`/`minarea_mask` are only
    ever built when `allow_rotation=True` (the angles depend on them) and
    so stay `None` under `keep_debug=True` whenever `allow_rotation=False`.
    A caller processing a whole page's worth of quads (`parse.py`) would
    otherwise hold every one of these extra full-size arrays alive for the
    rest of its OCR pass even though nothing but `scripts/
    debug_image_savers.py`'s debug-image dumpers ever reads them. `parse.py`
    passes `keep_debug=True` only when a caller actually asked for debug
    output (`debug_out is not None`)."""
    base = _axis_aligned_crop(bgr, quad)
    hough_angle = minarea_angle = None
    hough_mask = minarea_mask = None
    combined = 0.0
    if allow_rotation:
        gray = _grayscale(base)
        hough_mask = _hough_ink_mask(gray)
        minarea_mask = _minarea_ink_mask(gray)
        hough_angle = _hough_angle_deg(hough_mask)
        minarea_angle = _minarea_angle_deg(minarea_mask)
        combined = _combined_rotation_deg(hough_angle, minarea_angle)
    if combined == 0.0:
        rotated = base
    else:
        rotated = rotate(
            base, combined, resize=True, cval=255.0, order=1, preserve_range=True,
        ).astype(np.uint8)
    return rotated, RotationDebug(
        hough_angle_deg=hough_angle, minarea_angle_deg=minarea_angle, combined_angle_deg=combined,
        base_crop=base if keep_debug else None,
        dilated_ink_mask=hough_mask if keep_debug else None,
        minarea_mask=minarea_mask if keep_debug else None,
    )


def _normalize_bgr(crop: np.ndarray) -> np.ndarray:
    """A crop -> a 3-channel BGR array (paddleocr 2.x's
    TextClassifier/TextRecognizer/TextDetector are cv2/BGR)."""
    arr = np.asarray(crop, dtype=np.uint8)
    if arr.ndim == 2:  # grayscale -- gray RGB and gray BGR are identical
        return np.ascontiguousarray(np.repeat(arr[:, :, None], 3, axis=2))
    return np.ascontiguousarray(arr[:, :, :3][:, :, ::-1])
