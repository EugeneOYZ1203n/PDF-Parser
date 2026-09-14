"""The OCR backend: recognises pre-segmented word crops with PaddleOCR and
returns one `Text` per word.

Text *detection* is not PaddleOCR's job in this pipeline -- FAST (Phase D,
`pipelines/_steps.py::detect_text_fast`) first narrows classification's
clusters down to the ones that look like text, then Radon segmentation
(`OCR/radon.py::segment_clusters`, Phase E, called on every FAST-surviving
cluster) splits each one into word-level `Segment`s, each already carrying
its own deskewed, white-padded crop (`Segment.image`) -- so by the time a
`Segment` reaches this module it *is* one word, already rotated to
(approximately) upright and already OCR-ready: no render and no crop
normalization happen here at all, only a BGR conversion. Similarity dedup (Phase F,
`pipelines/_steps.py::elect_unique_segments`) then elects one representative
`Segment` per repeated shape and it's only those representatives this
module actually recognizes. The only remaining ambiguity Radon's
projection-profile skew estimate cannot resolve is a 0-vs-180-degree flip (a
baseline is a line, not an arrow) -- PaddleOCR's own angle classifier
(`use_angle_cls=True`) resolves it in one extra pass, then `text_recognizer`
runs once per batch. Radon's own residual skew angle (0.0 for an elected
representative's canonicalized copy) and the classifier's 0/180 correction
are combined into one final `direction` before the `Text` is returned --
see `recognize_segments`. (A representative's own canonicalizing rotation,
on top of this, is applied afterward by `pipelines/sub_pipelines/
ocr.py::restore_word_texts` when placing its `Text` back onto each real
word occurrence -- this module never sees that.)

`PaddleRecBackend` builds one `paddleocr.PaddleOCR` engine
(`config.OCR_VERSION` = PP-OCRv4, `config.OCR_LANG`), cached at class scope
by `(ocr_version, lang)` so a spawn pool started next finds the weights on
disk. This is the same API surface `archive/`'s `raster_parser` OCR uses,
so the `legacy` benchmark variant needs no compatibility shim.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from skimage.transform import ProjectiveTransform, SimilarityTransform, warp

from rastervec.P3_Vector_Parsing.FastIntoPaddle.config import (
    OCR_BATCH_SIZE,
    OCR_LANG,
    OCR_VERSION,
    PADDLE_DETECT_MAX_RENDER_DPI,
    PADDLE_DETECT_MIN_RENDER_SIDE_PX,
    PADDLE_WHITE_PAD_FRACTION,
)
from rastervec.commons.helpers.geometry import (
    PDF_POINTS_PER_INCH,
    compute_origin,
    transform_direction,
    union_bbox,
)
from rastervec.commons.logging_setup import get_logger
from rastervec.commons.models import Segment, Text, Vector
from rastervec.commons.renderer import (
    page_points_to_pixel,
    pixel_to_page_bbox,
    pixel_to_page_points,
    render_vector_cluster,
)

_LOG = get_logger("ocr.backend")

# PaddleOCR's own DB detector's det_limit_side_len -- a per-cluster render
# (one text line/word cluster, not a whole page) is rarely anywhere near
# this, so it's a generous ceiling rather than a tuned value.
_DETECT_LIMIT_SIDE_LEN = 4000


@dataclass
class OcrBox:
    """One recognised crop: text, confidence, and the classifier's own
    0/180 flip decision (degrees, 0 or 180)."""

    text: str
    confidence: float
    flip_deg: int = 0


@dataclass
class PaddleDetection:
    """One PaddleOCR-detector result from `PaddleDetectBackend.
    detect_on_cluster`, in page space. `quad` is the detector's own
    4-corner polygon (page-space points, same corner order PaddleOCR
    returns -- clockwise from top-left); `bbox` its axis-aligned envelope.
    `rotation_deg` combines the quad's own edge orientation (full
    precision, not snapped to 90 by us) with PaddleOCR's own
    `use_angle_cls` 0/180 flip -- see `detect_on_cluster`'s docstring for
    the exact recipe. This is a placeholder rotation source -- refined by
    the pipeline's later `rotate` step via `OCR/radon.py::sweep_rotation`
    (pure vector geometry, no pixels)."""

    bbox: "tuple[float, float, float, float]"
    quad: "list[tuple[float, float]]"
    rotation_deg: float


@dataclass
class ClusterDetection:
    """`PaddleDetectBackend.detect_on_cluster`'s full result for one
    cluster: the cluster's own **one** render (BGR, the exact image the
    detector saw, already white-padded via `pad_image` -- see `pad_x_px`/
    `pad_y_px`) plus the dpi it was rendered at, and every detected
    `PaddleDetection` (page space). `pipelines/_steps.py::
    rotate_paddle_detections` crops each detection's own region directly out
    of `image` -- via `dpi`/`pad_x_px`/`pad_y_px`/the cluster's own vectors
    -- instead of re-rendering, per the pipeline's "render each cluster
    once" design."""

    image: "np.ndarray"
    dpi: int
    detections: "list[PaddleDetection]"
    pad_x_px: int = 0
    pad_y_px: int = 0


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
        call. Each crop is used as handed over, only converted to BGR: it
        arrives already deskewed and already white-padded from
        `OCR/radon.py`, and `text_recognizer` does its own resize."""
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
    """The `current` pipeline's per-cluster text-detection step: PaddleOCR's
    own text-DETECTION model (`engine.text_detector`, PP-OCR's DB detector)
    run on one already-rendered `Vector` cluster at a time (never tiled,
    never a whole-page render -- that's `OCR/fast_detect.py::FastDetector`'s
    job). Cached separately from `PaddleRecBackend` (its own
    `_ENGINE_CACHE`) since it's built with `use_angle_cls=True` for the
    rotation-approximation step below, same as `PaddleRecBackend`."""

    _ENGINE_CACHE: dict[tuple[str, str], object] = {}

    def __init__(self, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        self.key = (ocr_version, lang)

    @classmethod
    def warmup(cls, ocr_version: str = OCR_VERSION, lang: str = OCR_LANG) -> None:
        cls(ocr_version, lang)._engine()

    def _engine(self):
        if self.key not in PaddleDetectBackend._ENGINE_CACHE:
            # torch must load before paddle on Windows -- see PaddleRecBackend._engine.
            import torch  # noqa: F401
            from paddleocr import PaddleOCR

            ocr_version, lang = self.key
            PaddleDetectBackend._ENGINE_CACHE[self.key] = PaddleOCR(
                ocr_version=ocr_version,
                lang=lang,
                use_angle_cls=True,
                show_log=False,
                det_limit_side_len=_DETECT_LIMIT_SIDE_LEN,
            )
        return PaddleDetectBackend._ENGINE_CACHE[self.key]

    def detect_on_cluster(
        self, vectors: "list[Vector]", *, dpi: int = 300, padding: float = 0.0,
    ) -> "ClusterDetection | None":
        """Renders `vectors` **once** (`renderer.render_vector_cluster`,
        dynamically dpi-bumped by `dpi_for_cluster` so a small cluster is
        never handed to the detector at a few dozen px), adds a white
        pixel-space margin (`pad_image`, `PADDLE_WHITE_PAD_FRACTION`) so
        text sitting at the cluster's own bbox edge isn't clipped/missed by
        the detector, then runs PaddleOCR's own detector on that padded
        render, and for each returned quad derives a rotation approximation
        using PaddleOCR's own mechanisms only (no custom heuristics):

        1. `_quad_rotation_deg` reads the quad's own dominant-edge
           orientation -- full precision, whatever the DB box itself
           reports (in practice close to axis-aligned, hence this being a
           coarse/placeholder rotation source -- see `PaddleDetection`'s
           own docstring; refined later by `OCR/radon.py::sweep_rotation`).
        2. `_rotate_crop` perspective-corrects that quad's own region out of
           the render onto an upright axis-aligned crop (the standard
           PaddleOCR "get_rotate_crop_image" step, reimplemented with
           `skimage` instead of cv2, per this project's convention).
        3. `engine.text_classifier` (`use_angle_cls`, the same flag
           `PaddleRecBackend` already enables) resolves the residual
           0-vs-180 flip on that upright crop.

        `rotation_deg = quad_rotation_deg + flip`. Each quad is returned by
        the detector in the *padded* render's pixel space; the white-pad
        offset is subtracted before mapping to page space (`renderer.
        pixel_to_page_bbox`/`pixel_to_page_points` invert the *unpadded*
        render). Returns a `ClusterDetection` carrying the one padded
        render (BGR) + the dpi it was made at + the pad offset (so
        `crop_rotated_detection` can crop straight out of it, no re-render)
        plus every page-space `PaddleDetection`. `None` for an empty
        `vectors`."""
        if not vectors:
            return None
        dpi_used = dpi_for_cluster(vectors, dpi, padding)
        image = render_vector_cluster(vectors, dpi_used, padding)
        bgr = _normalize_bgr(np.asarray(image))
        bgr, (pad_x_px, pad_y_px) = pad_image(bgr, PADDLE_WHITE_PAD_FRACTION)

        engine = self._engine()
        dt_boxes, _elapse = engine.text_detector(bgr)

        detections: list[PaddleDetection] = []
        for raw_quad in dt_boxes:
            quad = np.asarray(raw_quad, dtype=np.float64)
            quad_rotation = _quad_rotation_deg(quad)
            crop = _rotate_crop(bgr, quad)
            _, cls_results, _ = engine.text_classifier([crop])
            flip = 180 if str(cls_results[0][0]) == "180" else 0
            rotation_deg = _normalize_rotation(quad_rotation + flip)

            unpadded_quad = (quad - np.array([pad_x_px, pad_y_px])).tolist()
            page_quad = pixel_to_page_points(vectors, dpi_used, unpadded_quad, padding)
            page_bbox = pixel_to_page_bbox(vectors, dpi_used, unpadded_quad, padding)
            detections.append(
                PaddleDetection(bbox=page_bbox, quad=page_quad, rotation_deg=rotation_deg)
            )
        return ClusterDetection(
            image=bgr, dpi=dpi_used, detections=detections,
            pad_x_px=pad_x_px, pad_y_px=pad_y_px,
        )


def pad_image(
    img: np.ndarray, fraction: float = PADDLE_WHITE_PAD_FRACTION,
) -> "tuple[np.ndarray, tuple[int, int]]":
    """Surround `img` (2D grayscale or 3D H,W,C) with a white border of
    `fraction * max(width, height)` px on every side -- both axes padded
    off the *larger* dimension, so a wide/short or narrow/tall crop gets
    equal breathing room on all four sides instead of a border that scales
    independently per axis. Used for exactly two things in this pipeline:
    the cluster render before PaddleOCR's own detector sees it
    (`PaddleDetectBackend.detect_on_cluster`), and each detection's final
    crop before recognition (`crop_rotated_detection`) -- nothing else pads
    or rasterizes (`OCR/radon.py` is pure vector geometry). `pad_x_px`/
    `pad_y_px` are what a caller must subtract to get back into the
    *unpadded* image's pixel space. A zero-area image is returned unchanged
    with a `(0, 0)` offset."""
    if img.size == 0:
        return img, (0, 0)
    pad = int(round(max(img.shape[0], img.shape[1]) * fraction))
    pad_width = ((pad, pad), (pad, pad)) + ((0, 0),) * (img.ndim - 2)
    padded = np.pad(img, pad_width, mode="constant", constant_values=255)
    return padded, (pad, pad)


def rotation_transform(shape_hw: "tuple[int, int]", angle_deg: float):
    """Return `(out_shape, forward, inverse)` for a rotation of `angle_deg`
    about the input centre, onto a resized canvas that fits the whole
    rotated image (same construction skimage's own `rotate(resize=True)`
    uses, so the rotated image and the mapped corners stay consistent).
    `forward(pts_xy)` maps input pixel coords -> output pixel coords;
    `inverse(pts_xy)` maps output -> input. Both take/return `(N, 2)`
    arrays of `(x, y)` (column, row)."""
    h, w = shape_hw
    centre = np.array((w, h)) / 2.0 - 0.5
    tf = (
        SimilarityTransform(translation=-centre)
        + SimilarityTransform(rotation=np.deg2rad(-angle_deg))
        + SimilarityTransform(translation=centre)
    )
    box = np.array([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)], dtype=np.float64)
    out_box = tf(box)
    lo = out_box.min(axis=0)
    hi = out_box.max(axis=0)
    ow = int(np.ceil(hi[0] - lo[0])) + 1
    oh = int(np.ceil(hi[1] - lo[1])) + 1
    forward = tf + SimilarityTransform(translation=-lo)

    def inverse(pts: np.ndarray) -> np.ndarray:
        return forward.inverse(np.asarray(pts, dtype=np.float64))

    return (oh, ow), forward, inverse


def crop_rotated_detection(
    cluster_image: np.ndarray,
    cluster_dpi: int,
    cluster_vectors: "list[Vector]",
    detection_vectors: "list[Vector]",
    theta_deg: float,
    *,
    render_pad: "tuple[int, int]" = (0, 0),
    padding: float = 0.0,
) -> "np.ndarray | None":
    """Rotates the (already white-padded) `cluster_image` upright by
    `theta_deg` about its own centre, crops to `detection_vectors`'s union
    bbox mapped into that rotated pixel frame, then applies a fresh white
    pad (`pad_image`) -- the recognition-side margin. No word/character
    splitting: the whole detection's assigned vectors become one crop.
    `render_pad` is `cluster_image`'s own `(pad_x_px, pad_y_px)` (from
    `ClusterDetection`) -- page-space points must be shifted by it before
    use, since `cluster_image`'s pixel space is the *padded* render, not
    the one `page_points_to_pixel`/`pixel_to_page_bbox` invert. Returns
    `None` if the mapped region is empty."""
    if not detection_vectors:
        return None
    out_shape, forward, _inverse = rotation_transform(cluster_image.shape[:2], theta_deg)
    rotated = warp(
        cluster_image, forward.inverse, output_shape=out_shape,
        cval=255.0, order=1, preserve_range=True,
    ).astype(np.uint8)
    height, width = out_shape

    bbox = union_bbox([v.bbox for v in detection_vectors])
    corners_page = [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]
    corners_px = np.asarray(
        page_points_to_pixel(cluster_vectors, cluster_dpi, corners_page, padding), dtype=np.float64,
    ) + np.array(render_pad)
    rotated_corners = forward(corners_px)
    raw_x0 = float(rotated_corners[:, 0].min())
    raw_x1 = float(rotated_corners[:, 0].max())
    raw_y0 = float(rotated_corners[:, 1].min())
    raw_y1 = float(rotated_corners[:, 1].max())
    rx0 = max(0, int(math.floor(raw_x0)))
    rx1 = min(width, int(math.ceil(raw_x1)))
    ry0 = max(0, int(math.floor(raw_y0)))
    ry1 = min(height, int(math.ceil(raw_y1)))
    if raw_x0 < 0 or raw_y0 < 0 or raw_x1 > width or raw_y1 > height:
        _LOG.warning(
            "crop_rotated_detection: rotated detection bounds (%.1f,%.1f,%.1f,%.1f) "
            "clamped to render (%d,%d) -- content was clipped",
            raw_x0, raw_y0, raw_x1, raw_y1, width, height,
        )
    if rx1 <= rx0 or ry1 <= ry0:
        return None
    crop, _offset = pad_image(rotated[ry0:ry1, rx0:rx1], PADDLE_WHITE_PAD_FRACTION)
    return crop


def dpi_for_cluster(vectors: "list[Vector]", dpi: int, padding: float) -> int:
    """Dynamic dpi-bump rule (never down, capped), scoped to this
    backend's own `PADDLE_DETECT_MIN_RENDER_SIDE_PX`/
    `PADDLE_DETECT_MAX_RENDER_DPI` constants -- PaddleOCR's detector reads a
    tiny crop poorly."""
    x0, y0, x1, y1 = union_bbox([v.bbox for v in vectors])
    min_side_pt = min(x1 - x0, y1 - y0) + 2 * padding
    if min_side_pt <= 0:
        return dpi
    needed_dpi = math.ceil(PADDLE_DETECT_MIN_RENDER_SIDE_PX * PDF_POINTS_PER_INCH / min_side_pt)
    return min(max(dpi, needed_dpi), PADDLE_DETECT_MAX_RENDER_DPI)


def _normalize_rotation(angle_deg: float) -> float:
    """Wrap to `[-90, 90)` -- text direction is a line, not an arrow, so a
    0/180 ambiguity always remains mod 180 (resolved separately by the
    cls-flip term this is added to before calling this)."""
    return ((angle_deg + 90.0) % 180.0) - 90.0


def _quad_rotation_deg(quad: np.ndarray) -> float:
    """The quad's own dominant-edge orientation, as the rotation (degrees,
    counter-clockwise-positive, same sign convention as `OCR/radon.py::
    estimate_skew`) that would bring that edge to horizontal. `quad` is 4
    `(x, y)` pixel points in PaddleOCR's own corner order (clockwise from
    top-left); the longer of the top edge (0->1) and left edge (0->3) is
    taken as the text's own baseline direction, so this is robust to a
    quad that's taller than it is wide (vertical/rotated text)."""
    top = quad[1] - quad[0]
    left = quad[3] - quad[0]
    dx, dy = (top if np.hypot(*top) >= np.hypot(*left) else left)
    theta = math.degrees(math.atan2(dy, dx))
    return _normalize_rotation(-theta)


def _rotate_crop(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-correct crop of `quad` (4 `(x, y)` pixel points,
    PaddleOCR's own corner order) out of `bgr`, warped onto an axis-aligned
    rectangle sized to the quad's own edge lengths -- the standard
    PaddleOCR "get_rotate_crop_image" step, reimplemented with `skimage`
    (no cv2, per this project's convention -- `OCR/radon.py` already
    depends on `skimage.transform`)."""
    width = max(1, int(round(max(np.hypot(*(quad[1] - quad[0])), np.hypot(*(quad[2] - quad[3]))))))
    height = max(1, int(round(max(np.hypot(*(quad[3] - quad[0])), np.hypot(*(quad[2] - quad[1]))))))
    rect = np.array([(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float64)
    tf = ProjectiveTransform()
    tf.estimate(rect, quad)
    warped = warp(bgr, tf, output_shape=(height, width), cval=255.0, preserve_range=True)
    return np.clip(warped, 0, 255).astype(np.uint8)


def _normalize_bgr(crop: np.ndarray) -> np.ndarray:
    """A `Segment.image` -> a 3-channel BGR array (paddleocr 2.x's
    TextClassifier/TextRecognizer are cv2/BGR). Nothing else happens here:
    the crop already carries its white margin from `OCR/radon.py::pad_image`,
    and `text_recognizer` resizes to its own `rec_image_shape` internally,
    so there is no separate crop-normalization pass to run."""
    arr = np.asarray(crop, dtype=np.uint8)
    if arr.ndim == 2:  # grayscale -- gray RGB and gray BGR are identical
        return np.ascontiguousarray(np.repeat(arr[:, :, None], 3, axis=2))
    return np.ascontiguousarray(arr[:, :, :3][:, :, ::-1])


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


def recognize_segments(
    segments: list[Segment],
    *,
    batch_size: int = OCR_BATCH_SIZE,
    recognize_fn: "Callable[[list[np.ndarray]], list[OcrBox]] | None" = None,
) -> list[Text]:
    """Recognises every word-level `Segment` (each already carrying its own
    deskewed crop in `.image` -- no render happens here) in batches of
    `batch_size`, and returns one `Text` per input `Segment`, `source=
    "ocr"`, in the same coordinate frame `segment.vectors` already live in
    -- real page position for a plain word `Segment`, or an elected
    representative's own canonical frame (see `models/segment.py`'s
    docstring). `recognize_fn` defaults to a fresh
    `PaddleRecBackend().recognize_crops`; pass one (e.g. dispatching to a
    shared compute pool -- see `Reader/Parallel`) to replace the actual
    engine call without changing anything else here.

    `direction` combines that word's own Radon residual skew
    (`segment.angle` -- 0.0 for an elected representative's canonicalized
    copy) with the classifier's 0/180 flip correction. A representative's
    own canonicalizing rotation is layered on top of this by `pipelines/
    sub_pipelines/ocr.py::restore_word_texts` when placing it back onto
    each real word occurrence -- not here."""
    recognize_fn = recognize_fn if recognize_fn is not None else PaddleRecBackend().recognize_crops

    bboxes = [union_bbox([v.bbox for v in seg.vectors]) for seg in segments]

    texts: list[Text] = []
    for start in range(0, len(segments), max(1, batch_size)):
        batch = list(zip(segments[start:start + batch_size], bboxes[start:start + batch_size]))
        boxes = recognize_fn([seg.image for seg, _bbox in batch])
        for (seg, bbox), box in zip(batch, boxes):
            direction = transform_direction((1.0, 0.0), seg.angle + box.flip_deg)
            texts.append(Text(
                text=box.text,
                bbox=bbox,
                direction=direction,
                origin=compute_origin(bbox, direction),
                font="", font_size=0.0, color=None, flags=0,
                ascender=None, descender=None, wmode=0,
                block_no=0, line_no=0, word_no=0,
                page_index=0, seqno=0,
                confidence=box.confidence, source="ocr",
                orientation_source="ocr",
            ))
    return texts
