"""Per-backend pre-OCR debug image dumpers for `generate_pipeline_report.py`.
Each P3 backend's own `debug_out` dict (stashed at `res.extra["p3_debug"]"
-- see `core/result.py` -- when `run_pipeline(..., verbose=True)`) is the
only source of this data: the old, pre-phase-split `PipelineResult` had
these as top-level attributes (`spatial_clusters`/`cluster_detections`/
`rotated_segments`/`restored_texts`/`fast_result`), but the current
`core/result.py::PipelineResult` has none of them -- reading those
attributes off `res` (as this used to) always returns `None`/`[]` and
silently writes empty folders. Each backend gets its own distinct folder
set (see `generate_pipeline_report.py`'s module docstring) since their
internal pipelines genuinely differ -- LegacyRecreation has no FAST stage.

Every saver writes through an `_ImageReservoir`: the report generator
keeps one per folder per input document, so a folder holds at most
`_DEBUG_IMAGE_CAP` images drawn uniformly at random (fixed seed) from every
page, not the first N. A saver also accepts a plain folder `Path`
(uncapped -- every image is written).
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

_DEBUG_IMAGE_CAP = 100


class _ImageReservoir:
    """Disk-backed reservoir sample (Algorithm R) of at most `cap` images
    into `folder`: the n-th image offered is kept with probability cap/n,
    replacing (and deleting) a random earlier pick. `make_image` is only
    called for an image that is kept, so a rejected crop is never
    encoded. `cap=None` keeps everything. The folder is created on the
    first kept image only."""

    def __init__(self, folder: Path, cap: "int | None" = _DEBUG_IMAGE_CAP, seed: int = 0) -> None:
        self.folder = Path(folder)
        self.cap = cap
        self.seen = 0
        self.kept: list[Path] = []
        self._rng = random.Random(seed)

    def offer(self, name: str, make_image: "Callable[[], Image.Image]") -> bool:
        self.seen += 1
        slot = None
        if self.cap is not None and len(self.kept) >= self.cap:
            slot = self._rng.randrange(self.seen)
            if slot >= self.cap:
                return False
        image = make_image()
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / name
        image.save(path)
        if slot is None:
            self.kept.append(path)
        else:
            if self.kept[slot] != path:
                self.kept[slot].unlink(missing_ok=True)
            self.kept[slot] = path
        return True


def _as_reservoir(target: "_ImageReservoir | Path") -> _ImageReservoir:
    return target if isinstance(target, _ImageReservoir) else _ImageReservoir(target, cap=None)


def _safe_slug(text: str, limit: int = 40) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in (text or "")).strip("_")
    return keep[:limit] or "blank"


def _draw_boxes(img: Image.Image, boxes, outline=(220, 30, 30), width=2) -> Image.Image:
    out = img.convert("RGB")
    d = ImageDraw.Draw(out)
    for b in boxes:
        if b is None:
            continue
        x0, y0, x1, y1 = b
        d.rectangle(
            [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
            outline=outline, width=width,
        )
    return out


def _save_fastintopaddle_detect_images(p3_debug: dict, target, page_index: int) -> int:
    """One PNG per cluster -- exactly what `PaddleDetectBackend.
    detect_on_cluster` saw (`ClusterDetection.image`, already white-padded),
    with every detected box drawn on top. Each `PaddleDetection.bbox` is
    page space, so it must be mapped back into that render's own pixel
    space via `page_points_to_pixel` using the *same* `padding` the render
    itself used (`cluster_render_padding`), then shifted by the render's
    own white-pad offset (`cd.pad_x_px`/`cd.pad_y_px`) -- getting either
    wrong silently misaligns the overlay boxes. Reads
    `p3_debug["clusters"]`/`p3_debug["cluster_detections"]`."""
    clusters = p3_debug.get("clusters") or []
    cluster_detections = p3_debug.get("cluster_detections") or []
    if not clusters or not cluster_detections:
        return 0
    from rastervec.P3_Vector_Parsing.FastIntoPaddle.steps import cluster_render_padding
    from rastervec.commons.renderer import page_points_to_pixel

    reservoir = _as_reservoir(target)

    def _make(cd, cluster) -> Image.Image:
        padding = cluster_render_padding(cluster) if cluster else 0.0
        img = Image.fromarray(np.asarray(cd.image)[..., ::-1])  # BGR -> RGB
        boxes = []
        for det in cd.detections:
            (px0, py0), (px1, py1) = page_points_to_pixel(
                cluster, cd.dpi, [(det.bbox[0], det.bbox[1]), (det.bbox[2], det.bbox[3])],
                padding=padding,
            )
            boxes.append((px0 + cd.pad_x_px, py0 + cd.pad_y_px, px1 + cd.pad_x_px, py1 + cd.pad_y_px))
        return _draw_boxes(img, boxes, outline=(220, 30, 30))

    n = 0
    for i, cd in enumerate(cluster_detections):
        if cd is None or cd.image is None:
            continue
        cluster = clusters[i] if i < len(clusters) else []
        n += reservoir.offer(
            f"p{page_index}_cluster_{i:03d}.png", lambda cd=cd, cluster=cluster: _make(cd, cluster),
        )
    return n


def _save_segment_recog_images(segs: list, texts: list, target, page_index: int) -> int:
    """Shared body for FastIntoPaddle's/VectorClassification's own
    recog-image dumpers -- both hand PaddleOCR recognition a list of
    `commons.models.Segment` (each already carrying its own crop in
    `.image`) 1:1-aligned with a `list[Text]` of what got recognised."""
    if not segs:
        return 0
    reservoir = _as_reservoir(target)
    n = 0
    for i, seg in enumerate(segs):
        if seg.image is None:
            continue
        rec = texts[i].text if i < len(texts) else ""
        n += reservoir.offer(
            f"p{page_index}_word_{i:03d}__{_safe_slug(rec)}.png",
            lambda seg=seg: Image.fromarray(np.asarray(seg.image)),
        )
    return n


def _save_fastintopaddle_recog_images(p3_debug: dict, target, page_index: int) -> int:
    """One PNG per rotated detection crop -- the exact crop handed to
    PaddleOCR recognition (`Segment.image`), recognised text in the
    filename. Reads `p3_debug["segments"]` + `p3_debug["texts"]`."""
    return _save_segment_recog_images(
        p3_debug.get("segments") or [], p3_debug.get("texts") or [], target, page_index,
    )


def _save_crop_text_images(crops: list, target, page_index: int) -> int:
    """Shared body for LegacyRecreation's/VectorClassification's own
    recog-image dumpers -- both hand PaddleOCR recognition a list of
    `(crop, text)` pairs (one per PaddleOCR-detected quad within a
    rendered cluster/word-group), reads `p3_debug["ocr_crops"]`."""
    if not crops:
        return 0
    reservoir = _as_reservoir(target)
    n = 0
    for i, (crop, rec) in enumerate(crops):
        n += reservoir.offer(
            f"p{page_index}_word_{i:03d}__{_safe_slug(rec)}.png",
            lambda crop=crop: Image.fromarray(np.asarray(crop)),
        )
    return n


def _save_vectorclassification_recog_images(p3_debug: dict, target, page_index: int) -> int:
    """One PNG per PaddleOCR-recognised crop -- exactly what the recognizer
    saw, recognised text in the filename. Reads `p3_debug["ocr_crops"]`
    (`list[tuple[np.ndarray, str]]`)."""
    return _save_crop_text_images(p3_debug.get("ocr_crops") or [], target, page_index)


def _save_vectorclassification_detect_images(p3_debug: dict, target, page_index: int) -> int:
    """One PNG per seqno-cluster's own rendered+padded image, with every
    detected quad drawn on top -- exactly what `PaddleDetectBackend.detect`
    saw. Quads are already in that image's own pixel space (no page-space
    round-trip needed, unlike FastIntoPaddle's detect-image saver, since
    `parse.py` renders/pads each cluster itself and hands the detector that
    same array). Reads `p3_debug["cluster_detections"]`
    (`list[tuple[np.ndarray, list[np.ndarray]]]`, each a `(bgr, quads)`
    pair)."""
    entries = p3_debug.get("cluster_detections") or []
    if not entries:
        return 0
    reservoir = _as_reservoir(target)

    def _make(bgr, quads) -> Image.Image:
        img = Image.fromarray(np.asarray(bgr)[..., ::-1])  # BGR -> RGB
        draw = ImageDraw.Draw(img)
        for quad in quads or []:
            pts = [(float(x), float(y)) for x, y in quad]
            draw.polygon(pts, outline=(220, 30, 30), width=2)
        return img

    n = 0
    for i, (bgr, quads) in enumerate(entries):
        n += reservoir.offer(
            f"p{page_index}_cluster_{i:03d}.png", lambda bgr=bgr, quads=quads: _make(bgr, quads),
        )
    return n


def _draw_angle_line(img: Image.Image, angle_deg: "float | None", color) -> Image.Image:
    """A short line through the image's own center, tilted by `angle_deg`
    (same sign convention as `paddle_engine.py::_hough_angle_deg`/
    `_minarea_angle_deg` -- counter-clockwise-positive, the rotation that
    would bring a line at this angle to horizontal). No-op if `angle_deg` is
    `None`."""
    if angle_deg is None:
        return img
    out = img.convert("RGB")
    draw = ImageDraw.Draw(out)
    w, h = out.size
    cx, cy = w / 2.0, h / 2.0
    length = 0.4 * max(w, h)
    rad = math.radians(-angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    draw.line(
        [(cx - dx * length, cy - dy * length), (cx + dx * length, cy + dy * length)],
        fill=color, width=2,
    )
    return out


def _save_vectorclassification_hough_images(p3_debug: dict, target, page_index: int) -> int:
    """One PNG per detection's own dilated ink mask (what Hough line
    detection actually ran on), with a line drawn through it at the
    detected Hough angle and the hough/minarea/combined angle values in the
    filename. Reads `p3_debug["rotation"]` (see `paddle_engine.py::
    RotationDebug` / `parse.py`'s per-quad loop)."""
    entries = p3_debug.get("rotation") or []
    if not entries:
        return 0
    reservoir = _as_reservoir(target)

    def _make(entry) -> Image.Image:
        mask = entry.get("dilated_ink_mask")
        if mask is not None:
            img = Image.fromarray((np.asarray(mask) * 255).astype(np.uint8)).convert("RGB")
        else:
            img = Image.fromarray(np.asarray(entry["base_crop"])[..., ::-1])  # BGR -> RGB
        return _draw_angle_line(img, entry.get("hough_angle_deg"), (220, 30, 30))

    def _fmt(v) -> str:
        return "na" if v is None else f"{v:.1f}"

    n = 0
    for i, entry in enumerate(entries):
        if entry.get("base_crop") is None:
            continue
        name = (
            f"p{page_index}_det_{i:03d}"
            f"__h{_fmt(entry.get('hough_angle_deg'))}"
            f"__m{_fmt(entry.get('minarea_angle_deg'))}"
            f"__c{_fmt(entry.get('combined_angle_deg'))}.png"
        )
        n += reservoir.offer(name, lambda entry=entry: _make(entry))
    return n


def _save_vectorclassification_minarea_images(p3_debug: dict, target, page_index: int) -> int:
    """One PNG per detection's own non-dilated ink mask (what
    `cv2.minAreaRect` actually ran on), with a line drawn through it at the
    detected minAreaRect angle and the hough/minarea/combined angle values in
    the filename -- mirrors `_save_vectorclassification_hough_images` exactly,
    reading the same `p3_debug["rotation"]` entries' `minarea_mask`/
    `minarea_angle_deg` fields instead."""
    entries = p3_debug.get("rotation") or []
    if not entries:
        return 0
    reservoir = _as_reservoir(target)

    def _make(entry) -> Image.Image:
        mask = entry.get("minarea_mask")
        if mask is not None:
            img = Image.fromarray((np.asarray(mask) * 255).astype(np.uint8)).convert("RGB")
        else:
            img = Image.fromarray(np.asarray(entry["base_crop"])[..., ::-1])  # BGR -> RGB
        return _draw_angle_line(img, entry.get("minarea_angle_deg"), (220, 30, 30))

    def _fmt(v) -> str:
        return "na" if v is None else f"{v:.1f}"

    n = 0
    for i, entry in enumerate(entries):
        if entry.get("base_crop") is None:
            continue
        name = (
            f"p{page_index}_det_{i:03d}"
            f"__h{_fmt(entry.get('hough_angle_deg'))}"
            f"__m{_fmt(entry.get('minarea_angle_deg'))}"
            f"__c{_fmt(entry.get('combined_angle_deg'))}.png"
        )
        n += reservoir.offer(name, lambda entry=entry: _make(entry))
    return n


def _save_vectorclassification_classifier_before_images(p3_debug: dict, target, page_index: int) -> int:
    """One PNG per detection's own crop exactly as handed to
    `recognize_crops`, BEFORE its 0/180 classifier's physical flip. Reads
    `p3_debug["classifier_crops"]` (`list[tuple[np.ndarray, np.ndarray]]`,
    each a `(before, after)` pair -- see `parse.py`)."""
    pairs = p3_debug.get("classifier_crops") or []
    if not pairs:
        return 0
    reservoir = _as_reservoir(target)
    n = 0
    for i, (before, _after) in enumerate(pairs):
        n += reservoir.offer(
            f"p{page_index}_det_{i:03d}.png", lambda before=before: Image.fromarray(np.asarray(before)),
        )
    return n


def _save_vectorclassification_classifier_after_images(p3_debug: dict, target, page_index: int) -> int:
    """One PNG per detection's own crop AFTER the classifier's 0/180
    physical flip -- exactly what `text_recognizer` actually saw. Reads
    `p3_debug["classifier_crops"]`, same pairing as the `_before` saver."""
    pairs = p3_debug.get("classifier_crops") or []
    if not pairs:
        return 0
    reservoir = _as_reservoir(target)
    n = 0
    for i, (_before, after) in enumerate(pairs):
        n += reservoir.offer(
            f"p{page_index}_det_{i:03d}.png", lambda after=after: Image.fromarray(np.asarray(after)),
        )
    return n


def _save_legacyrecreation_ocr_images(p3_debug: dict, target, page_index: int) -> int:
    """One PNG per word group's own padded/DPI-boosted OCR render -- exactly
    what PaddleOCR's (recognition-only) engine saw, recognised text in the
    filename. Reads `p3_debug["ocr_crops"]` (`list[tuple[np.ndarray,
    str]]`)."""
    return _save_crop_text_images(p3_debug.get("ocr_crops") or [], target, page_index)
