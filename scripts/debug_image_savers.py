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

VectorClassification's own savers are organized by *which model/algorithm
call the image was the actual input to*, not by pipeline stage name, and
every saved image is exactly that input array (only a BGR<->RGB channel
reorder for correct PNG display, never an overlay or annotation drawn on
top): `for_paddle_detect/` (PaddleDetectBackend.detect), `for_rotation_
correction/{hough_line,minarea_rect,paddle_classifier}/` (the two raster
angle-estimation masks and the recognizer's own angle-classifier call), and
`for_paddle_recog/{0_retry,1_retry,2_retry,3_retry,failed}/` (whichever
recognize_crops/recognize_crops_raw pass decided a detection's outcome).
LegacyRecreation keeps its own single `paddle_ocr_images/` folder.

Every saver writes through an `_ImageReservoir`: the report generator keeps
one per leaf folder per input document, so a folder holds at most
`ReportConfig.debug_image_cap` images (default `_DEBUG_IMAGE_CAP`) drawn
uniformly at random (fixed seed) from every page, not the first N. A saver
also accepts a plain folder `Path` (uncapped -- every image is written).
"""
from __future__ import annotations

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


def _save_vectorclassification_detect_images(p3_debug: dict, target, page_index: int) -> int:
    """One PNG per seqno-cluster's own rendered+padded image -- exactly
    (channel order aside) what `PaddleDetectBackend.detect` saw, no overlay.
    Reads `p3_debug["cluster_detections"]` (`list[tuple[np.ndarray,
    list[np.ndarray]]]`, each a `(bgr, quads)` pair -- `quads` isn't needed
    here any more, since the detected boxes are no longer drawn on top)."""
    entries = p3_debug.get("cluster_detections") or []
    if not entries:
        return 0
    reservoir = _as_reservoir(target)
    n = 0
    for i, (bgr, _quads) in enumerate(entries):
        n += reservoir.offer(
            f"p{page_index}_cluster_{i:03d}.png",
            lambda bgr=bgr: Image.fromarray(np.asarray(bgr)[..., ::-1]),  # BGR -> RGB
        )
    return n


def _mask_to_image(mask) -> Image.Image:
    """A boolean/0-1 ink mask -> a viewable grayscale PNG -- a dtype
    conversion for encoding, not an overlay; no pixel's ink/non-ink value is
    changed."""
    return Image.fromarray((np.asarray(mask) * 255).astype(np.uint8))


def _save_vectorclassification_rotation_hough_images(p3_debug: dict, target, page_index: int) -> int:
    """One PNG per detection's own dilated ink mask -- exactly what
    `_hough_angle_deg` ran on, no overlay. Entries with no mask (`parse.py::
    _quad_allows_rotation` returned `False` for that quad, so the mask was
    never built) are skipped entirely -- there is no "what Hough saw" image
    for them. Reads `p3_debug["rotation"]` (see `paddle_engine.py::
    RotationDebug` / `parse.py`'s per-quad loop)."""
    entries = p3_debug.get("rotation") or []
    if not entries:
        return 0
    reservoir = _as_reservoir(target)
    n = 0
    for i, entry in enumerate(entries):
        mask = entry.get("dilated_ink_mask")
        if mask is None:
            continue
        n += reservoir.offer(f"p{page_index}_det_{i:03d}.png", lambda mask=mask: _mask_to_image(mask))
    return n


def _save_vectorclassification_rotation_minarea_images(p3_debug: dict, target, page_index: int) -> int:
    """One PNG per detection's own non-dilated ink mask -- exactly what
    `cv2.minAreaRect` ran on, no overlay. Mirrors
    `_save_vectorclassification_rotation_hough_images` exactly, reading the
    same `p3_debug["rotation"]` entries' `minarea_mask` field instead (same
    skip-if-`None` rule)."""
    entries = p3_debug.get("rotation") or []
    if not entries:
        return 0
    reservoir = _as_reservoir(target)
    n = 0
    for i, entry in enumerate(entries):
        mask = entry.get("minarea_mask")
        if mask is None:
            continue
        n += reservoir.offer(f"p{page_index}_det_{i:03d}.png", lambda mask=mask: _mask_to_image(mask))
    return n


def _save_vectorclassification_rotation_classifier_images(p3_debug: dict, target, page_index: int) -> int:
    """One PNG per detected quad -- exactly the crop handed to
    `PaddleRecBackend.recognize_crops`'s own internal `text_classifier` call
    (pass 1, pre-flip; every detection goes through the classifier exactly
    once, win or lose). Reads `p3_debug["paddle_classifier_crops"]`
    (`list[np.ndarray]`)."""
    crops = p3_debug.get("paddle_classifier_crops") or []
    if not crops:
        return 0
    reservoir = _as_reservoir(target)
    n = 0
    for i, crop in enumerate(crops):
        n += reservoir.offer(
            f"p{page_index}_det_{i:03d}.png", lambda crop=crop: Image.fromarray(np.asarray(crop)),
        )
    return n


def _save_vectorclassification_recog_bucket_images(
    p3_debug: dict, target, page_index: int, bucket: str,
) -> int:
    """One PNG per detection whose outcome matches `bucket` -- exactly the
    crop handed to `recognize_crops`/`recognize_crops_raw` for the pass that
    decided this detection's result: `"0"` (recognized on pass 1), `"1"`/
    `"2"`/`"3"` (recovered on that blank-retry pass), or `"failed"` (still
    blank after every retry -- the last rotation variant tried). Reads
    `p3_debug["recog_bucket_crops"][bucket]` (`list[tuple[np.ndarray,
    str]]`, see `parse.py`)."""
    entries = (p3_debug.get("recog_bucket_crops") or {}).get(bucket) or []
    if not entries:
        return 0
    reservoir = _as_reservoir(target)
    n = 0
    for i, (crop, rec) in enumerate(entries):
        n += reservoir.offer(
            f"p{page_index}_det_{i:03d}__{_safe_slug(rec)}.png",
            lambda crop=crop: Image.fromarray(np.asarray(crop)),
        )
    return n


def _save_vectorclassification_fast_images(
    p3_debug: dict, input_target, heatmap_target, page_index: int,
) -> int:
    """Per FAST detect group (post-recognition FAST text/drawing split,
    accepted groups only): `input_target` gets exactly the RGB array
    `FastDetector.detect` saw (the group crop, white-padded to
    `FAST_CROP_MAX_ASPECT`), `heatmap_target` its score map cut back to the
    unpadded crop as grayscale (white = 1.0). Same file name in both
    folders, so a pair lines up. Reads `p3_debug["fast_images"]`
    (`list[tuple[np.ndarray, np.ndarray]]`, only filled with
    `keep_debug_arrays`). The two folders sample independently when capped."""
    entries = p3_debug.get("fast_images") or []
    if not entries:
        return 0
    inputs, heatmaps = _as_reservoir(input_target), _as_reservoir(heatmap_target)
    n = 0
    for i, (rgb, heat) in enumerate(entries):
        name = f"p{page_index}_group_{i:03d}.png"
        n += inputs.offer(name, lambda rgb=rgb: Image.fromarray(np.asarray(rgb)))
        heatmaps.offer(name, lambda heat=heat: _mask_to_image(np.clip(heat, 0.0, 1.0)))
    return n


def _save_legacyrecreation_ocr_images(p3_debug: dict, target, page_index: int) -> int:
    """One PNG per word group's own padded/DPI-boosted OCR render -- exactly
    what PaddleOCR's (recognition-only) engine saw, recognised text in the
    filename. Reads `p3_debug["ocr_crops"]` (`list[tuple[np.ndarray,
    str]]`)."""
    return _save_crop_text_images(p3_debug.get("ocr_crops") or [], target, page_index)
