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
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


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


def _save_fastintopaddle_detect_images(p3_debug: dict, folder: Path, page_index: int) -> int:
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

    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, cd in enumerate(cluster_detections):
        if cd is None or cd.image is None:
            continue
        cluster = clusters[i] if i < len(clusters) else []
        padding = cluster_render_padding(cluster) if cluster else 0.0
        img = Image.fromarray(np.asarray(cd.image)[..., ::-1])  # BGR -> RGB
        boxes = []
        for det in cd.detections:
            (px0, py0), (px1, py1) = page_points_to_pixel(
                cluster, cd.dpi, [(det.bbox[0], det.bbox[1]), (det.bbox[2], det.bbox[3])],
                padding=padding,
            )
            boxes.append((px0 + cd.pad_x_px, py0 + cd.pad_y_px, px1 + cd.pad_x_px, py1 + cd.pad_y_px))
        img = _draw_boxes(img, boxes, outline=(220, 30, 30))
        img.save(folder / f"p{page_index}_cluster_{i:03d}.png")
        n += 1
    return n


def _save_segment_recog_images(segs: list, texts: list, folder: Path, page_index: int) -> int:
    """Shared body for FastIntoPaddle's/VectorClassification's own
    recog-image dumpers -- both hand PaddleOCR recognition a list of
    `commons.models.Segment` (each already carrying its own crop in
    `.image`) 1:1-aligned with a `list[Text]` of what got recognised."""
    if not segs:
        return 0
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, seg in enumerate(segs):
        if seg.image is None:
            continue
        img = Image.fromarray(np.asarray(seg.image))
        rec = texts[i].text if i < len(texts) else ""
        img.save(folder / f"p{page_index}_word_{i:03d}__{_safe_slug(rec)}.png")
        n += 1
    return n


def _save_fastintopaddle_recog_images(p3_debug: dict, folder: Path, page_index: int) -> int:
    """One PNG per rotated detection crop -- the exact crop handed to
    PaddleOCR recognition (`Segment.image`), recognised text in the
    filename. Reads `p3_debug["segments"]` + `p3_debug["texts"]`."""
    return _save_segment_recog_images(
        p3_debug.get("segments") or [], p3_debug.get("texts") or [], folder, page_index,
    )


def _save_crop_text_images(crops: list, folder: Path, page_index: int) -> int:
    """Shared body for LegacyRecreation's/VectorClassification's own
    recog-image dumpers -- both hand PaddleOCR recognition a list of
    `(crop, text)` pairs (one per PaddleOCR-detected quad within a
    rendered cluster/word-group), reads `p3_debug["ocr_crops"]`."""
    if not crops:
        return 0
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, (crop, rec) in enumerate(crops):
        img = Image.fromarray(np.asarray(crop))
        img.save(folder / f"p{page_index}_word_{i:03d}__{_safe_slug(rec)}.png")
        n += 1
    return n


def _save_vectorclassification_recog_images(p3_debug: dict, folder: Path, page_index: int) -> int:
    """One PNG per PaddleOCR-recognised crop -- exactly what the recognizer
    saw, recognised text in the filename. Reads `p3_debug["ocr_crops"]`
    (`list[tuple[np.ndarray, str]]`)."""
    return _save_crop_text_images(p3_debug.get("ocr_crops") or [], folder, page_index)


def _save_vectorclassification_detect_images(p3_debug: dict, folder: Path, page_index: int) -> int:
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
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, (bgr, quads) in enumerate(entries):
        img = Image.fromarray(np.asarray(bgr)[..., ::-1])  # BGR -> RGB
        draw = ImageDraw.Draw(img)
        for quad in quads or []:
            pts = [(float(x), float(y)) for x, y in quad]
            draw.polygon(pts, outline=(220, 30, 30), width=2)
        img.save(folder / f"p{page_index}_cluster_{i:03d}.png")
        n += 1
    return n


def _save_legacyrecreation_ocr_images(p3_debug: dict, folder: Path, page_index: int) -> int:
    """One PNG per word group's own padded/DPI-boosted OCR render -- exactly
    what PaddleOCR's (recognition-only) engine saw, recognised text in the
    filename. Reads `p3_debug["ocr_crops"]` (`list[tuple[np.ndarray,
    str]]`)."""
    crops = p3_debug.get("ocr_crops") or []
    if not crops:
        return 0
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, (crop, rec) in enumerate(crops):
        img = Image.fromarray(np.asarray(crop))
        img.save(folder / f"p{page_index}_word_{i:03d}__{_safe_slug(rec)}.png")
        n += 1
    return n
