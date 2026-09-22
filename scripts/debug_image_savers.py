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
internal pipelines genuinely differ -- VectorClassification has no
PaddleOCR *detect* stage at all, LegacyRecreation has neither a detect nor
a FAST stage.
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


def _save_vectorclassification_recog_images(p3_debug: dict, folder: Path, page_index: int) -> int:
    """One PNG per *deduped* representative segment -- exactly what
    PaddleOCR's recognizer saw for this backend (after
    `elect_unique_segments`'s dedup, not the pre-dedup `word_segments`),
    recognised text in the filename. Reads `p3_debug["ocr_uniques"]` +
    `p3_debug["ocr_unique_texts"]`."""
    return _save_segment_recog_images(
        p3_debug.get("ocr_uniques") or [], p3_debug.get("ocr_unique_texts") or [], folder, page_index,
    )


def _save_fastintopaddle_tile_images(p3_debug: dict, folder: Path, page_index: int) -> int:
    """One PNG per FAST tile -- the exact crop of the whole-page FAST render
    (`FastPageResult.page_image`, which is `debug_image_scale`-downsampled
    from the full-res image `FastDetector.detect`/`_detect_job` actually
    saw for that tile) matching that tile's page-space rect (`all_tiles`).
    Reads `p3_debug["fast"]`."""
    fr = p3_debug.get("fast")
    tiles = getattr(fr, "all_tiles", None) if fr is not None else None
    if fr is None or fr.page_image is None or not tiles:
        return 0
    from rastervec.P3_Vector_Parsing.FastIntoPaddle.config import FAST_PAGE_RENDER_DPI, FAST_TILE_SCALE_FACTOR
    from rastervec.commons.helpers.geometry import PDF_POINTS_PER_INCH

    debug_scale = getattr(fr, "debug_image_scale", 1.0)
    zoom = (FAST_PAGE_RENDER_DPI * FAST_TILE_SCALE_FACTOR) / PDF_POINTS_PER_INCH * debug_scale
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, rect in enumerate(tiles):
        x0, y0, x1, y1 = (c * zoom for c in rect)
        crop = fr.page_image.crop((int(x0), int(y0), int(x1), int(y1)))
        crop.save(folder / f"p{page_index}_tile_{i:03d}.png")
        n += 1
    return n


def _save_vectorclassification_cluster_images(p3_debug: dict, folder: Path, page_index: int) -> int:
    """VectorClassification has no per-tile FAST detector -- one whole-page
    mask is scored per surviving classification cluster instead (see
    `VectorClassification/fast_filter.py::detect_text_fast`). One PNG per
    *passed* cluster, cropped from that same whole-page render
    (`FastPageResult.page_image`) at the cluster's own bbox -- not a
    literal detector tile grid, but the closest equivalent: exactly the
    pixels that cluster's FAST score was sampled from. Reads
    `p3_debug["fast_result"]` + `p3_debug["fast_passed"]`."""
    fr = p3_debug.get("fast_result")
    clusters = p3_debug.get("fast_passed") or []
    if fr is None or fr.page_image is None or not clusters:
        return 0
    from rastervec.P3_Vector_Parsing.VectorClassification.config import FAST_PAGE_RENDER_DPI, FAST_TILE_SCALE_FACTOR
    from rastervec.commons.helpers.geometry import PDF_POINTS_PER_INCH, union_bbox

    zoom = (FAST_PAGE_RENDER_DPI * FAST_TILE_SCALE_FACTOR) / PDF_POINTS_PER_INCH
    folder.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, cluster in enumerate(clusters):
        if not cluster:
            continue
        x0, y0, x1, y1 = union_bbox([v.bbox for v in cluster])
        px0, py0, px1, py1 = x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom
        crop = fr.page_image.crop((int(px0), int(py0), int(px1) + 1, int(py1) + 1))
        crop.save(folder / f"p{page_index}_cluster_{i:03d}.png")
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
