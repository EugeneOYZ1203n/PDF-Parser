"""The high-level step functions `pipelines/current.py` calls. Each is a
thin adapter over one folder's own entrypoint -- the real work lives in
those folders, not here -- except the FAST/reassignment/drawing-merge logic
below, which is small enough to live here directly (see each function's own
docstring).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from rastervec.config import (
    FAST_DEBUG_IMAGE_SCALE,
    FAST_PAGE_RENDER_DPI,
    FAST_TILE_BLOCK_SIZE,
    FAST_TILE_CANDIDATE_MARGIN_FRAC,
    FAST_TILE_SCALE_FACTOR,
    FAST_VECTOR_KEEP_THRESHOLD,
    PADDLE_REASSIGN_MIN_COVERAGE,
    RADON_RENDER_PADDING_EXTRA_PT,
    RECLASSIFY_PASS_FRACTION,
)
from rastervec.commons.helpers.geometry import PDF_POINTS_PER_INCH, item_bbox, union_bbox
from rastervec.commons.helpers.geometry import bbox_area, bbox_intersection_area
from rastervec.commons.logging_setup import get_logger
from rastervec.commons.models import Page, Segment, Vector
from rastervec.P1_Reading_Native.native_text import extract_native_text as _extract_native_text
from rastervec.OCR.fast_detect import FastDetector
from rastervec.OCR.Paddle_OCR.ocr_backend import (
    ClusterDetection,
    PaddleDetectBackend,
    crop_rotated_detection,
)
from rastervec.OCR.radon import sweep_rotation
from rastervec.pipelines.result import FastPageResult
from rastervec.commons.renderer import render_page_paths
from rastervec.commons.renderer.stages import render_drawing  # noqa: F401 -- re-exported for callers
from rastervec.P1_Reading_Native.vector_extract import extract_vectors as _extract_vectors
from rastervec.P3_Vector_Parsing.FastIntoPaddle.similarity import SimilarityGroup, vector_similarity_group

log = get_logger("pipelines.steps")

# re-exported so `current.py` reads `extract_native_text(page)` etc.
extract_native_text = _extract_native_text
extract_vectors = _extract_vectors


def read_page(reader, page_index: int) -> Page:
    return reader.get_page(page_index)


# --------------------------------------------------------------------------
# FAST detection directly on every extracted Vector, independently, before
# any grouping/clustering exists at all. A Vector is scored over its own
# items' bboxes (`helpers.geometry.item_bbox` -- its actual line/fill
# geometry), not its aggregate `.bbox`, so a large near-empty bounding rect
# doesn't dilute the score with the heatmap value of its own empty interior.
# --------------------------------------------------------------------------
@dataclass
class FastFilterResult:
    passed: list[Vector]
    dropped: list[Vector]
    page_result: FastPageResult


def _sample_mask_items(mask, items: list[tuple], zoom: float) -> float:
    if mask is None:
        return 0.0
    mask_h, mask_w = mask.shape
    total_pixels = 0
    total_score = 0.0
    for item in items:
        x0, y0, x1, y1 = item_bbox(item)
        px0 = max(0, min(mask_w, int(x0 * zoom)))
        py0 = max(0, min(mask_h, int(y0 * zoom)))
        px1 = max(px0, min(mask_w, int(np.ceil(x1 * zoom))))
        py1 = max(py0, min(mask_h, int(np.ceil(y1 * zoom))))
        region = mask[py0:py1, px0:px1]
        if region.size:
            total_pixels += region.size
            total_score += float(region.sum())
    return total_score / total_pixels if total_pixels else 0.0


def _candidate_tile_bboxes(
    clusters: list[list[Vector]], *, zoom: float, margin: float,
) -> list[tuple[float, float, float, float]]:
    """Every cluster's page-space bbox, converted into `detect_tiled`'s
    tile-pixel space (page pt -> render px via `zoom`, the *combined*
    render dpi already baked in -- see `FAST_PAGE_RENDER_DPI`'s own note)
    and padded by `margin` px on every side, so a cluster sitting right at
    a tile boundary isn't dropped by an off-by-one intersection test -- the
    text-candidate set `detect_tiled` should only bother detecting tiles
    near."""
    boxes: list[tuple[float, float, float, float]] = []
    for cluster in clusters:
        if not cluster:
            continue
        x0, y0, x1, y1 = union_bbox([v.bbox for v in cluster])
        boxes.append((
            x0 * zoom - margin, y0 * zoom - margin,
            x1 * zoom + margin, y1 * zoom + margin,
        ))
    return boxes


def filter_vectors_fast(
    vectors: list[Vector],
    page: Page,
    *,
    enable_fast: bool = True,
    verbose: bool = False,
    compute=None,
    progress_counter=None,
) -> FastFilterResult:
    """Scores every extracted Vector, independently, against a whole-page
    FAST mask, sampled over each of its own items' bboxes (its real
    line/fill geometry) rather than its aggregate bbox, and keeps it only if
    that per-item coverage exceeds `FAST_VECTOR_KEEP_THRESHOLD`.
    `enable_fast=False` is a pass-through (every Vector passes). `compute`/
    `progress_counter` are forwarded to `detect_tiled`."""
    if not enable_fast:
        result = FastPageResult(None, None, None, {})
        return FastFilterResult(list(vectors), [], result)

    page_image = page_mask = None
    detect_seconds = None
    render_dpi = FAST_PAGE_RENDER_DPI * FAST_TILE_SCALE_FACTOR
    zoom = render_dpi / PDF_POINTS_PER_INCH
    if vectors:
        detector = FastDetector()
        page_image = render_page_paths(vectors, page.meta, render_dpi)
        candidate_bboxes = _candidate_tile_bboxes(
            [[v] for v in vectors], zoom=zoom,
            margin=FAST_TILE_BLOCK_SIZE * FAST_TILE_CANDIDATE_MARGIN_FRAC,
        )
        tile_report: list = [] if verbose else None
        start = time.perf_counter()
        try:
            page_mask = detector.detect_tiled(
                page_image, scale=1.0, desc="FAST text detection (per-vector)", compute=compute,
                candidate_bboxes=candidate_bboxes, progress_counter=progress_counter,
                tile_report=tile_report,
            )
        except FileNotFoundError as exc:
            log.warning("FAST detection skipped (keeping every vector): %s", exc)
            return filter_vectors_fast(
                vectors, page, enable_fast=False, verbose=verbose, compute=compute,
                progress_counter=progress_counter,
            )
        detect_seconds = time.perf_counter() - start

    vector_scores = [_sample_mask_items(page_mask, v.items, zoom) for v in vectors]

    passed: list[Vector] = []
    dropped: list[Vector] = []
    scores_by_vector: dict[int, float] = {}
    for i, v in enumerate(vectors):
        score = vector_scores[i]
        scores_by_vector[i] = score
        (passed if score > FAST_VECTOR_KEEP_THRESHOLD else dropped).append(v)

    skipped_tiles = all_tiles = tile_count = tile_seconds = None
    debug_image = None
    if verbose and vectors:
        def _to_page(rect):
            x0, y0, x1, y1 = rect
            return (x0 / zoom, y0 / zoom, x1 / zoom, y1 / zoom)

        tile_count = len(tile_report)
        all_tiles = [_to_page(e["rect_scaled"]) for e in tile_report]
        skipped_tiles = [_to_page(e["rect_scaled"]) for e in tile_report if not e["detected"]]
        tile_seconds = [e["seconds"] for e in tile_report if e.get("seconds") is not None]
        debug_size = (
            max(1, round(page_image.width * FAST_DEBUG_IMAGE_SCALE)),
            max(1, round(page_image.height * FAST_DEBUG_IMAGE_SCALE)),
        )
        debug_image = page_image.resize(debug_size)

    result = FastPageResult(
        debug_image, page_mask if verbose else None,
        detect_seconds, scores_by_vector,
        skipped_tiles=skipped_tiles, all_tiles=all_tiles,
        tile_count=tile_count, tile_seconds=tile_seconds,
        debug_image_scale=(FAST_DEBUG_IMAGE_SCALE if verbose else 1.0),
    )
    return FastFilterResult(passed, dropped, result)


# --------------------------------------------------------------------------
# Vector-level similarity grouping (P3_Vector_Parsing.FastIntoPaddle.similarity,
# imported above -- the old Vector_Similarity.similarity module this used to
# live in no longer exists), run before FAST -- every raw extracted Vector,
# independently, groups into a shape-similarity bucket regardless of FAST's
# own per-vector verdict.
# --------------------------------------------------------------------------
def similarity_group(vectors: list[Vector]) -> list[SimilarityGroup]:
    return vector_similarity_group(vectors)


# --------------------------------------------------------------------------
# Reclassify FAST's per-vector pass/fail up to a similarity-group consensus,
# one direction only: a shape FAST mostly accepted probably has a few
# members that only barely missed, so pull those up to pass too. There is
# no opposite "whole group fails" rule -- a member FAST passed keeps that
# verdict regardless of how the rest of its group scored.
# --------------------------------------------------------------------------
@dataclass
class ReclassifyResult:
    passed: list[Vector]
    dropped: list[Vector]
    fail_reclassified_pass: int  # members FAST dropped, reclassified to pass
    fail_count: int  # final dropped count (post-reclassification)
    pass_count: int  # final passed count (post-reclassification)


def reclassify_by_similarity(
    passed: list[Vector],
    dropped: list[Vector],
    groups: list[SimilarityGroup],
    *,
    pass_fraction: float = RECLASSIFY_PASS_FRACTION,
) -> ReclassifyResult:
    """Per similarity group: if under `pass_fraction` of its members failed
    FAST, every member passes; otherwise each member keeps FAST's own
    individual verdict. Groups are matched to `passed`/`dropped` by Vector
    identity (`id`), so `groups` must come from `similarity_group` run on
    the same Vector objects `filter_vectors_fast` scored."""
    dropped_ids = {id(v) for v in dropped}
    new_passed: list[Vector] = []
    new_dropped: list[Vector] = []
    fail_reclassified_pass = 0
    for g in groups:
        members = g.members
        if not members:
            continue
        failed = sum(1 for v in members if id(v) in dropped_ids)
        fail_frac = failed / len(members)
        if fail_frac < pass_fraction:
            for v in members:
                new_passed.append(v)
                if id(v) in dropped_ids:
                    fail_reclassified_pass += 1
        else:
            for v in members:
                (new_dropped if id(v) in dropped_ids else new_passed).append(v)
    return ReclassifyResult(
        passed=new_passed, dropped=new_dropped,
        fail_reclassified_pass=fail_reclassified_pass,
        fail_count=len(new_dropped), pass_count=len(new_passed),
    )


# --------------------------------------------------------------------------
# drawing output
# --------------------------------------------------------------------------
def build_drawing_output(*drop_lists: list[Vector]) -> list[Vector]:
    """Merges every rejected Vector -- FAST/reclassify/perimeter drops,
    PaddleOCR-reassignment drops -- into one flat, source-order list."""
    vectors: list[Vector] = []
    for drops in drop_lists:
        vectors.extend(drops)
    vectors.sort(key=lambda v: v.seqno)
    return vectors


# --------------------------------------------------------------------------
# Per-cluster PaddleOCR detect + overlap-based reassignment + rotation
# refinement/crop. See `pipelines/current.py` for the full step sequence
# these plug into.
# --------------------------------------------------------------------------
def cluster_render_padding(cluster: list[Vector]) -> float:
    """Half the cluster's own max stroke width, plus a small fixed margin
    (`RADON_RENDER_PADDING_EXTRA_PT`) so a stroke sitting at the exact edge
    of the cluster's bbox isn't clipped by the render frame -- shared by
    `detect_text_paddle_per_cluster` (which renders with it) and
    `rotate_paddle_detections` (which must map page-space points back into
    that exact same padded render's pixel space, or every coordinate
    drifts by `padding * zoom` px)."""
    return max((v.width or 0.0) for v in cluster) / 2.0 + RADON_RENDER_PADDING_EXTRA_PT


def detect_text_paddle_per_cluster(
    clusters: list[list[Vector]], *, dpi: int = 300,
) -> "list[ClusterDetection | None]":
    """One `PaddleDetectBackend.detect_on_cluster` call per cluster (never
    tiled, never whole-page -- see that method's own docstring), padded via
    `cluster_render_padding`. `None` for an empty cluster."""
    backend = PaddleDetectBackend()
    results: list[ClusterDetection | None] = []
    for cluster in clusters:
        if not cluster:
            results.append(None)
            continue
        results.append(
            backend.detect_on_cluster(cluster, dpi=dpi, padding=cluster_render_padding(cluster))
        )
    return results


@dataclass
class ReassignResult:
    text: "list[list[list[Vector]]]"  # per cluster, per detection
    drawing: list[Vector]


def reassign_by_overlap(
    clusters: list[list[Vector]],
    cluster_detections: "list[ClusterDetection | None]",
    *,
    min_coverage: float = PADDLE_REASSIGN_MIN_COVERAGE,
) -> ReassignResult:
    """For every Vector in every cluster, reassigns it to whichever of that
    cluster's own PaddleOCR detections covers the most of its own bbox area
    -- a real three-way outcome: a Vector whose best coverage is still
    under `min_coverage` (including a cluster with no detections at all)
    becomes a drawing vector instead. This is the "FAST said maybe,
    PaddleOCR didn't back it up" half of the pipeline's final drawing
    output (the other half is `filter_vectors_fast`'s own `dropped`) --
    combine both into `build_drawing_output`'s input. A detection that ends
    up with zero assigned vectors is implicitly dropped:
    `rotate_paddle_detections` skips it (nothing to crop)."""
    per_cluster_text: list[list[list[Vector]]] = []
    drawing: list[Vector] = []
    for cluster, cd in zip(clusters, cluster_detections):
        detections = cd.detections if cd is not None else []
        assigned: list[list[Vector]] = [[] for _ in detections]
        det_boxes = [d.bbox for d in detections]
        for v in cluster:
            v_area = bbox_area(v.bbox)
            best_i, best_coverage = -1, 0.0
            for i, box in enumerate(det_boxes):
                if v_area <= 0.0:
                    continue
                coverage = bbox_intersection_area(v.bbox, box) / v_area
                if coverage > best_coverage:
                    best_coverage, best_i = coverage, i
            if best_i >= 0 and best_coverage >= min_coverage:
                assigned[best_i].append(v)
            else:
                drawing.append(v)
        per_cluster_text.append(assigned)
    return ReassignResult(text=per_cluster_text, drawing=drawing)


def rotate_paddle_detections(
    clusters: list[list[Vector]],
    cluster_detections: "list[ClusterDetection | None]",
    reassigned: "list[list[list[Vector]]]",
    *,
    debug_out: "list | None" = None,
) -> list[Segment]:
    """For each cluster's own PaddleOCR detections with at least one
    assigned vector: refine PaddleOCR's coarse `rotation_deg` via
    `OCR.radon.sweep_rotation` (pure vector geometry, no pixels), then crop
    the already-rendered cluster image to that detection's own region at
    the refined angle (`ocr_backend.crop_rotated_detection`, which also
    applies the recognition-side white pad). One `Segment` per detection --
    no word/character splitting, "just rotated clusters". `debug_out`, when
    given, collects one dict per detection: `detection_bbox`,
    `resolved_theta`, `source_theta`."""
    segments: list[Segment] = []
    for cluster, cd, assigned in zip(clusters, cluster_detections, reassigned):
        if cd is None or not cd.detections:
            continue
        padding = cluster_render_padding(cluster)
        for detection, det_vectors in zip(cd.detections, assigned):
            if not det_vectors:
                continue
            sweep = sweep_rotation(det_vectors, center_theta_deg=detection.rotation_deg)
            crop = crop_rotated_detection(
                cd.image, cd.dpi, cluster, det_vectors, sweep.theta_deg,
                render_pad=(cd.pad_x_px, cd.pad_y_px), padding=padding,
            )
            if crop is None:
                continue
            segments.append(Segment(vectors=det_vectors, angle=sweep.theta_deg, image=crop))
            if debug_out is not None:
                debug_out.append({
                    "detection_bbox": detection.bbox,
                    "resolved_theta": sweep.theta_deg,
                    "source_theta": detection.rotation_deg,
                })
    return segments
