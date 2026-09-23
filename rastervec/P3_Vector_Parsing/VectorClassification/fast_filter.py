"""FAST-based text filtering: scores every classification cluster (at its
real page position) against a whole-page FAST mask, keeping only clusters
that look like text.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from rastervec.P3_Vector_Parsing.VectorClassification.config import FAST_COMBINED_KEEP_THRESHOLD, FAST_PAGE_RENDER_DPI
from rastervec.commons.helpers.geometry import PDF_POINTS_PER_INCH, union_bbox
from rastervec.commons.models import Vector
from rastervec.commons.renderer import render_page_paths


@dataclass
class FastPageResult:
    """`detect_text_fast`'s whole-page result. `scores` is keyed by a
    cluster's own index into that step's `clusters` input list. `all_tiles`/
    `skipped_tiles`/`tile_count`/`tile_seconds` are the real per-tile FAST
    detector geometry (`verbose`-only), mirroring `FastIntoPaddle/
    steps.py::FastPageResult`."""

    page_image: object
    page_mask: "np.ndarray | None"
    detect_seconds: float | None
    scores: dict
    skipped_tiles: list | None = None
    all_tiles: list | None = None
    tile_count: int | None = None
    tile_seconds: list | None = None


def _candidate_tile_bboxes(
    clusters: list[list[Vector]], *, zoom: float, margin: float,
) -> list[tuple[float, float, float, float]]:
    """Every cluster's page-space bbox, converted into `detect_tiled`'s
    tile-pixel space and padded by `margin` px on every side."""
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


@dataclass
class FastStepResult:
    passed: list[list[Vector]]
    dropped_vectors: list[Vector]
    page_result: FastPageResult


def _sample_mask(mask, vectors: list[Vector], zoom: float) -> float:
    if mask is None:
        return 0.0
    mask_h, mask_w = mask.shape
    total_pixels = 0
    total_score = 0.0
    for v in vectors:
        x0, y0, x1, y1 = v.bbox
        px0 = max(0, min(mask_w, int(x0 * zoom)))
        py0 = max(0, min(mask_h, int(y0 * zoom)))
        px1 = max(px0, min(mask_w, int(np.ceil(x1 * zoom))))
        py1 = max(py0, min(mask_h, int(np.ceil(y1 * zoom))))
        region = mask[py0:py1, px0:px1]
        if region.size:
            total_pixels += region.size
            total_score += float(region.sum())
    return total_score / total_pixels if total_pixels else 0.0


def detect_text_fast(
    clusters: list[list[Vector]],
    page,
    *,
    enable_fast: bool = True,
    verbose: bool = False,
    compute=None,
    progress_counter=None,
) -> FastStepResult:
    """Scores every classification cluster (at its real page position)
    against a whole-page FAST mask; a cluster passes on its own score alone."""
    from rastervec.P3_Vector_Parsing.VectorClassification.fast_detect import FastDetector
    from rastervec.P3_Vector_Parsing.VectorClassification.config import FAST_TILE_BLOCK_SIZE, FAST_TILE_CANDIDATE_MARGIN_FRAC, FAST_TILE_SCALE_FACTOR

    if not enable_fast:
        result = FastPageResult(None, None, None, {})
        return FastStepResult(list(clusters), [], result)

    page_image = page_mask = None
    detect_seconds = None
    render_dpi = FAST_PAGE_RENDER_DPI * FAST_TILE_SCALE_FACTOR
    zoom = render_dpi / PDF_POINTS_PER_INCH
    all_vectors = [v for cluster in clusters for v in cluster]
    if all_vectors:
        detector = FastDetector()
        page_image = render_page_paths(all_vectors, page.meta, render_dpi)
        candidate_bboxes = _candidate_tile_bboxes(
            clusters, zoom=zoom,
            margin=FAST_TILE_BLOCK_SIZE * FAST_TILE_CANDIDATE_MARGIN_FRAC,
        )
        tile_report: list = [] if verbose else None
        start = time.perf_counter()
        page_mask = detector.detect_tiled(
            page_image, scale=1.0, desc="FAST text detection", compute=compute,
            candidate_bboxes=candidate_bboxes, progress_counter=progress_counter,
            tile_report=tile_report,
        )
        detect_seconds = time.perf_counter() - start

    cluster_scores = [_sample_mask(page_mask, cluster, zoom) for cluster in clusters]

    passed: list[list[Vector]] = []
    dropped_vectors: list[Vector] = []
    scores_by_cluster: dict[int, float] = {}
    for i, cluster in enumerate(clusters):
        score = cluster_scores[i]
        scores_by_cluster[i] = score
        if score > FAST_COMBINED_KEEP_THRESHOLD:
            passed.append(cluster)
        else:
            dropped_vectors.extend(cluster)

    skipped_tiles = all_tiles = tile_count = tile_seconds = None
    if verbose and all_vectors:
        def _to_page(rect_scaled):
            x0, y0, x1, y1 = rect_scaled
            return (x0 / zoom, y0 / zoom, x1 / zoom, y1 / zoom)

        tile_count = len(tile_report)
        all_tiles = [_to_page(e["rect_scaled"]) for e in tile_report]
        skipped_tiles = [_to_page(e["rect_scaled"]) for e in tile_report if not e["detected"]]
        tile_seconds = [e["seconds"] for e in tile_report if e.get("seconds") is not None]

    result = FastPageResult(
        page_image if verbose else None, page_mask if verbose else None,
        detect_seconds, scores_by_cluster,
        skipped_tiles=skipped_tiles, all_tiles=all_tiles,
        tile_count=tile_count, tile_seconds=tile_seconds,
    )
    return FastStepResult(passed, dropped_vectors, result)
