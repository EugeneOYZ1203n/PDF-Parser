"""Archived (2026-09) -- the old Vector_Classification+pixel-Radon
`current` pipeline's own step functions, cut out of the live
`rastervec/pipelines/_steps.py` when that pipeline was retired in favor of
the FAST-filter/PaddleOCR-detect pipeline (see `archive/rastervec/
pipelines/_common.py`, which called these). Frozen storage -- not
maintained, not guaranteed to still import cleanly against the live
`rastervec` tree (e.g. `FastPageResult` may have moved).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from rastervec.P3_Vector_Parsing.VectorClassification.config import FAST_COMBINED_KEEP_THRESHOLD, FAST_PAGE_RENDER_DPI, UNIQUE_CLUSTER_TOLERANCE
from rastervec.commons.helpers.geometry import (
    PDF_POINTS_PER_INCH,
    item_points,
    max_dimension,
    transform_point,
    transform_vector,
    union_bbox,
)
from rastervec.commons.models import Segment, SegmentMeta, Vector
from rastervec.commons.renderer import render_page_paths


@dataclass
class FastPageResult:
    """`detect_text_fast`'s whole-page result. `scores` is keyed by a
    cluster's own index into that step's `clusters` input list."""

    page_image: object
    page_mask: "np.ndarray | None"
    detect_seconds: float | None
    scores: dict


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

    result = FastPageResult(
        page_image if verbose else None, page_mask if verbose else None,
        detect_seconds, scores_by_cluster,
    )
    return FastStepResult(passed, dropped_vectors, result)


# --------------------------------------------------------------------------
# similarity grouping + representative election -- ran on Radon's own
# word-level `Segment`s (`archive/rastervec/OCR/radon.py::segment_clusters`).
# --------------------------------------------------------------------------
def _normalize_segment(seg: Segment) -> tuple[list[Vector], tuple[float, float]]:
    rotated = [transform_vector(v, offset=(0.0, 0.0), rotation_deg=-seg.angle) for v in seg.vectors]
    ox, oy, _x1, _y1 = union_bbox([v.bbox for v in rotated])
    canonical = [transform_vector(v, offset=(-ox, -oy), rotation_deg=0.0) for v in rotated]
    return canonical, (ox, oy)


def _segment_signature(seg: Segment) -> tuple:
    return tuple(sorted(item[0] for v in seg.vectors for item in v.items))


def _point_cloud(vectors: list[Vector]) -> list[tuple[float, float]]:
    return sorted(pt for v in vectors for item in v.items for pt in item_points(item))


def _clouds_close(a, b, tolerance: float) -> bool:
    return len(a) == len(b) and all(
        ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= tolerance
        for (ax, ay), (bx, by) in zip(a, b)
    )


def _segments_similar(a: Segment, b: Segment, canon_a, canon_b, tolerance: float) -> bool:
    if _segment_signature(a) != _segment_signature(b):
        return False
    vecs_a, _ = canon_a
    vecs_b, _ = canon_b
    pts_a, pts_b = _point_cloud(vecs_a), _point_cloud(vecs_b)
    if len(pts_a) != len(pts_b):
        return False
    scale_a = max(max_dimension(union_bbox([v.bbox for v in vecs_a])), 1e-6)
    scale_b = max(max_dimension(union_bbox([v.bbox for v in vecs_b])), 1e-6)
    tol = tolerance * max(scale_a, scale_b)
    if _clouds_close(pts_a, pts_b, tol):
        return True
    flipped_b = sorted((-x, -y) for x, y in pts_b)
    return _clouds_close(pts_a, flipped_b, tol)


def group_similar_segments(
    segments: list[Segment], tolerance: float = UNIQUE_CLUSTER_TOLERANCE,
) -> list[list[int]]:
    canon = [_normalize_segment(seg) for seg in segments]
    groups: list[list[int]] = []
    reps: list[int] = []
    for i, seg in enumerate(segments):
        matched = False
        for gi, rep_i in enumerate(reps):
            if _segments_similar(seg, segments[rep_i], canon[i], canon[rep_i], tolerance):
                groups[gi].append(i)
                matched = True
                break
        if not matched:
            groups.append([i])
            reps.append(i)
    return groups


def _segment_meta(seg: Segment, unique_index: int, origin_after_rotation) -> SegmentMeta:
    offset = transform_point(origin_after_rotation, offset=(0.0, 0.0), rotation_deg=seg.angle)
    first = seg.vectors[0]
    return SegmentMeta(
        unique_index=unique_index, offset=offset, rotation=seg.angle,
        page_index=first.page_index, seqno=first.seqno,
    )


def elect_unique_segments(
    segments: list[Segment], groups: list[list[int]],
) -> tuple[list[Segment], list[SegmentMeta]]:
    canon = [_normalize_segment(seg) for seg in segments]
    uniques: list[Segment] = []
    metas: list[SegmentMeta] = []
    for group in groups:
        rep_i = group[0]
        rep_vectors, _ = canon[rep_i]
        idx = len(uniques)
        uniques.append(Segment(vectors=rep_vectors, angle=0.0, image=segments[rep_i].image))
        for i in group:
            metas.append(_segment_meta(segments[i], idx, canon[i][1]))
    return uniques, metas
