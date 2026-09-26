"""FAST-based text filtering: scores every member vector of every
classification cluster (at its real page position) individually against a
whole-page FAST mask, keeping a whole cluster (every one of its vectors) if
any single member scores above threshold.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from rastervec.P3_Vector_Parsing.VectorClassification.config import FAST_PAGE_RENDER_DPI, FAST_VECTOR_ANY_THRESHOLD
from rastervec.commons.helpers.geometry import PDF_POINTS_PER_INCH, union_bbox
from rastervec.commons.models import Vector
from rastervec.commons.renderer import render_page_paths


@dataclass
class FastPageResult:
    """`detect_text_fast`'s whole-page result. `scores` is keyed by a
    cluster's own index into that step's `clusters` input list, each value
    the list of that cluster's own member vectors' individual FAST scores
    (a cluster passes if any one of them exceeds `FAST_VECTOR_ANY_THRESHOLD`
    -- there is no single whole-cluster score any more). `all_tiles`/
    `skipped_tiles`/`tile_count`/`tile_seconds` are the real per-tile FAST
    detector geometry (`verbose`-only). `n_clusters`/`n_passed_clusters` are always
    populated (cheap ints, not verbose-gated) -- the "clusters dropped by
    FAST" benchmark stat (`scripts/generate_pipeline_report.py`) reads them
    straight off this dataclass."""

    page_image: object
    page_mask: "np.ndarray | None"
    detect_seconds: float | None
    scores: dict
    skipped_tiles: list | None = None
    all_tiles: list | None = None
    tile_count: int | None = None
    tile_seconds: list | None = None
    n_clusters: int = 0
    n_passed_clusters: int = 0


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


def _build_integral_image(mask: "np.ndarray | None") -> "np.ndarray | None":
    """Summed-area table of `mask` (zero-padded one row/col on the top-left,
    so a bbox starting at pixel 0 needs no special-casing in
    `_vector_mask_scores`'s corner-sum lookup), or `None` if there's no mask
    to score against. Built once per page and reused for every vector's own
    O(1) region-sum lookup, instead of each vector doing its own
    numpy-slice-and-`.sum()` call."""
    if mask is None:
        return None
    padded = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.float64)
    padded[1:, 1:] = mask
    return np.cumsum(np.cumsum(padded, axis=0), axis=1)


def _vector_mask_scores(
    integral: "np.ndarray | None", mask_shape: "tuple[int, int] | None",
    vectors: list[Vector], zoom: float,
) -> list[float]:
    """Every `vectors` entry's own mean FAST-mask coverage inside its
    page-space bbox (scaled to mask-pixel space and clipped to the mask's
    own bounds, same convention the old per-vector `_sample_mask` used),
    computed in a handful of vectorized numpy ops over `integral` (see
    `_build_integral_image`) rather than one Python call + array slice per
    vector -- the whole page's worth of vectors (across every cluster) is
    scored in one shot by `detect_text_fast`."""
    if integral is None or not vectors or mask_shape is None:
        return [0.0] * len(vectors)
    mask_h, mask_w = mask_shape
    bboxes = np.array([v.bbox for v in vectors], dtype=np.float64)
    x0, y0, x1, y1 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    px0 = np.clip((x0 * zoom).astype(np.int64), 0, mask_w)
    py0 = np.clip((y0 * zoom).astype(np.int64), 0, mask_h)
    px1 = np.maximum(px0, np.clip(np.ceil(x1 * zoom).astype(np.int64), 0, mask_w))
    py1 = np.maximum(py0, np.clip(np.ceil(y1 * zoom).astype(np.int64), 0, mask_h))

    # Standard summed-area-table corner lookup: sum over mask[y0:y1, x0:x1]
    # == integral[y1,x1] - integral[y0,x1] - integral[y1,x0] + integral[y0,x0]
    # (no extra +1 offset needed -- `integral` is already padded so px0/py0
    # index the row/col immediately before the region starts).
    total_score = (
        integral[py1, px1] - integral[py0, px1] - integral[py1, px0] + integral[py0, px0]
    )
    total_pixels = (py1 - py0) * (px1 - px0)
    scores = np.where(total_pixels > 0, total_score / np.maximum(total_pixels, 1), 0.0)
    return scores.tolist()


def detect_text_fast(
    clusters: list[list[Vector]],
    page,
    *,
    enable_fast: bool = True,
    verbose: bool = False,
    compute=None,
    progress_counter=None,
) -> FastStepResult:
    """Scores every member vector of every classification cluster (at its
    real page position) individually against a whole-page FAST mask; a
    cluster passes (keeping all its vectors) if ANY one of them scores
    above `FAST_VECTOR_ANY_THRESHOLD` -- not a whole-cluster average."""
    from rastervec.P3_Vector_Parsing.VectorClassification.fast_detect import FastDetector
    from rastervec.P3_Vector_Parsing.VectorClassification.config import FAST_TILE_BLOCK_SIZE, FAST_TILE_CANDIDATE_MARGIN_FRAC, FAST_TILE_SCALE_FACTOR

    if not enable_fast:
        result = FastPageResult(
            None, None, None, {}, n_clusters=len(clusters), n_passed_clusters=len(clusters),
        )
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

    # A cluster passes on ANY single member vector's own score alone -- not
    # a whole-cluster average -- so one strong ink-looking vector saves the
    # whole cluster (all its vectors, including any weaker-scoring ones)
    # from being dropped to drawing output. Every vector across every
    # cluster is scored in one vectorized pass (see `_vector_mask_scores`),
    # then re-split back into the per-cluster shape the rest of this
    # function expects.
    integral = _build_integral_image(page_mask)
    flat_scores = iter(_vector_mask_scores(
        integral, page_mask.shape if page_mask is not None else None, all_vectors, zoom,
    ))
    vector_scores_by_cluster = [
        [next(flat_scores) for _v in cluster] for cluster in clusters
    ]

    passed: list[list[Vector]] = []
    dropped_vectors: list[Vector] = []
    scores_by_cluster: dict[int, list[float]] = {}
    for i, cluster in enumerate(clusters):
        scores = vector_scores_by_cluster[i]
        scores_by_cluster[i] = scores
        if any(s > FAST_VECTOR_ANY_THRESHOLD for s in scores):
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
        n_clusters=len(clusters), n_passed_clusters=len(passed),
    )
    return FastStepResult(passed, dropped_vectors, result)
