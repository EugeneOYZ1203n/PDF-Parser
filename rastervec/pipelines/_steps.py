"""The high-level step functions the pipeline files call. Each is a thin
adapter over one folder's own entrypoint -- the real work lives in those
folders, not here -- except the segment-similarity/FAST/drawing-merge logic
below, which is small enough to live here directly (see each function's own
docstring).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from rastervec.config import (
    FAST_COMBINED_KEEP_THRESHOLD,
    FAST_PAGE_RENDER_DPI,
    FAST_TILE_BLOCK_SIZE,
    FAST_TILE_CANDIDATE_MARGIN_FRAC,
    FAST_TILE_SCALE_FACTOR,
    UNIQUE_CLUSTER_TOLERANCE,
)
from rastervec.helpers.geometry import (
    PDF_POINTS_PER_INCH,
    item_points,
    max_dimension,
    transform_point,
    transform_vector,
    union_bbox,
)
from rastervec.logging_setup import get_logger
from rastervec.models import Page, Segment, SegmentMeta, Vector
from rastervec.native_text import extract_native_text as _extract_native_text
from rastervec.OCR.fast_detect import FastDetector
from rastervec.OCR.radon import segment_clusters  # noqa: F401 -- re-exported for callers
from rastervec.pipelines.result import FastPageResult
from rastervec.renderer import render_page_paths
from rastervec.renderer.stages import render_drawing, render_similarity  # noqa: F401 -- re-exported for callers
from rastervec.Vector.vector import extract_vectors as _extract_vectors

log = get_logger("pipelines.steps")

# re-exported so the pipeline files read `extract_native_text(page)` etc.
extract_native_text = _extract_native_text
extract_vectors = _extract_vectors


def read_page(reader, page_index: int) -> Page:
    return reader.get_page(page_index)


# --------------------------------------------------------------------------
# Phase D: FAST detection directly on classification's surviving clusters
# (plain `list[list[Vector]]`, real page position) -- runs before Radon and
# before similarity grouping, so each cluster is scored and kept/dropped
# entirely on its own; there is no group to average or require every member
# of to pass.
# --------------------------------------------------------------------------
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


def _candidate_tile_bboxes(
    clusters: list[list[Vector]], *, zoom: float, tile_scale: float, margin: float,
) -> list[tuple[float, float, float, float]]:
    """Every cluster's page-space bbox, converted into `detect_tiled`'s
    scaled tile-pixel space (page pt -> `FAST_PAGE_RENDER_DPI`-render px
    via `zoom` -> `* tile_scale`) and padded by `margin` px on every side,
    so a cluster sitting right at a tile boundary isn't dropped by an
    off-by-one intersection test -- the text-candidate set `detect_tiled`
    should only bother detecting tiles near."""
    boxes: list[tuple[float, float, float, float]] = []
    for cluster in clusters:
        if not cluster:
            continue
        x0, y0, x1, y1 = union_bbox([v.bbox for v in cluster])
        scale = zoom * tile_scale
        boxes.append((
            x0 * scale - margin, y0 * scale - margin,
            x1 * scale + margin, y1 * scale + margin,
        ))
    return boxes


def detect_text_fast(
    clusters: list[list[Vector]],
    page: Page,
    *,
    enable_fast: bool = True,
    verbose: bool = False,
    compute=None,
    progress_counter=None,
) -> FastStepResult:
    """Scores every classification cluster (at its real page position)
    against a whole-page FAST mask; a cluster passes on its own score alone
    -- no grouping/dedup has happened yet at this point, so there is no
    "every member of a group must pass" check anymore. A passing cluster's
    real (unmodified) `Vector`s flow on to Radon segmentation; a failing
    cluster's vectors are flattened into drawing output. `enable_fast=False`
    is a pass-through (every cluster passes). `compute`, when given a shared
    compute-pool proxy (see `Reader/Parallel`), is forwarded to
    `detect_tiled` so each tile's detection runs on that pool instead of
    locally. `progress_counter`, when given, is forwarded to `detect_tiled`
    too -- see that function's own docstring."""
    if not enable_fast:
        result = FastPageResult(None, None, None, {})
        return FastStepResult(list(clusters), [], result)

    page_image = page_mask = None
    detect_seconds = None
    zoom = FAST_PAGE_RENDER_DPI / PDF_POINTS_PER_INCH
    all_vectors = [v for cluster in clusters for v in cluster]
    if all_vectors:
        detector = FastDetector()
        page_image = render_page_paths(all_vectors, page.meta, FAST_PAGE_RENDER_DPI)
        candidate_bboxes = _candidate_tile_bboxes(
            clusters, zoom=zoom, tile_scale=FAST_TILE_SCALE_FACTOR,
            margin=FAST_TILE_BLOCK_SIZE * FAST_TILE_CANDIDATE_MARGIN_FRAC,
        )
        start = time.perf_counter()
        try:
            page_mask = detector.detect_tiled(
                page_image, desc="FAST text detection", compute=compute,
                candidate_bboxes=candidate_bboxes, progress_counter=progress_counter,
            )
        except FileNotFoundError as exc:
            log.warning("FAST detection skipped (keeping every cluster): %s", exc)
            return detect_text_fast(
                clusters, page, enable_fast=False, verbose=verbose, compute=compute,
                progress_counter=progress_counter,
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
        page_image if verbose else None,
        page_mask if verbose else None,
        detect_seconds, scores_by_cluster,
    )
    return FastStepResult(passed, dropped_vectors, result)


# --------------------------------------------------------------------------
# Phase F: similarity grouping + representative election -- runs on Radon's
# own word-level `Segment`s (Phase E, `OCR/radon.py::segment_clusters`,
# called directly from `_common.py` on every FAST-surviving cluster -- see
# that module's own docstring), real page position, Radon's precise
# `angle`. This is the only `Segment` lifecycle in this pipeline now; there
# is no cluster-level PCA angle estimate anymore.
# --------------------------------------------------------------------------
def _normalize_segment(seg: Segment) -> tuple[list[Vector], tuple[float, float]]:
    """Canonical-frame vectors for one Segment: rotate every member Vector
    by `-seg.angle` about the world origin, then translate so the rotated
    bbox's own origin sits at (0, 0). Also returns that rotated bbox's
    origin (pre-translation) -- the piece `SegmentMeta.offset` is derived
    from (see `elect_unique_segments` and `models/segment.py::
    SegmentMeta`'s docstring for the exact inverse)."""
    rotated = [transform_vector(v, offset=(0.0, 0.0), rotation_deg=-seg.angle) for v in seg.vectors]
    ox, oy, _x1, _y1 = union_bbox([v.bbox for v in rotated])
    canonical = [transform_vector(v, offset=(-ox, -oy), rotation_deg=0.0) for v in rotated]
    return canonical, (ox, oy)


def _segment_signature(seg: Segment) -> tuple:
    return tuple(sorted(item[0] for v in seg.vectors for item in v.items))


def _point_cloud(vectors: list[Vector]) -> list[tuple[float, float]]:
    return sorted(pt for v in vectors for item in v.items for pt in item_points(item))


def _clouds_close(a: list[tuple[float, float]], b: list[tuple[float, float]], tolerance: float) -> bool:
    return len(a) == len(b) and all(
        ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= tolerance
        for (ax, ay), (bx, by) in zip(a, b)
    )


def _segments_similar(a: Segment, b: Segment, canon_a, canon_b, tolerance: float) -> bool:
    """True if `a` and `b` are the same shape (same multiset of item kinds)
    and, once both are normalized to their own canonical (translation +
    `angle`-rotation) frame, every corresponding point pair sits within
    `tolerance * max(scale_a, scale_b)` of each other -- checked against
    both the direct normalization and its 180-degree-flipped mirror (a
    genuine ambiguity Radon's own skew estimate cannot resolve either,
    since a baseline is a line, not an arrow)."""
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
    """Groups `segments` by whole-page, translation+rotation-tolerant shape
    equivalence -- each inner list is the indices (into `segments`) of one
    similarity group, e.g. every occurrence of the same repeated word.
    Greedy O(n^2): each segment joins the first existing group whose
    representative it's `_segments_similar` to, else starts a new group.
    Rotation is removed using each segment's own (Radon-precise) `angle`."""
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


def _segment_meta(seg: Segment, unique_index: int, origin_after_rotation: tuple[float, float]) -> SegmentMeta:
    offset = transform_point(origin_after_rotation, offset=(0.0, 0.0), rotation_deg=seg.angle)
    first = seg.vectors[0]
    return SegmentMeta(
        unique_index=unique_index, offset=offset, rotation=seg.angle,
        page_index=first.page_index, seqno=first.seqno,
    )


def elect_unique_segments(
    segments: list[Segment], groups: list[list[int]],
) -> tuple[list[Segment], list[SegmentMeta]]:
    """Elects `group[0]` as each similarity group's representative and
    canonicalizes it (translation+rotation-normalized via
    `_normalize_segment`) into a zero-angle `Segment` -- `vectors` in
    canonical frame, `image` carried over unchanged from the real
    representative (Radon already captured it upright; no re-render). One
    `SegmentMeta` is built per group member (including the representative's
    own occurrence), recording how to transform that occurrence's
    (eventual, canonical-frame) OCR `Text` back onto its own real page
    position (see `pipelines/sub_pipelines/ocr.py::restore_word_texts`)."""
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


# --------------------------------------------------------------------------
# drawing output
# --------------------------------------------------------------------------
def build_drawing_output(
    classification_dropped: list[Vector],
    fast_dropped: list[Vector],
) -> list[Vector]:
    """Merges every rejected Vector -- classification drops, FAST drops --
    into one flat, source-order list. Nothing is reassembled: a `Vector`
    was never decomposed, so there's no per-drawing regrouping left to do,
    unlike the pre-refactor `build_drawing_vectors`."""
    vectors = list(classification_dropped) + list(fast_dropped)
    vectors.sort(key=lambda v: v.seqno)
    return vectors
