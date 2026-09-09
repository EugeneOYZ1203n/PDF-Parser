"""The high-level step functions the pipeline files call. Each is a thin
adapter over one folder's own entrypoint -- the real work lives in those
folders, not here -- except the segment-similarity/FAST/drawing-merge logic
below, which is small enough to live here directly (see each function's own
docstring).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from math import atan2, degrees

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
from rastervec.models import Page, Segment, SegmentMeta, UniqueSegment, Vector
from rastervec.native_text import extract_native_text as _extract_native_text
from rastervec.OCR.fast_detect import FastDetector
from rastervec.OCR.radon import segment_clusters
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
# Phase C.5: cluster-level dedup candidates (pre-Radon)
# --------------------------------------------------------------------------
def _cluster_angle(vectors: list[Vector]) -> float:
    """PCA principal-axis angle (degrees) over `vectors`' own point cloud:
    translate to the centroid, then the angle of the eigenvector of largest
    variance. Pure point-cloud math -- no rendering, no Radon transform --
    so it's cheap enough to run on every surviving classification cluster,
    unlike Radon's precise (but render-dependent) skew estimate. Ported from
    the pre-refactor `Vector_Classification/cluster_filters.py::
    group_similar_clusters`'s own `_normalized_point_cloud` (removed when
    similarity grouping moved to post-Radon `Segment`s; revived here now
    that grouping needs to run before Radon again). Sign convention matches
    `_normalize_segment`'s `-seg.angle` rotation and `helpers.geometry.
    transform_point`'s counter-clockwise matrix directly -- no adjustment
    needed to slot this into the same canonicalization code as a Radon
    angle."""
    pts = [pt for v in vectors for item in v.items for pt in item_points(item)]
    if not pts:
        return 0.0
    cx = sum(x for x, _y in pts) / len(pts)
    cy = sum(y for _x, y in pts) / len(pts)
    sxx = sum((x - cx) ** 2 for x, _y in pts)
    syy = sum((y - cy) ** 2 for _x, y in pts)
    sxy = sum((x - cx) * (y - cy) for x, y in pts)
    theta = 0.5 * atan2(2 * sxy, sxx - syy)
    return degrees(theta)


def build_cluster_candidates(clusters: list[list[Vector]]) -> list[Segment]:
    """One `Segment` per non-empty classification cluster, `angle` a PCA
    estimate (`_cluster_angle`) rather than Radon's precise skew -- the
    "similarity" step's input, feeding the same `group_similar_segments`/
    `detect_text_fast` used by the old word-level dedup, now one level up.
    `Segment.image` stays `None` here (see `models/segment.py`'s
    docstring)."""
    return [Segment(vectors=c, angle=_cluster_angle(c)) for c in clusters if c]


# --------------------------------------------------------------------------
# Phase E: similarity grouping, shared by both the cluster-level pass
# (pre-Radon, PCA-estimated angle -- see `build_cluster_candidates` above)
# and, historically, a word-level pass (post-Radon, exact angle) this
# function never actually needed to distinguish: it only ever reads
# `Segment.vectors`/`.angle`, whichever lifecycle produced them.
# --------------------------------------------------------------------------
def _normalize_segment(seg: Segment) -> tuple[list[Vector], tuple[float, float]]:
    """Canonical-frame vectors for one Segment: rotate every member Vector
    by `-seg.angle` about the world origin, then translate so the rotated
    bbox's own origin sits at (0, 0). Also returns that rotated bbox's
    origin (pre-translation) -- the piece `SegmentMeta.offset` is derived
    from (see `pipelines/_steps.py::detect_text_fast` and
    `models/segment.py::SegmentMeta`'s docstring for the exact inverse)."""
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
    genuine ambiguity for a PCA principal axis, which is always mod-180;
    also historically true of Radon's own 0-vs-180 skew ambiguity)."""
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
    similarity group, e.g. every occurrence of the same repeated dimension
    label. Greedy O(n^2): each segment joins the first existing group whose
    representative it's `_segments_similar` to, else starts a new group.
    Rotation is removed using each segment's own `angle` -- generic over
    whichever lifecycle produced `segments` (see `models/segment.py`'s
    docstring): a PCA estimate for the cluster-level pass this function is
    actually called with today, or Radon's exact skew for a word-level
    pass."""
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


# --------------------------------------------------------------------------
# Phase F: FAST detection on segments (real page position), all-must-pass.
# Called today with the cluster-level candidates `build_cluster_candidates`
# produces -- so a passing group's `UniqueSegment` is one whole
# representative cluster, and each `SegmentMeta` a real cluster occurrence
# -- but, like Phase E, generic over whatever `Segment`s it's given.
# --------------------------------------------------------------------------
@dataclass
class FastStepResult:
    uniques: list[UniqueSegment]
    metas: list[SegmentMeta]
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
    segments: list[Segment], *, zoom: float, tile_scale: float, margin: float,
) -> list[tuple[float, float, float, float]]:
    """Every segment's page-space bbox, converted into `detect_tiled`'s
    scaled tile-pixel space (page pt -> `FAST_PAGE_RENDER_DPI`-render px
    via `zoom` -> `* tile_scale`) and padded by `margin` px on every side,
    so a segment sitting right at a tile boundary isn't dropped by an
    off-by-one intersection test -- the text-candidate set `detect_tiled`
    should only bother detecting tiles near."""
    boxes: list[tuple[float, float, float, float]] = []
    for seg in segments:
        if not seg.vectors:
            continue
        x0, y0, x1, y1 = union_bbox([v.bbox for v in seg.vectors])
        scale = zoom * tile_scale
        boxes.append((
            x0 * scale - margin, y0 * scale - margin,
            x1 * scale + margin, y1 * scale + margin,
        ))
    return boxes


def _segment_meta(seg: Segment, unique_index: int, origin_after_rotation: tuple[float, float]) -> SegmentMeta:
    offset = transform_point(origin_after_rotation, offset=(0.0, 0.0), rotation_deg=seg.angle)
    first = seg.vectors[0]
    return SegmentMeta(
        unique_index=unique_index, offset=offset, rotation=seg.angle,
        page_index=first.page_index, seqno=first.seqno,
    )


def detect_text_fast(
    segments: list[Segment],
    groups: list[list[int]],
    page: Page,
    *,
    enable_fast: bool = True,
    verbose: bool = False,
    compute=None,
    progress_counter=None,
) -> FastStepResult:
    """Scores every segment (at its real page position) against a
    whole-page FAST mask; a group passes only if *every* member's score
    exceeds `FAST_COMBINED_KEEP_THRESHOLD`. A passing group materializes
    its first member's canonical (translation+rotation-normalized) vectors
    as one `UniqueSegment`, plus one `SegmentMeta` per member (including
    that first member itself) for later restoration. A failing group's
    real (un-normalized) member vectors are flattened into drawing output.
    `enable_fast=False` is a pass-through (every group passes).
    `compute`, when given a shared compute-pool proxy (see
    `Reader/Parallel`), is forwarded to `detect_tiled` so each tile's
    detection runs on that pool instead of locally. `progress_counter`,
    when given, is forwarded to `detect_tiled` too -- see that function's
    own docstring."""
    canon = [_normalize_segment(seg) for seg in segments]

    def _materialize(group: list[int]) -> tuple[UniqueSegment, list[SegmentMeta]]:
        rep_i = group[0]
        rep_vectors, _ = canon[rep_i]
        unique_index_placeholder = 0  # filled by caller after append
        metas = [
            _segment_meta(segments[i], unique_index_placeholder, canon[i][1])
            for i in group
        ]
        return UniqueSegment(vectors=rep_vectors), metas

    if not enable_fast:
        uniques: list[UniqueSegment] = []
        metas: list[SegmentMeta] = []
        for group in groups:
            unique, group_metas = _materialize(group)
            idx = len(uniques)
            uniques.append(unique)
            for m in group_metas:
                m.unique_index = idx
            metas.extend(group_metas)
        result = FastPageResult(None, None, None, {})
        return FastStepResult(uniques, metas, [], result)

    page_image = page_mask = None
    detect_seconds = None
    zoom = FAST_PAGE_RENDER_DPI / PDF_POINTS_PER_INCH
    all_vectors = [v for seg in segments for v in seg.vectors]
    if all_vectors:
        detector = FastDetector()
        page_image = render_page_paths(all_vectors, page.meta, FAST_PAGE_RENDER_DPI)
        candidate_bboxes = _candidate_tile_bboxes(
            segments, zoom=zoom, tile_scale=FAST_TILE_SCALE_FACTOR,
            margin=FAST_TILE_BLOCK_SIZE * FAST_TILE_CANDIDATE_MARGIN_FRAC,
        )
        start = time.perf_counter()
        try:
            page_mask = detector.detect_tiled(
                page_image, desc="FAST text detection", compute=compute,
                candidate_bboxes=candidate_bboxes, progress_counter=progress_counter,
            )
        except FileNotFoundError as exc:
            log.warning("FAST detection skipped (keeping every segment): %s", exc)
            return detect_text_fast(
                segments, groups, page, enable_fast=False, verbose=verbose, compute=compute,
                progress_counter=progress_counter,
            )
        detect_seconds = time.perf_counter() - start

    seg_scores = [_sample_mask(page_mask, seg.vectors, zoom) for seg in segments]

    uniques = []
    metas = []
    dropped_vectors: list[Vector] = []
    scores_by_group: dict[int, float] = {}
    for gi, group in enumerate(groups):
        member_scores = [seg_scores[i] for i in group]
        min_score = min(member_scores) if member_scores else 0.0
        scores_by_group[gi] = min_score
        if min_score > FAST_COMBINED_KEEP_THRESHOLD:
            unique, group_metas = _materialize(group)
            idx = len(uniques)
            uniques.append(unique)
            for m in group_metas:
                m.unique_index = idx
            metas.extend(group_metas)
        else:
            for i in group:
                dropped_vectors.extend(segments[i].vectors)

    result = FastPageResult(
        page_image if verbose else None,
        page_mask if verbose else None,
        detect_seconds, scores_by_group,
    )
    return FastStepResult(uniques, metas, dropped_vectors, result)


# --------------------------------------------------------------------------
# Phase G: Radon segmentation, now run only on the small set of elected
# representative clusters Phase F dedup down to (was: every surviving
# classification cluster, before dedup) -- see `OCR/radon.py`'s own
# docstring for the segmentation logic itself.
# --------------------------------------------------------------------------
def segment_unique_clusters(uniques: list[UniqueSegment]) -> list[list[Segment]]:
    """Radon-segments each representative cluster's own (already
    cluster-canonical-frame) vectors into word-level `Segment`s -- one
    `segment_clusters([u.vectors])` call per unique, so a non-representative
    cluster instance is never rendered or Radon-processed at all. Returns
    one inner list per input `unique`, same order, each entry that
    representative's own words (with `Segment.image` populated -- see
    `models/segment.py`'s docstring)."""
    return [segment_clusters([u.vectors]) for u in uniques]


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
