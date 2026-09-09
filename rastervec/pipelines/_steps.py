"""The high-level step functions the pipeline files call. Each is a thin
adapter over one folder's own entrypoint -- the real work lives in those
folders, not here -- except the segment-similarity/FAST/drawing-merge logic
below, which is small enough to live here directly (see each function's own
docstring).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from rastervec.config import FAST_COMBINED_KEEP_THRESHOLD, FAST_PAGE_RENDER_DPI, UNIQUE_CLUSTER_TOLERANCE
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
from rastervec.pipelines.result import FastPageResult
from rastervec.renderer import render_page_paths, render_reconstructed_page
from rastervec.Vector.vector import extract_vectors as _extract_vectors

if TYPE_CHECKING:
    from rastervec.pipelines.result import PipelineResult
    from rastervec.renderer.notebook import RenderResult

log = get_logger("pipelines.steps")

# re-exported so the pipeline files read `extract_native_text(page)` etc.
extract_native_text = _extract_native_text
extract_vectors = _extract_vectors


def read_page(reader, page_index: int) -> Page:
    return reader.get_page(page_index)


# --------------------------------------------------------------------------
# Phase E: segment similarity grouping (post-Radon -- rotation-exact via
# each Segment's own known precise angle, instead of a PCA-based rotation
# search over pre-Radon clusters)
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
    exact-Radon-angle-rotation) frame, every corresponding point pair sits
    within `tolerance * max(scale_a, scale_b)` of each other -- checked
    against both the direct normalization and its 180-degree-flipped
    mirror (Radon's own 0-vs-180 ambiguity can leave two otherwise-identical
    segments canonicalized a half-turn apart)."""
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
    """Groups `segments` by whole-page, translation+rotation-exact shape
    equivalence -- each inner list is the indices (into `segments`) of one
    similarity group, e.g. every occurrence of the same repeated dimension
    label. Greedy O(n^2): each segment joins the first existing group whose
    representative it's `_segments_similar` to, else starts a new group.
    Rotation is removed using each segment's own precise Radon `angle`
    (never a search), so this is cheaper and more accurate than the old
    pre-Radon PCA-based cluster-similarity check it replaces."""
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
# Phase F: FAST detection on segments (real page position), all-must-pass
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
    detection runs on that pool instead of locally."""
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
    all_vectors = [v for seg in segments for v in seg.vectors]
    if all_vectors:
        detector = FastDetector()
        page_image = render_page_paths(all_vectors, page.meta, FAST_PAGE_RENDER_DPI)
        start = time.perf_counter()
        try:
            page_mask = detector.detect_tiled(
                page_image, desc="FAST text detection", compute=compute,
            )
        except FileNotFoundError as exc:
            log.warning("FAST detection skipped (keeping every segment): %s", exc)
            return detect_text_fast(
                segments, groups, page, enable_fast=False, verbose=verbose, compute=compute,
            )
        detect_seconds = time.perf_counter() - start

    zoom = FAST_PAGE_RENDER_DPI / PDF_POINTS_PER_INCH
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


def render_similarity(res: "PipelineResult") -> "RenderResult":
    """Notebook visualization for the similarity-grouping step: every
    segment's bbox, plus a note on how much the grouping is expected to
    save Phase G's OCR call count (each group beyond size 1 means every
    extra member reuses one render+recognition instead of paying for its
    own)."""
    from rastervec.renderer.notebook import RenderResult

    segments = res.segments or []
    groups = res.similarity_groups or []
    dup_groups = [g for g in groups if len(g) > 1]
    saved = sum(len(g) - 1 for g in dup_groups)
    return RenderResult(
        categories=[{
            "name": f"segments ({len(segments)}) in {len(groups)} similarity group(s)",
            "bboxes": [union_bbox([v.bbox for v in seg.vectors]) for seg in segments if seg.vectors],
        }],
        note=(
            f"{len(segments)} segment(s) -> {len(groups)} group(s) "
            f"({len(dup_groups)} with >1 member, dedup saves {saved} OCR call(s) if all pass FAST)"
        ),
    )


def render_drawing(res: "PipelineResult", *, zoom: float = 1.0) -> "RenderResult":
    """Notebook visualization for the drawing-vectors output: dashed vs
    solid bboxes, plus a full page reconstruction."""
    from rastervec.helpers.geometry import is_dashed
    from rastervec.renderer.notebook import DEFAULT_PATH_COLOR, RenderResult

    dv = res.vectors or []
    dashed = [d for d in dv if is_dashed(d.dashes)]
    solid = [d for d in dv if not is_dashed(d.dashes)]
    recon = render_reconstructed_page(res.page.meta, drawing_vectors=dv, zoom=zoom)
    return RenderResult(
        categories=[
            {"name": f"dashed ({len(dashed)})", "color": DEFAULT_PATH_COLOR, "bboxes": [d.bbox for d in dashed]},
            {"name": f"solid ({len(solid)})", "color": DEFAULT_PATH_COLOR, "bboxes": [d.bbox for d in solid]},
            {"name": "full reconstruction", "isolated": recon, "overlay": recon},
        ],
        note=f"{len(dv)} drawing vector(s)",
    )
