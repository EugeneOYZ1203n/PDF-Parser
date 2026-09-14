"""FastIntoPaddle's own step functions -- similarity grouping through
rotation refinement. Self-contained: imports only this folder's own
duplicated fast_detect.py/paddle_engine.py/radon.py/layer_color_separation.py
and commons/. Ported from the old rastervec/pipelines/_steps.py +
pipelines/current.py before those were retired in favor of this folder.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from rastervec.P3_Vector_Parsing.FastIntoPaddle.config import (
    CLUSTER_TOLERANCE_MAX,
    CLUSTER_TOLERANCE_SCALE,
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
from rastervec.commons.helpers.geometry import (
    PDF_POINTS_PER_INCH,
    bbox_area,
    bbox_intersection_area,
    item_bbox,
    rect_gap,
    union_bbox,
)
from rastervec.commons.logging_setup import get_logger
from rastervec.commons.models import Page, Segment, Vector
from rastervec.commons.renderer import render_page_paths
from rastervec.P3_Vector_Parsing.FastIntoPaddle.fast_detect import FastDetector
from rastervec.P3_Vector_Parsing.FastIntoPaddle.layer_color_separation import (
    separate_by_color,
    separate_by_layer,
    separate_by_width,
)
from rastervec.P3_Vector_Parsing.FastIntoPaddle.paddle_engine import (
    ClusterDetection,
    PaddleDetectBackend,
    crop_rotated_detection,
)
from rastervec.P3_Vector_Parsing.FastIntoPaddle.radon import sweep_rotation
from rastervec.P3_Vector_Parsing.FastIntoPaddle.similarity import SimilarityGroup, vector_similarity_group

log = get_logger("P3.FastIntoPaddle.steps")


@dataclass
class FastPageResult:
    """`filter_vectors_fast`'s whole-page result. `scores` is keyed by a
    Vector's own index into that step's `vectors` input list."""

    page_image: object
    page_mask: "np.ndarray | None"
    detect_seconds: float | None
    scores: dict
    skipped_tiles: list | None = None
    all_tiles: list | None = None
    tile_count: int | None = None
    tile_seconds: list | None = None
    debug_image_scale: float = 1.0


def similarity_group(vectors: list[Vector]) -> list[SimilarityGroup]:
    return vector_similarity_group(vectors)


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
class FastFilterResult:
    passed: list[Vector]
    dropped: list[Vector]
    page_result: FastPageResult


def filter_vectors_fast(
    vectors: list[Vector],
    page: Page,
    *,
    enable_fast: bool = True,
    verbose: bool = False,
    compute=None,
    progress_counter=None,
) -> FastFilterResult:
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


@dataclass
class ReclassifyResult:
    passed: list[Vector]
    dropped: list[Vector]
    fail_reclassified_pass: int
    fail_count: int
    pass_count: int


def reclassify_by_similarity(
    passed: list[Vector],
    dropped: list[Vector],
    groups: list[SimilarityGroup],
    *,
    pass_fraction: float = RECLASSIFY_PASS_FRACTION,
) -> ReclassifyResult:
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


def separate_by_layer_color_width(vectors: list) -> list[list]:
    buckets: list[list] = []
    for layer_bucket in separate_by_layer(vectors).values():
        for color_bucket in separate_by_color(layer_bucket).values():
            buckets.extend(separate_by_width(color_bucket).values())
    return buckets


def _dynamic_tolerance(bbox: tuple, base_tolerance: float, scale: float, cap: float) -> float:
    x0, y0, x1, y1 = bbox
    return min(base_tolerance + scale * max(x1 - x0, y1 - y0), cap)


class _BBoxUnionFind:
    def __init__(self) -> None:
        self.parent: list[int] = []
        self.bbox: list[tuple] = []

    def make_set(self, bbox: tuple) -> int:
        idx = len(self.parent)
        self.parent.append(idx)
        self.bbox.append(bbox)
        return idx

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        self.parent[ra] = rb
        self.bbox[rb] = union_bbox([self.bbox[ra], self.bbox[rb]])
        return rb


def cluster_bucket_spatial(
    vectors: list,
    base_tolerance: float,
    scale: float = CLUSTER_TOLERANCE_SCALE,
    cap: float = CLUSTER_TOLERANCE_MAX,
) -> list[list]:
    if not vectors:
        return []
    uf = _BBoxUnionFind()
    ids: list[int] = []
    for v in vectors:
        idx = uf.make_set(v.bbox)
        existing_roots = {uf.find(j) for j in ids}
        matched = [
            r for r in existing_roots
            if rect_gap(uf.bbox[r], v.bbox) <= _dynamic_tolerance(uf.bbox[r], base_tolerance, scale, cap)
        ]
        cur = idx
        for r in matched:
            cur = uf.union(cur, r)
        ids.append(idx)

    groups: dict[int, list] = {}
    for v, idx in zip(vectors, ids):
        groups.setdefault(uf.find(idx), []).append(v)
    return list(groups.values())


def cluster_buckets(
    buckets: list[list],
    base_tolerance: float,
    scale: float = CLUSTER_TOLERANCE_SCALE,
    cap: float = CLUSTER_TOLERANCE_MAX,
) -> list[list]:
    clusters: list[list] = []
    for bucket in buckets:
        clusters.extend(cluster_bucket_spatial(bucket, base_tolerance, scale, cap))
    return clusters


def build_drawing_output(*drop_lists: list[Vector]) -> list[Vector]:
    vectors: list[Vector] = []
    for drops in drop_lists:
        vectors.extend(drops)
    vectors.sort(key=lambda v: v.seqno)
    return vectors


def cluster_render_padding(cluster: list[Vector]) -> float:
    return max((v.width or 0.0) for v in cluster) / 2.0 + RADON_RENDER_PADDING_EXTRA_PT


def detect_text_paddle_per_cluster(
    clusters: list[list[Vector]], *, dpi: int = 300,
) -> "list[ClusterDetection | None]":
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
    text: "list[list[list[Vector]]]"
    drawing: list[Vector]


def reassign_by_overlap(
    clusters: list[list[Vector]],
    cluster_detections: "list[ClusterDetection | None]",
    *,
    min_coverage: float = PADDLE_REASSIGN_MIN_COVERAGE,
) -> ReassignResult:
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
