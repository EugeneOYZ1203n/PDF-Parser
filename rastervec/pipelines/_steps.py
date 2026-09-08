"""The high-level step functions the pipeline files call. Each is a thin
adapter over one folder's own entrypoint -- the real work lives in those
folders, not here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from rastervec.config import (
    FAST_COMBINED_KEEP_THRESHOLD,
    FAST_PAGE_RENDER_DPI,
    SPATIAL_REGROUP_TOLERANCE_PX,
)
from rastervec.helpers.clustering import cluster_spatial
from rastervec.logging_setup import get_logger
from rastervec.helpers.geometry import PDF_POINTS_PER_INCH, union_bbox
from rastervec.models import DrawingVector, Page, VectorPath
from rastervec.native_text import extract_native_text as _extract_native_text
from rastervec.OCR.fast_detect import FastDetector
from rastervec.pipelines.result import FastPageResult
from rastervec.renderer import render_page_paths, render_reconstructed_page
from rastervec.Vector.vector import extract_vectors as _extract_vectors
from rastervec.Vector_Classification.classification import build_drawing_vectors

if TYPE_CHECKING:
    from rastervec.pipelines.result import PipelineResult
    from rastervec.renderer.notebook import RenderResult

log = get_logger("pipelines.steps")

_SPATIAL_REGROUP_COLOR = "#8b5cf6"

# re-exported so the pipeline files read `extract_native_text(page)` etc.
extract_native_text = _extract_native_text
extract_vectors = _extract_vectors


def read_page(reader, page_index: int) -> Page:
    return reader.get_page(page_index)


# --------------------------------------------------------------------------
# FAST text detection
# --------------------------------------------------------------------------
@dataclass
class FastStepResult:
    passed: list[list[VectorPath]]
    dropped: list[list[VectorPath]]
    page_result: FastPageResult


def _sample_mask(mask, cluster: list[VectorPath], zoom: float) -> float:
    if mask is None:
        return 0.0
    mask_h, mask_w = mask.shape
    total_pixels = 0
    total_score = 0.0
    for p in cluster:
        x0, y0, x1, y1 = p.bbox
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
    vector_paths: list[VectorPath],
    text_clusters: list[list[VectorPath]],
    similarity_groups: list[list[list[VectorPath]]] | None,
    page: Page,
    *,
    enable_fast: bool = True,
    verbose: bool = False,
) -> FastStepResult:
    """Whole-page FAST render + tiled detect; each text-candidate cluster
    scored against the mask, min'd across its similarity group. Clusters
    over `FAST_COMBINED_KEEP_THRESHOLD` pass; the rest become drawing
    content. `enable_fast=False` is a pass-through (keep every cluster)."""
    clusters = text_clusters or []
    if not enable_fast:
        result = FastPageResult(None, None, None, {}, list(clusters), [])
        return FastStepResult(list(clusters), [], result)

    page_image = page_mask = None
    detect_seconds = None
    all_paths = vector_paths or []
    if all_paths:
        detector = FastDetector()
        page_image = render_page_paths(all_paths, page.meta, FAST_PAGE_RENDER_DPI)
        start = time.perf_counter()
        try:
            page_mask = detector.detect_tiled(page_image, desc="FAST text detection")
        except FileNotFoundError as exc:
            log.warning("FAST detection skipped (keeping every cluster): %s", exc)
            result = FastPageResult(None, None, None, {}, list(clusters), [])
            return FastStepResult(list(clusters), [], result)
        detect_seconds = time.perf_counter() - start

    zoom = FAST_PAGE_RENDER_DPI / PDF_POINTS_PER_INCH
    score_by_cluster = {
        id(cluster): _sample_mask(page_mask, cluster, zoom) for cluster in clusters
    }
    groups = similarity_groups or [[cluster] for cluster in clusters]
    scores: dict[int, float] = {}
    for group in groups:
        gs = [score_by_cluster.get(id(c), 0.0) for c in group]
        final = min(gs) if gs else 0.0
        for c in group:
            scores[id(c)] = final

    passed: list[list[VectorPath]] = []
    dropped: list[list[VectorPath]] = []
    for cluster in clusters:
        (passed if scores.get(id(cluster), 0.0) > FAST_COMBINED_KEEP_THRESHOLD else dropped).append(cluster)

    result = FastPageResult(
        page_image if verbose else None,
        page_mask if verbose else None,
        detect_seconds, scores, passed, dropped,
    )
    return FastStepResult(passed, dropped, result)


# --------------------------------------------------------------------------
# spatial regroup
# --------------------------------------------------------------------------
@dataclass
class RegroupResult:
    clusters: list[list[VectorPath]]
    similarity_id: dict[int, int]


def spatial_regroup(
    fast_passed: list[list[VectorPath]],
    cluster_similarity_id: dict[int, int] | None,
) -> RegroupResult:
    """Re-merge FAST-passed clusters whose union bboxes touch/overlap --
    OCR reads one merged region better than several adjacent fragments.
    Carries the similarity-group id forward when a merge's inputs all
    agreed on one."""
    passed = fast_passed or []
    similarity_id = cluster_similarity_id or {}

    merged = cluster_spatial(
        passed, get_bbox=lambda c: union_bbox([p.bbox for p in c]),
        threshold=SPATIAL_REGROUP_TOLERANCE_PX,
    )

    regrouped: list[list[VectorPath]] = []
    regrouped_similarity_id: dict[int, int] = {}
    for pieces in merged:
        flat = [p for piece in pieces for p in piece]
        ids = {similarity_id[id(piece)] for piece in pieces if id(piece) in similarity_id}
        if len(ids) == 1:
            regrouped_similarity_id[id(flat)] = next(iter(ids))
        regrouped.append(flat)

    return RegroupResult(regrouped, regrouped_similarity_id)


def render_regroup(res: "PipelineResult") -> "RenderResult":
    """Notebook visualization for the spatial-regroup step: regrouped
    cluster boxes/paths."""
    from rastervec.renderer.notebook import RenderResult

    rg = res.regrouped_clusters or []
    return RenderResult(
        categories=[{
            "name": f"regrouped clusters ({len(rg)})",
            "color": _SPATIAL_REGROUP_COLOR,
            "bboxes": [union_bbox([p.bbox for p in c]) for c in rg if c],
            "paths": [p for c in rg for p in c],
            "path_color": _SPATIAL_REGROUP_COLOR,
        }],
        note=(
            f"{len(res.fast_passed or [])} FAST-passed -> {len(rg)} regrouped "
            f"(union-bbox tol {SPATIAL_REGROUP_TOLERANCE_PX}pt)"
        ),
    )


# --------------------------------------------------------------------------
# drawing output
# --------------------------------------------------------------------------
def build_drawing_output(
    classification_dropped: list[VectorPath],
    fast_dropped: list[list[VectorPath]],
    ocr_failed: list[list[VectorPath]],
) -> list[DrawingVector]:
    """Fold every rejected group -- classification drops, FAST drops, OCR
    blanks -- back into one `DrawingVector` per original drawing, in the
    source PDF's own draw order."""
    paths: list[VectorPath] = list(classification_dropped)
    for cluster in fast_dropped or []:
        paths.extend(cluster)
    for cluster in ocr_failed or []:
        paths.extend(cluster)
    paths.sort(key=lambda p: (p.seq, p.item_index))
    return build_drawing_vectors(paths)


def render_drawing(res: "PipelineResult", *, zoom: float = 1.0) -> "RenderResult":
    """Notebook visualization for the drawing-vectors output: dashed vs
    solid bboxes, plus a full page reconstruction."""
    from rastervec.renderer.notebook import DEFAULT_PATH_COLOR, RenderResult

    dv = res.drawing_vectors or []
    dashed = [d for d in dv if d.dashed]
    solid = [d for d in dv if not d.dashed]
    recon = render_reconstructed_page(res.page.meta, drawing_vectors=dv, zoom=zoom)
    return RenderResult(
        categories=[
            {"name": f"dashed ({len(dashed)})", "color": DEFAULT_PATH_COLOR, "bboxes": [d.bbox for d in dashed]},
            {"name": f"solid ({len(solid)})", "color": DEFAULT_PATH_COLOR, "bboxes": [d.bbox for d in solid]},
            {"name": "full reconstruction", "isolated": recon, "overlay": recon},
        ],
        note=f"{len(dv)} drawing vector(s)",
    )
