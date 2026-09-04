"""Pipeline orchestration: shared stage-running machinery driven by the CLI
in this file, `run_page_context`, and
`rastervec/notebooks/pipeline_stage_visualization.ipynb`.

Only calls high-level class methods -- no inline extraction logic. A stage
is a (key, label, run) triple appended to Pipeline.STAGES; run(ctx) mutates
the shared PipelineContext and returns this stage's output data. Adding a
new stage once it's actually implemented means adding one StageSpec here;
the CLI picks it up automatically, and the visualization notebook gets a
per-stage cell calling `visualize(<key>, [...])`.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from PIL import Image

    from rastervec.OCR.Paddle_OCR.ocr_backend import OcrBackend
    from rastervec.output_types import NativePDFElements

if __name__ == "__main__" and __package__ is None:
    # Allow running this file directly (`python rastervec/pipeline.py`),
    # not just as a module (`python -m rastervec.pipeline`), by putting
    # the repo root -- the parent of this package -- on sys.path.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rastervec.helpers.clustering import Clustering
from rastervec.helpers.geometry import union_bbox
from rastervec.logging_setup import configure_logging, get_logger
from rastervec.models import (
    ClusterOcrResult,
    DrawingVector,
    Page,
    TextRecord,
    TextVectorResult,
    TextWord,
    VectorPath,
    VectorRecord,
)
from rastervec.Native_Text.native import Native
from rastervec.OCR.FAST_Text_Detect.fast_detect import FastDetector
from rastervec.OCR.Paddle_OCR.light_backend import LightPaddleOcrBackend
from rastervec.OCR.Paddle_OCR.render_ocr import RenderOCR
from rastervec.Reader.reader import Reader
from rastervec.renderer import render_page_paths
from rastervec.Vector.vector import Vector
from rastervec.Vector_Classification.classification import StepResult, VectorClassifier

# fast_text_detect stage: pass threshold applied to each cluster's
# candidates-render score, after min-ing it across each cluster's own
# similarity group (see _run_fast_text_detect).
FAST_COMBINED_KEEP_THRESHOLD = 0.2

# spatial_regroup stage: aggregate (union) bbox gap tolerance (PDF points)
# for re-merging FAST-passed clusters before OCR -- two clusters merge only
# when their union bboxes are within this gap (rect_gap is 0.0 for
# overlapping/touching boxes) *and* they share the same (layer, color) key
# (see _cluster_lc_key/_run_spatial_regroup). Unlike the earlier
# clustering steps this still ignores which unique_clusters similarity
# group FAST scored a cluster under -- similarity-group boundaries are the
# only ones spatial_regroup crosses.
SPATIAL_REGROUP_TOLERANCE_PX = 1.0

# ocr_compare stage: when True, _run_ocr_compare uses LightPaddleOcrBackend
# (own ink-projection line/word segmentation + PaddleOCR recognition-only +
# DocImgOrientationClassification rotation) instead of the full PP-OCRv6
# detect+rec+doc-orient+textline pipeline. A PipelineContext.ocr_backend
# override always wins over this flag.
USE_LIGHT_OCR_BACKEND = True

_LOG = get_logger("pipeline")

# (layer, color) -- one Vector.separate_by_color() bucket.
GroupKey = tuple[str, tuple]

# DPI the whole-page FAST render is rasterized at (fast_text_detect stage),
# before FastDetector.detect_tiled's own further upscale (see
# helpers/fast_detect.py's TILED_SCALE_FACTOR) -- not full OCR resolution
# (RenderOCR's own per-cluster/per-group renders use 300 DPI).
FAST_PAGE_RENDER_DPI = 150


@dataclass
class ClusteringStageResult:
    """One (layer, color) group's result from the Vector Classification
    pipeline (see Vector_Classification/classification.py's module
    docstring): `steps` is exactly `VectorClassifier.cluster()`'s return
    value -- one `StepResult` per step, each holding every named
    `CategoryResult` that step produced.
    `steps[-1].categories["kept"]` is the final surviving clusters, handed
    to text_candidates. Every non-`"kept"` `role="dropped"` category
    across every step was "classified as Drawing" the moment it was
    produced (see _run_drawing_vectors)."""

    steps: list[StepResult]


@dataclass
class FastPageResult:
    """fast_text_detect's whole-page result: `page_image`/`page_mask` (via
    renderer.render_page_paths + FastDetector.detect_tiled, over every
    extracted vector path on the page -- drawing content included, not
    just text candidates), plus `scores`: for every text-candidate cluster
    (keyed by `id(cluster)`), its own page-mask score after taking the min
    across each cluster's own similarity group (see _run_fast_text_detect).
    `passed`/`dropped` split `scores` against FAST_COMBINED_KEEP_THRESHOLD
    (same lists as ctx.fast_passed/ctx.fast_dropped, duplicated here so the
    debug app's stage renderer is self-contained, like every other stage's
    StageOutput.data). `page_image`/`page_mask` are `None` when there were
    zero vector paths on the page (FastDetector's underlying torch model is
    never even constructed in that case). `detect_seconds` is how long the
    `FastDetector.detect_tiled()` call took, `None` if it never ran --
    surfaced by the debug app as a timing readout."""

    page_image: "Image.Image | None"
    page_mask: "np.ndarray | None"
    detect_seconds: float | None
    scores: dict[int, float]
    passed: list[list[VectorPath]]
    dropped: list[list[VectorPath]]


@dataclass
class PipelineContext:
    """Accumulates state across stages for one page run."""

    reader: Reader
    page_index: int
    # fast_text_detect toggle: when False, _run_fast_text_detect is a
    # pass-through (every text candidate flows to spatial_regroup/OCR, and
    # unique_clusters' similarity groups go unused) -- speed testing only,
    # default keeps FAST on.
    enable_fast: bool = True
    # ocr_compare backend override: None -> _run_ocr_compare picks per the
    # module-level USE_LIGHT_OCR_BACKEND flag.
    ocr_backend: "OcrBackend | None" = None
    page: Page | None = None
    native_words: list[TextWord] | None = None
    # native: richer, additive counterpart to native_words (see
    # Native.extract_records) -- carries wmode/block_no/line_no/word_no
    # alongside everything native_words already has.
    native_records: list[TextRecord] | None = None
    vector_paths: list[VectorPath] | None = None
    # vector_extract: richer, additive counterpart to vector_paths (see
    # Vector.extract_records) -- one VectorRecord per raw get_drawings()
    # drawing, carrying every drawing-level field vector_paths drops.
    vector_records: list[VectorRecord] | None = None
    # text_candidates: one VectorRecord per final surviving "kept" cluster
    # (see VectorClassifier.build_vector_records), role="kept", groups set
    # from that cluster's own StepResult.cluster_groups lineage.
    text_candidate_records: list[VectorRecord] | None = None
    paths_by_layer: dict[str, list[VectorPath]] | None = None
    paths_by_layer_color: dict[str, dict[tuple, list[VectorPath]]] | None = None
    clustering: dict[GroupKey, ClusteringStageResult] | None = None
    # text_candidates: every clustering bucket's final surviving "kept"
    # clusters, flattened together (every filter/cluster step's drops go to
    # drawing_vectors instead, never here).
    text_clusters: list[list[VectorPath]] | None = None
    # text_candidates: id(cluster) -> the pre-spatial-clustering "groups"
    # that cluster is composed of (see Vector_Classification/
    # classification.py's module docstring for the group/cluster
    # distinction) -- merged across every (layer, color) bucket's own
    # StepResult.cluster_groups.
    cluster_groups: dict[int, list[list[VectorPath]]] | None = None
    # unique_clusters: text_clusters grouped by whole-page geometric
    # similarity (see VectorClassifier.group_similar_clusters);
    # cluster_similarity_id is id(cluster) -> index into similarity_groups,
    # for O(1) lookup.
    similarity_groups: list[list[list[VectorPath]]] | None = None
    cluster_similarity_id: dict[int, int] | None = None
    # fast_text_detect: the whole-page renders/masks + per-cluster scores.
    fast_result: FastPageResult | None = None
    # fast_text_detect: text_clusters split by the combined FAST score
    # (see FastPageResult) against FAST_COMBINED_KEEP_THRESHOLD.
    fast_passed: list[list[VectorPath]] | None = None
    fast_dropped: list[list[VectorPath]] | None = None
    # spatial_regroup: fast_passed re-merged where aggregate bboxes are
    # within SPATIAL_REGROUP_TOLERANCE_PX *and* share a (layer, color) key
    # (_cluster_lc_key) -- still ignores unique_clusters similarity-group
    # boundaries. This, not fast_passed, is what ocr_compare actually OCRs.
    regrouped_clusters: list[list[VectorPath]] | None = None
    # ocr_compare: one ClusterOcrResult per regrouped_clusters cluster --
    # a single direct OCR call over the whole cluster, no fallback tiers.
    cluster_ocr_results: list[ClusterOcrResult] | None = None
    # ocr_compare: just the resolved reading out of cluster_ocr_results,
    # kept as its own flat list for to_native_pdf_elements/reconstruction.
    ocr_results: list[TextVectorResult] | None = None
    # ocr_compare: full path lists of every cluster whose OCR reading came
    # back blank (see _run_ocr_compare) -- folded into drawing_vectors,
    # same as every other rejection in this pipeline.
    ocr_failed: list[list[VectorPath]] | None = None
    # drawing_vectors now runs last -- folds in the clustering chain's own
    # drops, fast_dropped (FAST found no text in these), and ocr_failed
    # (OCR resolution failed for these).
    drawing_vectors: list[DrawingVector] | None = None
    # _run_stages: wall-clock seconds each stage's run(ctx) took, keyed by
    # stage key, in run order. Filled for every stage that ran (including
    # one that raised). sum(stage_durations.values()) is the per-page
    # pipeline total -- consumed by the benchmark notebook's timing report.
    stage_durations: dict[str, float] | None = None
    # Future stages add fields here as they're implemented (raster_images,
    # etc.) so later stages can read earlier stages' output.

    def to_native_pdf_elements(self) -> "NativePDFElements":
        """Serialization/export boundary: standardized output_types.py DTOs
        built from whatever this run's native_words/drawing_vectors ended
        up as. A plain method, not a pipeline stage -- the internal
        dataclasses stay canonical throughout the pipeline; this is only
        for callers that want the pymupdf-mirroring output shape (e.g. a
        future --dump-json CLI flag, or evaluation.py's serialization
        boundary)."""
        from rastervec.output_types import NativePDFElements

        return NativePDFElements.from_extract(
            words=self.native_words or [], drawings=self.drawing_vectors or [],
        )


@dataclass
class StageOutput:
    key: str
    label: str
    status: str  # "ok" | "error"
    data: Any = None
    error: str | None = None
    # Wall-clock seconds spec.run(ctx) took (set even when status="error").
    duration_seconds: float | None = None


@dataclass
class StageSpec:
    key: str
    label: str
    run: Callable[[PipelineContext], Any]


def _run_reader(ctx: PipelineContext) -> Page:
    ctx.page = ctx.reader.get_page(ctx.page_index)
    return ctx.page


def _run_native(ctx: PipelineContext) -> list[TextWord]:
    native = Native()
    ctx.native_words = native.extract_text(ctx.page)
    ctx.native_records = native.extract_records(ctx.page)
    return ctx.native_words


def _run_vector_extract(ctx: PipelineContext) -> list[VectorPath]:
    vector = Vector()
    ctx.vector_paths = vector.extract_paths(ctx.page)
    ctx.vector_records = vector.extract_records(ctx.page)
    return ctx.vector_paths


def _run_layer_separation(ctx: PipelineContext) -> dict[str, list[VectorPath]]:
    ctx.paths_by_layer = Vector().separate_by_layer(ctx.vector_paths)
    return ctx.paths_by_layer


def _run_color_separation(ctx: PipelineContext) -> dict[str, dict[tuple, list[VectorPath]]]:
    vector = Vector()
    ctx.paths_by_layer_color = {
        layer: vector.separate_by_color(paths)
        for layer, paths in ctx.paths_by_layer.items()
    }
    return ctx.paths_by_layer_color


def _iter_groups(
    paths_by_layer_color: dict[str, dict[tuple, list[VectorPath]]]
) -> list[tuple[GroupKey, list[VectorPath]]]:
    return [
        ((layer, color), paths)
        for layer, color_groups in paths_by_layer_color.items()
        for color, paths in color_groups.items()
    ]


def _run_clustering(ctx: PipelineContext) -> dict[GroupKey, ClusteringStageResult]:
    classifier = VectorClassifier()
    result: dict[GroupKey, ClusteringStageResult] = {}

    for key, paths in _iter_groups(ctx.paths_by_layer_color):
        result[key] = ClusteringStageResult(steps=classifier.cluster(paths, ctx.page))

    ctx.clustering = result
    return result


def _run_text_candidates(ctx: PipelineContext) -> list[list[VectorPath]]:
    """Gathers every clustering bucket's final surviving "kept" clusters
    (every filter/cluster step's drops went to drawing_vectors instead,
    never here) into ctx.text_clusters, and merges every bucket's own
    StepResult.cluster_groups (id(cluster) -> its composing pre-spatial
    "groups") into one ctx.cluster_groups dict -- consumed by
    ocr_compare's group-vs-cluster comparison."""
    text_clusters: list[list[VectorPath]] = []
    cluster_groups: dict[int, list[list[VectorPath]]] = {}
    text_candidate_records: list[VectorRecord] = []
    classifier = VectorClassifier()
    for cluster_result in ctx.clustering.values():
        if not cluster_result.steps:
            continue
        last = cluster_result.steps[-1]
        text_clusters.extend(last.categories["kept"].groups)
        if last.cluster_groups:
            cluster_groups.update(last.cluster_groups)
        text_candidate_records.extend(classifier.build_vector_records(cluster_result.steps))

    ctx.text_clusters = text_clusters
    ctx.cluster_groups = cluster_groups
    ctx.text_candidate_records = text_candidate_records
    return text_clusters


def _run_unique_clusters(ctx: PipelineContext) -> list[list[list[VectorPath]]]:
    """Whole-page geometric similarity grouping of text_clusters (see
    VectorClassifier.group_similar_clusters/Glossary.md's "similarity
    group" entry) -- run before fast_text_detect so its scoring can be
    min'd across each similarity group."""
    groups = VectorClassifier().group_similar_clusters(ctx.text_clusters or [])
    ctx.similarity_groups = groups
    ctx.cluster_similarity_id = {
        id(cluster): gi for gi, group in enumerate(groups) for cluster in group
    }
    return groups


def _sample_mask(mask: "np.ndarray | None", cluster: list[VectorPath], zoom: float) -> float:
    """Mean mask probability over each of `cluster`'s own member paths'
    individual bbox regions (pixel-count-weighted across paths), mapped
    from page-space into `mask`'s pixel space at `zoom`) -- not the
    cluster's aggregate bbox, which would dilute (or inflate) the score
    with interior whitespace the vectors themselves never actually touch.
    0.0 if `mask` is None or no member path has any pixel area."""
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


def _run_fast_text_detect(ctx: PipelineContext) -> FastPageResult:
    """Renders one whole-page image of every extracted vector path on the
    page (ctx.vector_paths -- drawing content included, not just surviving
    text-candidate clusters) via renderer.render_page_paths, and runs
    FastDetector.detect_tiled once over it -- `detect_tiled` upscales the
    render and tiles/rotates it (see helpers/fast_detect.py) for a more
    robust per-region score than one downsized whole-page pass would give.
    Each text-candidate cluster's own score is `_sample_mask` over the
    resulting mask (sampled at its own bbox region, whether or not that
    region happened to include non-candidate paths too), then -- per
    unique_clusters' similarity group -- the final score is the min of that
    score across every member of that group (so if any similar-looking
    cluster on the page scored low, every cluster judged the "same" shares
    that low score). A cluster passes (ctx.fast_passed) if its final score
    exceeds FAST_COMBINED_KEEP_THRESHOLD; the rest are reclassified as
    drawing content (ctx.fast_dropped, folded into drawing_vectors). A page
    with zero vector paths renders/detects nothing, so FastDetector's
    underlying torch model is never even constructed (see
    helpers/fast_detect.py).

    When ctx.enable_fast is False this is a pass-through: every
    text-candidate cluster is kept (ctx.fast_passed = ctx.text_clusters),
    nothing is dropped, and no render/detection runs -- unique_clusters'
    similarity groups then go unused. Speed-testing toggle only; the
    default keeps FAST on."""
    clusters = ctx.text_clusters or []
    all_paths = ctx.vector_paths or []

    if not ctx.enable_fast:
        result = FastPageResult(
            page_image=None, page_mask=None, detect_seconds=None,
            scores={}, passed=list(clusters), dropped=[],
        )
        ctx.fast_result = result
        ctx.fast_passed = list(clusters)
        ctx.fast_dropped = []
        return result

    page_image = page_mask = None
    detect_seconds = None

    if all_paths:
        detector = FastDetector()
        page_image = render_page_paths(
            all_paths, ctx.page.meta, FAST_PAGE_RENDER_DPI,
        )
        start = time.perf_counter()
        page_mask = detector.detect_tiled(page_image, desc="FAST text detection")
        detect_seconds = time.perf_counter() - start

    zoom = FAST_PAGE_RENDER_DPI / 72.0
    score_by_cluster: dict[int, float] = {
        id(cluster): _sample_mask(page_mask, cluster, zoom) for cluster in clusters
    }

    similarity_groups = ctx.similarity_groups or [[cluster] for cluster in clusters]
    scores: dict[int, float] = {}
    for group in similarity_groups:
        group_scores = [score_by_cluster.get(id(cluster), 0.0) for cluster in group]
        final_score = min(group_scores) if group_scores else 0.0
        for cluster in group:
            scores[id(cluster)] = final_score

    passed: list[list[VectorPath]] = []
    dropped: list[list[VectorPath]] = []
    for cluster in clusters:
        (passed if scores.get(id(cluster), 0.0) > FAST_COMBINED_KEEP_THRESHOLD else dropped).append(cluster)

    result = FastPageResult(
        page_image=page_image, page_mask=page_mask,
        detect_seconds=detect_seconds,
        scores=scores, passed=passed, dropped=dropped,
    )
    ctx.fast_result = result
    ctx.fast_passed = passed
    ctx.fast_dropped = dropped
    return result


def _cluster_lc_key(cluster: list[VectorPath]) -> tuple:
    """A cluster's (layer, color) bucket key, taken from its first member
    path -- clusters are homogeneous in (layer, color) because
    classification runs strictly within one Vector.separate_by_layer /
    separate_by_color bucket and never merges across them, so cluster[0]
    is representative. Matches Layer_Color_Separation's own keys:
    separate_by_layer keys on `layer or ""`, separate_by_color on the
    (stroke_color, fill_color, stroke_opacity, fill_opacity) 4-tuple."""
    if not cluster:
        return ("", (None, None, None, None))
    p = cluster[0]
    return (p.layer or "", (p.stroke_color, p.fill_color, p.stroke_opacity, p.fill_opacity))


def _run_spatial_regroup(ctx: PipelineContext) -> list[list[VectorPath]]:
    """Re-merges FAST-passed clusters whose aggregate (union) bboxes
    overlap or sit within SPATIAL_REGROUP_TOLERANCE_PX of each other
    (rect_gap is 0.0 for overlapping/touching boxes) *and* which share the
    same (layer, color) bucket key (_cluster_lc_key). Two nearby
    same-paint clusters that classification/FAST happened to keep as
    separate pieces are stitched back into one before OCR sees them, since
    OCR reads better over one merged text region than several adjacent
    fragments -- but a nearby cluster in a different layer or a different
    stroke/fill colour is left alone, matching how every earlier
    clustering step stays within one (layer, color) bucket (see
    Vector_Classification/classification.py's module docstring). The only
    boundary this step still crosses is unique_clusters' similarity
    groups.

    `extra_close` gates the merge on the shared (layer, color) key;
    Clustering.cluster_spatial's `threshold` is the aggregate-bbox gap
    tolerance. Nothing else is tracked onto the merged piece -- ocr_compare
    OCRs each merged piece directly."""
    passed = ctx.fast_passed or []

    merged = Clustering().cluster_spatial(
        passed, get_bbox=lambda c: union_bbox([p.bbox for p in c]),
        threshold=SPATIAL_REGROUP_TOLERANCE_PX,
        extra_close=lambda a, b: _cluster_lc_key(a) == _cluster_lc_key(b),
    )

    regrouped = [[p for piece in pieces for p in piece] for pieces in merged]
    ctx.regrouped_clusters = regrouped
    return regrouped


def _run_ocr_compare(ctx: PipelineContext) -> list[ClusterOcrResult]:
    """Cluster OCR only: one direct RenderOCR.ocr_cluster call per
    spatial_regroup cluster, no fallback tiers, no similarity-group reuse.
    A cluster's reading counts as failed if its text comes back blank --
    its full path list is collected into ctx.ocr_failed (folded into
    drawing_vectors, same as every other rejection in this pipeline), in
    addition to being kept (blank) in ctx.ocr_results.

    Backend: ctx.ocr_backend if set, else LightPaddleOcrBackend when
    USE_LIGHT_OCR_BACKEND (the default -- own ink-projection segmentation
    + PaddleOCR recognition-only), else RenderOCR's own PaddleOcrBackend
    default (full PP-OCRv6 detect+rec+orientation pipeline)."""
    backend = ctx.ocr_backend
    if backend is None and USE_LIGHT_OCR_BACKEND:
        backend = LightPaddleOcrBackend()
    render_ocr = RenderOCR(backend=backend)
    clusters = ctx.regrouped_clusters or []

    results: list[ClusterOcrResult] = []
    ocr_results: list[TextVectorResult] = []
    ocr_failed: list[list[VectorPath]] = []

    for cluster in tqdm(clusters, desc="OCR compare", unit="cluster"):
        start = time.perf_counter()
        resolved = render_ocr.ocr_cluster(cluster, ctx.page)
        ocr_seconds = time.perf_counter() - start

        results.append(ClusterOcrResult(cluster=cluster, resolved=resolved, ocr_seconds=ocr_seconds))
        ocr_results.append(resolved)
        if not resolved.text.strip():
            ocr_failed.append(cluster)

    ctx.cluster_ocr_results = results
    ctx.ocr_results = ocr_results
    ctx.ocr_failed = ocr_failed
    return results


def _run_drawing_vectors(ctx: PipelineContext) -> list[DrawingVector]:
    """Final aggregation, run last: every category any step in the
    clustering chain marked `role="dropped"` was already "classified as
    Drawing" the moment it was produced (see ClusteringStageResult's
    docstring), so it's folded in here unconditionally -- plus every path
    belonging to a cluster FAST found no text in (ctx.fast_dropped) and
    every path belonging to a cluster whose OCR resolution failed (ctx.
    ocr_failed), both reclassified as drawing content for the same reason.
    Whatever ocr_results still holds text for is the only content that
    doesn't end up in drawing_vectors."""
    classifier = VectorClassifier()
    all_drawing_paths: list[VectorPath] = []

    for cluster_result in ctx.clustering.values():
        for step in cluster_result.steps:
            for category in step.categories.values():
                if category.role == "dropped":
                    for group in category.groups:
                        all_drawing_paths.extend(group)

    for cluster in ctx.fast_dropped or []:
        all_drawing_paths.extend(cluster)

    for cluster in ctx.ocr_failed or []:
        all_drawing_paths.extend(cluster)

    # ctx.clustering is keyed by (layer, color) -- iterating its buckets
    # loses each path's original PDF stacking order across layer/color
    # boundaries. Sort back to (seq, item_index) -- the original
    # get_drawings() draw order -- before aggregating into DrawingVectors,
    # so render order matches the source PDF regardless of which bucket a
    # path was classified into during clustering.
    all_drawing_paths.sort(key=lambda p: (p.seq, p.item_index))

    ctx.drawing_vectors = classifier.build_drawing_vectors(all_drawing_paths)
    return ctx.drawing_vectors


class Pipeline:
    """Runs Pipeline.STAGES in order for one page, collecting each
    stage's output. A stage that raises does not stop the run or crash a
    caller (e.g. the debug app) -- it's recorded as status="error"."""

    STAGES: list[StageSpec] = [
        StageSpec(key="reader", label="Reader", run=_run_reader),
        StageSpec(key="native", label="Native Text", run=_run_native),
        StageSpec(key="vector_extract", label="Vector Extraction", run=_run_vector_extract),
        StageSpec(key="layer_separation", label="Layer Separation", run=_run_layer_separation),
        StageSpec(key="color_separation", label="Color Separation", run=_run_color_separation),
        StageSpec(key="clustering", label="Clustering", run=_run_clustering),
        StageSpec(key="text_candidates", label="Text Candidates", run=_run_text_candidates),
        StageSpec(key="unique_clusters", label="Unique Clusters", run=_run_unique_clusters),
        StageSpec(key="fast_text_detect", label="FAST: Text Detect", run=_run_fast_text_detect),
        StageSpec(key="spatial_regroup", label="Spatial Regroup", run=_run_spatial_regroup),
        StageSpec(key="ocr_compare", label="OCR Compare", run=_run_ocr_compare),
        StageSpec(key="drawing_vectors", label="Drawing Vectors", run=_run_drawing_vectors),
    ]

    @classmethod
    def stage_keys(cls) -> list[str]:
        return [spec.key for spec in cls.STAGES]

    def run_page(
        self,
        reader: Reader,
        page_index: int,
        final_stage: str | None = None,
        *,
        enable_fast: bool = True,
        ocr_backend: "OcrBackend | None" = None,
    ) -> list[StageOutput]:
        """Runs Pipeline.STAGES in order, stopping after `final_stage`
        (inclusive) instead of running every stage -- e.g. `final_stage=
        "fast_text_detect"` skips ocr_compare (and any later stage)
        entirely, never even constructing a RenderOCR/PaddleOCR engine, so
        it's a real way to skip the OCR round-trip while iterating on
        earlier stages, not just a display-time filter. `None` (default)
        runs every stage, unchanged from before this parameter existed.

        `enable_fast=False` turns fast_text_detect into a pass-through
        (every text candidate reaches OCR) -- speed-testing toggle.
        `ocr_backend` overrides the ocr_compare backend (default: light
        backend per USE_LIGHT_OCR_BACKEND)."""
        if final_stage is not None and final_stage not in self.stage_keys():
            raise ValueError(
                f"unknown final_stage {final_stage!r}; must be one of {self.stage_keys()}"
            )

        ctx = PipelineContext(reader=reader, page_index=page_index)
        ctx.enable_fast = enable_fast
        ctx.ocr_backend = ocr_backend
        return self._run_stages(ctx, final_stage)

    @classmethod
    def _run_stages(
        cls, ctx: PipelineContext, final_stage: str | None,
    ) -> list[StageOutput]:
        outputs: list[StageOutput] = []
        ctx.stage_durations = {}
        for spec in cls.STAGES:
            start = time.perf_counter()
            try:
                data = spec.run(ctx)
                elapsed = time.perf_counter() - start
                outputs.append(
                    StageOutput(spec.key, spec.label, "ok", data, duration_seconds=elapsed)
                )
            except Exception as exc:  # noqa: BLE001 -- debug tool: surface, don't crash
                elapsed = time.perf_counter() - start
                _LOG.exception("stage %s failed", spec.key)
                outputs.append(
                    StageOutput(
                        spec.key, spec.label, "error", None, str(exc),
                        duration_seconds=elapsed,
                    )
                )
            ctx.stage_durations[spec.key] = elapsed
            if spec.key == final_stage:
                break
        return outputs


def run_page_context(
    reader: Reader, page_index: int, final_stage: str | None = None,
    *, enable_fast: bool = True, ocr_backend: "OcrBackend | None" = None,
) -> PipelineContext:
    """Like Pipeline.run_page, but returns the PipelineContext itself
    (every field the run's stages set, e.g. ctx.text_clusters) instead of
    the list[StageOutput] -- for callers that want to read pipeline state
    directly rather than each stage's StageOutput.data (e.g. Evaluation/
    Labelling/manual_label.py, which needs ctx.text_clusters to draw
    clickable cluster bboxes). Runs the exact same Pipeline.STAGES list/
    order as run_page, so future stage changes never need mirroring at a
    call site that would otherwise hand-roll its own partial stage
    sequence.

    `enable_fast=False` -> fast_text_detect pass-through; `ocr_backend`
    overrides the ocr_compare backend (see Pipeline.run_page)."""
    if final_stage is not None and final_stage not in Pipeline.stage_keys():
        raise ValueError(
            f"unknown final_stage {final_stage!r}; must be one of {Pipeline.stage_keys()}"
        )
    ctx = PipelineContext(reader=reader, page_index=page_index)
    ctx.enable_fast = enable_fast
    ctx.ocr_backend = ocr_backend
    Pipeline._run_stages(ctx, final_stage)
    return ctx


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the rastervec extraction pipeline on one PDF page."
    )
    parser.add_argument("--pdf", required=True, help="Path to the input PDF.")
    parser.add_argument(
        "--page",
        type=int,
        default=0,
        help="0-based page index to process (default: 0).",
    )
    parser.add_argument(
        "--final-stage",
        choices=Pipeline.stage_keys(),
        default=None,
        help="Stop after this stage instead of running the whole pipeline -- e.g. "
        "--final-stage fast_text_detect skips ocr_compare (and the PaddleOCR "
        "engine it would otherwise build) entirely. Default: run every stage.",
    )
    parser.add_argument(
        "--no-fast",
        action="store_true",
        help="Turn fast_text_detect into a pass-through (every text candidate "
        "reaches OCR). Speed-testing toggle. Default: FAST on.",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging."
    )
    verbosity.add_argument(
        "-q", "--quiet", action="store_true", help="Only log WARNING and above."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    configure_logging(level)

    with Reader(args.pdf) as reader:
        outputs = Pipeline().run_page(
            reader, args.page, final_stage=args.final_stage,
            enable_fast=not args.no_fast,
        )

        for output in outputs:
            if output.status == "error":
                _LOG.error("stage %s (%s) failed: %s", output.key, output.label, output.error)
                continue

            if output.key == "native":
                words: list[TextWord] = output.data
                _LOG.info(
                    "extracted %d native word(s) from page %d", len(words), args.page
                )
                for word in words[:10]:
                    _LOG.info("  seq=%d %r @ %s", word.seq, word.text, word.bbox)
            else:
                _LOG.info("stage %s (%s) ok", output.key, output.label)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
