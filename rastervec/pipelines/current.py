"""The current extraction pipeline.

    native       = extract_native_text(page)                        # list[Text]
    vectors      = extract_vectors(page)                            # list[Vector]
    fast         = filter_vectors_fast(vectors, page)                # per-Vector,
                                                                      # independent -> passed + dropped
    buckets      = separate_by_layer_color_width(fast.passed)        # (layer, color, width) split
    clusters     = cluster_buckets(buckets, tolerance)                # per-bucket union-find spatial
                                                                       # merge (order-independent):
                                                                       # each set tracks its own bbox,
                                                                       # a vector joins any existing
                                                                       # set within tolerance of that
                                                                       # set's own bbox (tolerance
                                                                       # growing with the set's own
                                                                       # max side length, capped)
    cluster_detections = detect_text_paddle_per_cluster(clusters)     # ONE render per cluster ->
                                                                       # PaddleOCR's own detector ->
                                                                       # quad + rotation approx
    reassignment = reassign_by_overlap(clusters, cluster_detections)  # vector -> best-overlapping
                                                                       # detection (text) or drawing
    segments     = rotate_paddle_detections(...)                      # OCR/radon.sweep_rotation
                                                                       # (vector-geometry angle
                                                                       # refinement, no splitting) +
                                                                       # crop_rotated_detection
    texts        = recognize_segments(segments)                       # PaddleRecBackend,
                                                                       # no similarity/dedup stage
    drawing_vectors = build_drawing_output(reassignment.drawing, fast.dropped)

See `_steps.py` / `Vector/layer_color_separation.py` / `OCR/Paddle_OCR/ocr_backend.py` /
`OCR/radon.py` for each call. This whole module is deprecated but still live -- see
`CLAUDE.md`'s top-of-file note and `docs/old_pipeline_migration.md`; new code should call
`rastervec.core.pipeline.run_pipeline` instead.

The shape-similarity grouping + reclassification step (`similarity_group`/
`reclassify_by_similarity`, backed by the old `Vector_Similarity/similarity.py` ->
`P3_Vector_Parsing/FastIntoPaddle/similarity.py` module) that used to run between `fast` and
`separation` has been removed -- it wasn't actually relied on by anything outside this module's
own tests, and its backing module was deleted along with the `FastIntoPaddle` P3 backend.
`separation` now builds straight from `fast.passed`, and `drawing_vectors` straight from
`fast.dropped`.

CLI: `python -m rastervec.pipelines.current --pdf PATH --page N [-v] [--no-fast]`
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass

from rastervec.config import (
    CLUSTER_TOLERANCE_MAX,
    CLUSTER_TOLERANCE_SCALE,
    FAST_PADDLE_SEQ_MERGE_TOLERANCE,
)
from rastervec.commons.helpers.geometry import rect_gap, union_bbox
from rastervec.commons.logging_setup import get_logger
from rastervec.OCR.Paddle_OCR.ocr_backend import recognize_segments
from rastervec.pipelines._steps import (
    build_drawing_output,
    detect_text_paddle_per_cluster,
    extract_native_text,
    extract_vectors,
    filter_vectors_fast,
    read_page,
    reassign_by_overlap,
    rotate_paddle_detections,
)
from rastervec.pipelines.result import PipelineResult, StepOutcome
from rastervec.P1_Reading_Native.reader import Reader
from rastervec.Vector.layer_color_separation import separate_by_color, separate_by_layer, separate_by_width

_LOG = get_logger("pipelines.current")

__all__ = ["run_pipeline", "STEP_NAMES", "PipelineResult"]

STEP_NAMES = [
    "read", "native", "vectors", "fast",
    "separation", "clusters", "paddle_detect", "assignment", "rotate", "ocr", "drawing",
]


class StepTimer:
    """`with timer("name"): ...` -- records wall-clock per step. On a
    verbose run a failing step is logged, recorded as a `StepOutcome`, and
    suppressed so the rest of the run still produces partial state; on a
    normal run the exception propagates (the benchmark wraps each run)."""

    def __init__(self, *, verbose: bool) -> None:
        self.verbose = verbose
        self.durations: dict[str, float] = {}
        self.outcomes: dict[str, StepOutcome] = {}
        self._name: str | None = None
        self._start = 0.0

    def __call__(self, name: str) -> "StepTimer":
        self._name = name
        return self

    def __enter__(self) -> "StepTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = time.perf_counter() - self._start
        name = self._name or "?"
        self.durations[name] = elapsed
        if exc_type is None:
            self.outcomes[name] = StepOutcome(name, "ok", None, elapsed)
            return False
        _LOG.exception("pipeline step %s failed", name)
        self.outcomes[name] = StepOutcome(name, "error", str(exc), elapsed)
        return self.verbose


def separate_by_layer_color_width(vectors: list) -> list[list]:
    """`(layer, color, width)` separation: chains `separate_by_layer`/
    `separate_by_color` with a third level, `separate_by_width` -- the flat
    list of resulting buckets (bucket identity itself isn't needed
    downstream, only that vectors from different buckets never merge)."""
    buckets: list[list] = []
    for layer_bucket in separate_by_layer(vectors).values():
        for color_bucket in separate_by_color(layer_bucket).values():
            buckets.extend(separate_by_width(color_bucket).values())
    return buckets


def _dynamic_tolerance(bbox: tuple, base_tolerance: float, scale: float, cap: float) -> float:
    x0, y0, x1, y1 = bbox
    return min(base_tolerance + scale * max(x1 - x0, y1 - y0), cap)


class _BBoxUnionFind:
    """Union-find where every root also carries its set's own aggregate
    bbox (the union of every member's bbox merged into it so far)."""

    def __init__(self) -> None:
        self.parent: list[int] = []
        self.bbox: list[tuple] = []  # only meaningful for a root index

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
    """Order-independent union-find spatial merge: each vector is compared
    against **every current distinct set's own aggregate bbox** (not just a
    seqno-adjacent neighbor) via `rect_gap(v.bbox, set_bbox) <=
    tolerance(set_bbox)`, where the tolerance grows with that set's own
    current max side length (`_dynamic_tolerance`, capped at `cap` to
    prevent a runaway spiral where a growing set's own bbox loosens its own
    admission, growing it further still). A vector matching more than one
    set unions all of them together. A vector matching none starts its own
    new set. Never drops anything -- pure regrouping. O(n * live-sets) per
    bucket -- fine since a bucket is already scoped to one (layer, color,
    width) group, typically far smaller than a whole page's vector count."""
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
    """Per-bucket union-find spatial merge (see `cluster_bucket_spatial`),
    flattened into one list of clusters -- never merges across buckets."""
    clusters: list[list] = []
    for bucket in buckets:
        clusters.extend(cluster_bucket_spatial(bucket, base_tolerance, scale, cap))
    return clusters


def run_pipeline(
    pdf_path: str, page_index: int = 0, *, enable_fast: bool = True, verbose: bool = False,
    compute=None, progress_counter=None, stop_after: str | None = None,
) -> PipelineResult:
    """Run the current pipeline on one page. `verbose=True` also retains
    every intermediate value on the returned `PipelineResult`. `compute`,
    when given a `multiprocessing.managers.SyncManager`-hosted `Pool`
    proxy (see `Reader/Parallel`), dispatches FAST tile detection and OCR
    crop recognition to that shared pool instead of running them locally.
    `progress_counter`, when given, is incremented as FAST tiles / OCR crop
    batches complete instead of driving a local `tqdm` bar. `stop_after`,
    when given one of `STEP_NAMES`, skips every step after it (its
    `PipelineResult` fields stay `None`)."""
    timer = StepTimer(verbose=verbose)
    page = native = vectors = None
    fast = buckets = clusters = None
    cluster_detections = reassignment = segments = texts = drawing = None
    rotation_debug: list | None = [] if verbose else None

    if stop_after is not None and stop_after not in STEP_NAMES:
        raise ValueError(f"stop_after must be one of {STEP_NAMES}, got {stop_after!r}")
    stop_idx = len(STEP_NAMES) - 1 if stop_after is None else STEP_NAMES.index(stop_after)

    def _reached(step: str) -> bool:
        return STEP_NAMES.index(step) <= stop_idx

    with Reader(pdf_path) as reader:
        with timer("read"):
            page = read_page(reader, page_index)
        if _reached("native"):
            with timer("native"):
                native = extract_native_text(page)
        if _reached("vectors"):
            with timer("vectors"):
                vectors = extract_vectors(page)
        if _reached("fast"):
            with timer("fast"):
                fast = filter_vectors_fast(
                    vectors or [], page,
                    enable_fast=enable_fast, verbose=verbose, compute=compute,
                    progress_counter=progress_counter,
                )
        if _reached("separation"):
            with timer("separation"):
                buckets = separate_by_layer_color_width(fast.passed if fast else [])
        if _reached("clusters"):
            with timer("clusters"):
                clusters = cluster_buckets(buckets or [], FAST_PADDLE_SEQ_MERGE_TOLERANCE)
        if _reached("paddle_detect"):
            with timer("paddle_detect"):
                cluster_detections = detect_text_paddle_per_cluster(clusters or [])
        if _reached("assignment"):
            with timer("assignment"):
                reassignment = reassign_by_overlap(clusters or [], cluster_detections or [])
        if _reached("rotate"):
            with timer("rotate"):
                segments = rotate_paddle_detections(
                    clusters or [], cluster_detections or [],
                    reassignment.text if reassignment else [],
                    debug_out=rotation_debug,
                )
        if _reached("ocr"):
            with timer("ocr"):
                recognize_fn = None
                if compute is not None:
                    from rastervec.OCR.Paddle_OCR.ocr_backend import _recognize_crops_job

                    recognize_fn = lambda crops: compute.apply(_recognize_crops_job, (crops,))  # noqa: E731
                texts = recognize_segments(segments or [], recognize_fn=recognize_fn)
        if _reached("drawing"):
            with timer("drawing"):
                drawing = build_drawing_output(
                    reassignment.drawing if reassignment else [],
                    fast.dropped if fast else [],
                )

    # The Reader (and its fitz document) is closed now -- detach the dead page
    # handle so a stale-pointer render crashes loudly at the call site instead.
    if page is not None:
        page.fitz_page = None

    all_texts = list(native or []) + list(texts or [])

    return PipelineResult(
        page=page,
        texts=all_texts,
        vectors=drawing or [],
        step_durations=timer.durations,
        engine="current",
        # verbose extras
        native_words=(native if verbose else None),
        vectors_raw=(vectors if verbose else None),
        fast_result=(fast.page_result if verbose and fast else None),
        fast_passed=([[v] for v in fast.passed] if verbose and fast else None),
        fast_dropped_vectors=(fast.dropped if verbose and fast else None),
        separation_buckets=(buckets if verbose else None),
        spatial_clusters=(clusters if verbose else None),
        cluster_detections=(cluster_detections if verbose else None),
        reassigned_drawing=(reassignment.drawing if verbose and reassignment else None),
        reassigned_text=(
            [v for cluster_assigned in reassignment.text for det in cluster_assigned for v in det]
            if verbose and reassignment else None
        ),
        rotated_segments=(segments if verbose else None),
        rotation_debug=(rotation_debug if verbose else None),
        restored_texts=(texts if verbose else None),
        step_outputs=(timer.outcomes if verbose else None),
    )


if __name__ == "__main__":
    from rastervec.pipelines._cli import main

    raise SystemExit(main("current", sys.argv[1:]))
