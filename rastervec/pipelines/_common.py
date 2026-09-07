"""The shared body of the current pipeline. `current.py` is a one-liner
over this; the only knob is `enable_fast` (speed-testing).
"""
from __future__ import annotations

import time

from rastervec.logging_setup import get_logger
from rastervec.pipelines._steps import (
    build_drawing_output,
    detect_text_fast,
    extract_native_text,
    extract_vectors,
    read_page,
    spatial_regroup,
)
from rastervec.pipelines.result import PipelineResult, StepOutcome
from rastervec.pipelines.sub_pipelines.ocr import recognize, segment_for_ocr
from rastervec.pipelines.sub_pipelines.vector_classification import classify_vectors
from rastervec.Reader.reader import Reader

_LOG = get_logger("pipelines.current")

STEP_NAMES = [
    "read", "native", "vectors", "classify", "fast",
    "regroup", "segment", "ocr", "drawing",
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


def run_current_pipeline(
    pdf_path: str, page_index: int, *, enable_fast: bool = True, verbose: bool = False,
) -> PipelineResult:
    timer = StepTimer(verbose=verbose)
    page = native = vectors = cls = fast = regrouped = segments = ocr = drawing = None

    with Reader(pdf_path) as reader:
        with timer("read"):
            page = read_page(reader, page_index)
        with timer("native"):
            native = extract_native_text(page)
        with timer("vectors"):
            vectors = extract_vectors(page)
        with timer("classify"):
            cls = classify_vectors(vectors.paths, page, verbose=verbose)
        with timer("fast"):
            fast = detect_text_fast(
                vectors.paths, cls.text_clusters, cls.similarity_groups, page,
                enable_fast=enable_fast, verbose=verbose,
            )
        with timer("regroup"):
            regrouped = spatial_regroup(fast.passed, cls.cluster_similarity_id)
        with timer("segment"):
            segments = segment_for_ocr(regrouped.clusters, verbose=verbose)
        with timer("ocr"):
            ocr = recognize(
                segments, regrouped.clusters, page,
                similarity_id=regrouped.similarity_id,
            )
        with timer("drawing"):
            drawing = build_drawing_output(cls.dropped, fast.dropped, ocr.failed)

    # The Reader (and its fitz document) is closed now -- detach the dead page
    # handle so a stale-pointer render crashes loudly at the call site instead.
    # Consumers that need a live page call `PipelineResult.open_page()`.
    if page is not None:
        page.fitz_page = None

    return PipelineResult(
        page=page,
        native_words=native or [],
        drawing_vectors=drawing or [],
        ocr_results=ocr.results if ocr else [],
        cluster_ocr_results=ocr.cluster_results if ocr else [],
        text_clusters=cls.text_clusters if cls else [],
        regrouped_clusters=regrouped.clusters if regrouped else [],
        clustering=cls.clustering if cls else {},
        cluster_groups=cls.cluster_groups if cls else {},
        fast_dropped=fast.dropped if fast else [],
        ocr_failed=ocr.failed if ocr else [],
        step_durations=timer.durations,
        engine="current",
        # verbose extras
        vector_paths=(vectors.paths if verbose and vectors else None),
        vector_records=(vectors.records if verbose and vectors else None),
        paths_by_layer=(cls.paths_by_layer if verbose and cls else None),
        paths_by_layer_color=(cls.paths_by_layer_color if verbose and cls else None),
        text_candidate_records=(cls.records if verbose and cls else None),
        similarity_groups=(cls.similarity_groups if verbose and cls else None),
        cluster_similarity_id=(cls.cluster_similarity_id if verbose and cls else None),
        fast_passed=(fast.passed if verbose and fast else None),
        fast_result=(fast.page_result if verbose and fast else None),
        regrouped_cluster_similarity_id=(regrouped.similarity_id if verbose and regrouped else None),
        segmentations=(segments if verbose else None),
        step_outputs=(timer.outcomes if verbose else None),
    )
