"""The shared body of the current pipeline. `current.py` is a one-liner
over this; the only knob is `enable_fast` (speed-testing).
"""
from __future__ import annotations

import time

from rastervec.logging_setup import get_logger
from rastervec.pipelines._steps import (
    build_drawing_output,
    detect_text_fast,
    detect_text_paddle,
    extract_native_text,
    extract_vectors,
    read_page,
)
from rastervec.pipelines.result import PipelineResult, StepOutcome
from rastervec.pipelines.sub_pipelines.vector_classification import classify_vectors
from rastervec.Reader.reader import Reader

_LOG = get_logger("pipelines.current")

# Test branch (`test/paddle-detect-post-fast`): everything after `fast` is
# replaced by one `paddle_detect` step -- see `_steps.py::detect_text_paddle`.
STEP_NAMES = [
    "read", "native", "vectors", "classify",
    "fast", "paddle_detect", "drawing",
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
    compute=None, progress_counter=None, stop_after: str | None = None,
) -> PipelineResult:
    timer = StepTimer(verbose=verbose)
    page = native = vectors = cls = fast = paddle_boxes = drawing = None

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
        if _reached("classify"):
            with timer("classify"):
                cls = classify_vectors(vectors or [], page, verbose=verbose)
        if _reached("fast"):
            with timer("fast"):
                flat_clusters = [
                    [v for group in cluster for v in group] for cluster in (cls.text_clusters if cls else [])
                ]
                fast = detect_text_fast(
                    flat_clusters, page,
                    enable_fast=enable_fast, verbose=verbose, compute=compute,
                    progress_counter=progress_counter,
                )
        if _reached("paddle_detect"):
            with timer("paddle_detect"):
                paddle_boxes = detect_text_paddle(fast.passed if fast else [], page)
        if _reached("drawing"):
            with timer("drawing"):
                drawing = build_drawing_output(
                    cls.drawing_vectors if cls else [], fast.dropped_vectors if fast else [],
                )

    # The Reader (and its fitz document) is closed now -- detach the dead page
    # handle so a stale-pointer render crashes loudly at the call site instead.
    # Consumers that need a live page call `PipelineResult.open_page()`.
    if page is not None:
        page.fitz_page = None

    all_texts = list(native or [])

    return PipelineResult(
        page=page,
        texts=all_texts,
        vectors=drawing or [],
        step_durations=timer.durations,
        engine="current",
        paddle_boxes=paddle_boxes,
        # verbose extras
        native_words=(native if verbose else None),
        vectors_raw=(vectors if verbose else None),
        vectors_by_layer=(cls.vectors_by_layer if verbose and cls else None),
        vectors_by_layer_color=(cls.vectors_by_layer_color if verbose and cls else None),
        text_clusters=(cls.text_clusters if verbose and cls else None),
        clustering=(cls.clustering if verbose and cls else None),
        classification_dropped=(cls.drawing_vectors if verbose and cls else None),
        fast_result=(fast.page_result if verbose and fast else None),
        fast_passed=(fast.passed if verbose and fast else None),
        fast_dropped_vectors=(fast.dropped_vectors if verbose and fast else None),
        step_outputs=(timer.outcomes if verbose else None),
    )
