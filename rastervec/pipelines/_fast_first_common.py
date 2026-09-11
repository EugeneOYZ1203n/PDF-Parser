"""The shared body of the `fast_first` pipeline. `fast_first.py` is a
one-liner over this, mirroring `_common.py` / `current.py`'s split.

Skips `Vector_Classification`'s 12-step chain entirely: `extract_vectors` ->
per-Vector FAST filter (`_steps.filter_vectors_fast`, drops a Vector whose
own line/fill geometry doesn't clear the heatmap) -> spatial clustering of
the survivors (`Vector_Classification.group_filters.combine_overlapping_seq`
-- sort by seqno, chain-merge consecutive Vectors by bbox-gap tolerance;
already exactly this shape, reused unmodified) -> `paddle_detect` ->
`drawing`. `paddle_detect`/`drawing` are reused verbatim from the `current`
pipeline's own tail (`_steps.detect_text_paddle` / `_steps.
build_drawing_output`) -- this pipeline ends at PaddleOCR's own detected
boxes, no recognized text, same as `current` on this branch.
"""
from __future__ import annotations

from rastervec.config import FAST_FIRST_SEQ_MERGE_TOLERANCE
from rastervec.pipelines._common import StepTimer
from rastervec.pipelines._steps import (
    build_drawing_output,
    detect_text_paddle,
    extract_native_text,
    extract_vectors,
    filter_vectors_fast,
    read_page,
)
from rastervec.pipelines.result import PipelineResult
from rastervec.Reader.reader import Reader
from rastervec.Vector_Classification.group_filters import combine_overlapping_seq

STEP_NAMES = [
    "read", "native", "vectors", "fast_filter", "spatial_cluster",
    "paddle_detect", "drawing",
]


def run_fast_first_pipeline(
    pdf_path: str, page_index: int, *, enable_fast: bool = True, verbose: bool = False,
    compute=None, progress_counter=None, stop_after: str | None = None,
) -> PipelineResult:
    timer = StepTimer(verbose=verbose)
    page = native = vectors = fast_filter = spatial_clusters = paddle_boxes = drawing = None

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
        if _reached("fast_filter"):
            with timer("fast_filter"):
                fast_filter = filter_vectors_fast(
                    vectors or [], page,
                    enable_fast=enable_fast, verbose=verbose, compute=compute,
                    progress_counter=progress_counter,
                )
        if _reached("spatial_cluster"):
            with timer("spatial_cluster"):
                spatial_clusters, _ = combine_overlapping_seq(
                    [[v] for v in (fast_filter.passed if fast_filter else [])],
                    FAST_FIRST_SEQ_MERGE_TOLERANCE,
                )
        if _reached("paddle_detect"):
            with timer("paddle_detect"):
                paddle_boxes = detect_text_paddle(spatial_clusters or [], page)
        if _reached("drawing"):
            with timer("drawing"):
                drawing = build_drawing_output([], fast_filter.dropped if fast_filter else [])

    # The Reader (and its fitz document) is closed now -- detach the dead page
    # handle so a stale-pointer render crashes loudly at the call site instead.
    if page is not None:
        page.fitz_page = None

    all_texts = list(native or [])

    return PipelineResult(
        page=page,
        texts=all_texts,
        vectors=drawing or [],
        step_durations=timer.durations,
        engine="fast_first",
        paddle_boxes=paddle_boxes,
        # verbose extras
        native_words=(native if verbose else None),
        vectors_raw=(vectors if verbose else None),
        fast_result=(fast_filter.page_result if verbose and fast_filter else None),
        fast_passed=(
            [[v] for v in fast_filter.passed] if verbose and fast_filter else None
        ),
        fast_dropped_vectors=(fast_filter.dropped if verbose and fast_filter else None),
        spatial_clusters=(spatial_clusters if verbose else None),
        step_outputs=(timer.outcomes if verbose else None),
    )
