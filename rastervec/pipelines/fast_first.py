"""`fast_first` pipeline: skips `Vector_Classification`'s 12-step chain
entirely in favor of per-Vector FAST filtering followed by a plain
seqno-consecutive spatial merge.

    native          = extract_native_text(page)                       # list[Text]
    vectors         = extract_vectors(page)                           # list[Vector]
    fast_filter     = filter_vectors_fast(vectors, page)               # per-Vector,
                                                                        # independent -> passed + dropped
    spatial_clusters, _ = combine_overlapping_seq(                     # seqno-sorted,
        [[v] for v in fast_filter.passed], FAST_FIRST_SEQ_MERGE_TOLERANCE)  # consecutive bbox-gap merge
    paddle_boxes    = detect_text_paddle(spatial_clusters, page)       # PaddleOCR's own
                                                                        # detector -> page-space bboxes, no text
    drawing_vectors = build_drawing_output([], fast_filter.dropped)

See `_fast_first_common.run_fast_first_pipeline` for the real block sequence
and `_steps.py` / `Vector_Classification/group_filters.py` for each call.

CLI: `python -m rastervec.pipelines.fast_first --pdf PATH --page N [-v] [--no-fast]`
"""
from __future__ import annotations

import sys

from rastervec.pipelines._fast_first_common import STEP_NAMES, run_fast_first_pipeline
from rastervec.pipelines.result import PipelineResult

__all__ = ["run_pipeline", "run_fast_first_pipeline", "STEP_NAMES", "PipelineResult"]


def run_pipeline(
    pdf_path: str, page_index: int = 0, *, enable_fast: bool = True, verbose: bool = False,
    compute=None, progress_counter=None, stop_after: str | None = None,
) -> PipelineResult:
    """Run the `fast_first` pipeline on one page. `verbose=True` also
    retains every intermediate value on the returned `PipelineResult`.
    `compute`/`progress_counter`/`stop_after` behave exactly as in
    `rastervec.pipelines.current.run_pipeline` -- see that module's own
    docstring."""
    return run_fast_first_pipeline(
        pdf_path, page_index, enable_fast=enable_fast, verbose=verbose, compute=compute,
        progress_counter=progress_counter, stop_after=stop_after,
    )


if __name__ == "__main__":
    from rastervec.pipelines._cli import main

    raise SystemExit(main("fast_first", sys.argv[1:]))
