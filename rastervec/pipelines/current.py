"""The current extraction pipeline.

    native          = extract_native_text(page)                # list[Text]
    vectors         = extract_vectors(page)                     # list[Vector]
    classification  = classify_vectors(vectors, page)           # tiered text clusters + drawing drops
    segments        = segment_clusters(flat_clusters)           # Radon deskew + word split -> list[Segment]
    groups          = group_similar_segments(segments)          # rotation-exact shape dedup -> list[list[int]]
    fast            = detect_text_fast(segments, groups, page)  # all-must-pass -> UniqueSegments + SegmentMetas
    unique_texts    = recognize(fast.uniques)                   # one OCR Text per UniqueSegment
    restored        = restore_segment_texts(unique_texts, ...)  # duplicated back onto every real occurrence
    drawing_vectors = build_drawing_output(...)

See `_common.run_current_pipeline` for the real block sequence and
`_steps.py` / `sub_pipelines/` for each call.

CLI: `python -m rastervec.pipelines.current --pdf PATH --page N [-v] [--no-fast]`
"""
from __future__ import annotations

import sys

from rastervec.pipelines._common import STEP_NAMES, run_current_pipeline
from rastervec.pipelines.result import PipelineResult

__all__ = ["run_pipeline", "run_current_pipeline", "STEP_NAMES", "PipelineResult"]


def run_pipeline(
    pdf_path: str, page_index: int = 0, *, enable_fast: bool = True, verbose: bool = False,
    compute=None,
) -> PipelineResult:
    """Run the current pipeline on one page. `verbose=True` also retains
    every intermediate value on the returned `PipelineResult`. `compute`,
    when given a `multiprocessing.managers.SyncManager`-hosted `Pool`
    proxy (see `Reader/Parallel`), dispatches FAST tile detection and OCR
    crop recognition to that shared pool instead of running them locally
    -- `None` (the default) preserves today's fully-local behavior."""
    return run_current_pipeline(
        pdf_path, page_index, enable_fast=enable_fast, verbose=verbose, compute=compute,
    )


if __name__ == "__main__":
    from rastervec.pipelines._cli import main

    raise SystemExit(main("current", sys.argv[1:]))
