"""The current extraction pipeline.

    native           = extract_native_text(page)                      # list[Text]
    vectors          = extract_vectors(page)                          # list[Vector]
    classification   = classify_vectors(vectors, page)                # tiered text clusters + drawing drops
    cluster_segments = build_cluster_candidates(flat_clusters)        # PCA angle estimate -> list[Segment]
    groups           = group_similar_segments(cluster_segments)       # shape dedup -> list[list[int]]
    fast             = detect_text_fast(cluster_segments, groups, page)  # all-must-pass -> UniqueSegments
                                                                          # (one representative CLUSTER each)
                                                                          # + SegmentMetas (real occurrences)
    word_segments    = segment_unique_clusters(fast.uniques)          # Radon, representatives only -> per-unique
                                                                       # list[Segment] (word-level, w/ .image)
    unique_texts     = recognize_unique_clusters(word_segments)       # per-unique list[Text]
    restored         = restore_cluster_texts(unique_texts, ...)       # every word, onto every real occurrence
    drawing_vectors  = build_drawing_output(...)

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
    compute=None, progress_counter=None,
) -> PipelineResult:
    """Run the current pipeline on one page. `verbose=True` also retains
    every intermediate value on the returned `PipelineResult`. `compute`,
    when given a `multiprocessing.managers.SyncManager`-hosted `Pool`
    proxy (see `Reader/Parallel`), dispatches FAST tile detection and OCR
    crop recognition to that shared pool instead of running them locally
    -- `None` (the default) preserves today's fully-local behavior.
    `progress_counter`, when given a shared counter (see `Reader/Parallel/
    pool.py::run_parallel`), is incremented as FAST tiles / OCR crop
    batches complete instead of driving a local `tqdm` bar -- `None` (the
    default) preserves today's local-tqdm-or-nothing behavior."""
    return run_current_pipeline(
        pdf_path, page_index, enable_fast=enable_fast, verbose=verbose, compute=compute,
        progress_counter=progress_counter,
    )


if __name__ == "__main__":
    from rastervec.pipelines._cli import main

    raise SystemExit(main("current", sys.argv[1:]))
