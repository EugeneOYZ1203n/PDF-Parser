"""**Test branch (`test/paddle-detect-post-fast`)**: everything after FAST
is replaced by a single PaddleOCR text-DETECTION call -- no Radon
segmentation, no similarity dedup, no recognition. See `_steps.py::
detect_text_paddle` and `OCR/Paddle_OCR/ocr_backend.py::PaddleDetectBackend`.

    native           = extract_native_text(page)                      # list[Text]
    vectors          = extract_vectors(page)                          # list[Vector]
    classification   = classify_vectors(vectors, page)                # tiered text clusters + drawing drops
    fast             = detect_text_fast(flat_clusters, page)          # per-cluster, independent -> passed clusters
                                                                       # + dropped_vectors
    paddle_boxes     = detect_text_paddle(fast.passed, page)          # whole-page render -> PaddleOCR's own
                                                                       # detector -> page-space bboxes, no text
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
    compute=None, progress_counter=None, stop_after: str | None = None,
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
    default) preserves today's local-tqdm-or-nothing behavior. `stop_after`,
    when given one of `STEP_NAMES`, skips every step after it (its
    `PipelineResult` fields stay `None`) -- for stage-report runs that only
    need the earlier stages."""
    return run_current_pipeline(
        pdf_path, page_index, enable_fast=enable_fast, verbose=verbose, compute=compute,
        progress_counter=progress_counter, stop_after=stop_after,
    )


if __name__ == "__main__":
    from rastervec.pipelines._cli import main

    raise SystemExit(main("current", sys.argv[1:]))
