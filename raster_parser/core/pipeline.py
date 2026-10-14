"""`run_pipeline` -- one page through the four phases.

    Phase 1  read the page: native text, vectors, embedded images     (always the same)
    Phase 2  images -> vectors + text                                   (`p2`, pluggable)
    Phase 3  Phase 1's vectors -> drawings + text                       (`p3`, pluggable)
    Phase 4  combine everything into one (texts, vectors) pair          (always the same)

Phase 2 and Phase 3 are independent branches: Phase 3 never sees Phase 2's output.

`stop_after` (one of `STEP_NAMES`) stops after that phase: later backends are not called and get
no `step_durations` entry. Phase 4 still runs on whatever ran, so with Phase 3 skipped Phase 1's
raw vectors are the output. It is for tools (the pipeline report); `extract()` always runs
everything.
"""
from __future__ import annotations

import time
from pathlib import Path

from raster_parser.commons.logging_setup import get_logger
from raster_parser.core.registry import DEFAULT_P2, DEFAULT_P3, resolve_p2, resolve_p3
from raster_parser.core.result import PipelineResult
from raster_parser.core.settings import RunSettings
from raster_parser.P1_Reading_Native.phase1 import read_and_extract
from raster_parser.P4_Output_Organization.organize import organize_outputs

_LOG = get_logger("pipeline")

STEP_NAMES = ("phase1", "phase2", "phase3")


def run_pipeline(
    pdf_path: str | Path,
    page_index: int = 0,
    *,
    p2: str = DEFAULT_P2,
    p3: str = DEFAULT_P3,
    settings: RunSettings | None = None,
    stop_after: str | None = None,
) -> PipelineResult:
    settings = settings or RunSettings()
    phase2 = resolve_p2(p2)  # fail on a bad name before any work
    phase3 = resolve_p3(p3)
    if stop_after is not None and stop_after not in STEP_NAMES:
        raise ValueError(f"stop_after must be one of {STEP_NAMES} or None, got {stop_after!r}")
    last = STEP_NAMES.index(stop_after) if stop_after else len(STEP_NAMES) - 1
    durations: dict[str, float] = {}

    start = time.perf_counter()
    phase1 = read_and_extract(str(pdf_path), page_index)
    durations["phase1"] = time.perf_counter() - start

    vectors_p2, texts_p2 = [], []
    if last >= 1:
        start = time.perf_counter()
        vectors_p2, texts_p2 = phase2(phase1.images, phase1.page, settings)
        durations["phase2"] = time.perf_counter() - start

    vectors_p3, texts_p3 = phase1.vectors, []
    if last >= 2:
        start = time.perf_counter()
        vectors_p3, texts_p3 = phase3(phase1.vectors, phase1.page, settings)
        durations["phase3"] = time.perf_counter() - start

    start = time.perf_counter()
    texts, vectors = organize_outputs(
        phase1.texts, texts_p2, texts_p3, vectors_p2, vectors_p3, phase1.page,
    )
    durations["phase4"] = time.perf_counter() - start

    _LOG.info(
        "page %d: p2=%s p3=%s -> %d texts, %d vectors (%s)", page_index, p2, p3, len(texts),
        len(vectors), ", ".join(f"{k} {v:.2f}s" for k, v in durations.items()),
    )
    return PipelineResult(
        page=phase1.page, texts=texts, vectors=vectors, p2=p2, p3=p3, step_durations=durations,
    )
