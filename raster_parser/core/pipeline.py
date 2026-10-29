"""`run_pipeline` -- one page through the four phases.

    native_read       read the page: native text, vectors, embedded images   (always the same)
    raster_vectorize  images -> vectors + text                               (`p2`, pluggable)
    vector_classify   native_read's vectors -> drawings + text               (`p3`, pluggable)
    output_assemble   combine everything into one (texts, vectors) pair      (always the same)

raster_vectorize and vector_classify are independent branches: vector_classify never sees
raster_vectorize's output. `PHASE_NAMES` are also the `step_durations` keys.

`stop_after` (one of `STEP_NAMES`) stops after that phase: later backends are not called and get
no `step_durations` entry. output_assemble still runs on whatever ran, so with vector_classify
skipped native_read's raw vectors are the output. It is for tools (the pipeline report);
`extract()` always runs everything.
"""

from __future__ import annotations

import time
from pathlib import Path

from raster_parser.commons.logging_setup import get_logger
from raster_parser.models.result import ExtractResult
from raster_parser.commons.progress import note, stage
from raster_parser.core.registry import (
    DEFAULT_VECTORIZE,
    DEFAULT_CLASSIFY,
    resolve_vectorize,
    resolve_classify,
)
from raster_parser.models.settings import RunSettings
from raster_parser.phase_native_read.native_read import read_and_extract
from raster_parser.phase_output_assemble.organize import organize_outputs

_LOG = get_logger("pipeline")

PHASE_NAMES = ("native_read", "raster_vectorize", "vector_classify", "output_assemble")
STEP_NAMES = PHASE_NAMES[:-1]  # the phases `stop_after` can name


def run_pipeline(
    pdf_path: str | Path,
    page_index: int = 0,
    *,
    p2: str = DEFAULT_VECTORIZE,
    p3: str = DEFAULT_CLASSIFY,
    settings: RunSettings | None = None,
    stop_after: str | None = None,
) -> ExtractResult:
    settings = settings or RunSettings()
    vectorize = resolve_vectorize(p2)  # fail on a bad name before any work
    classify = resolve_classify(p3)
    if stop_after is not None and stop_after not in STEP_NAMES:
        raise ValueError(f"stop_after must be one of {STEP_NAMES} or None, got {stop_after!r}")
    last = STEP_NAMES.index(stop_after) if stop_after else len(STEP_NAMES) - 1
    durations: dict[str, float] = {}

    tag = f"{Path(pdf_path).name} p{page_index}"
    start = time.perf_counter()
    with stage(f"{tag} › native_read"):
        native = read_and_extract(str(pdf_path), page_index)
    durations["native_read"] = time.perf_counter() - start

    vectors_p2, texts_p2 = [], []
    if last >= 1:
        start = time.perf_counter()
        with stage(f"{tag} › raster_vectorize {p2} ({len(native.images)} images)"):
            vectors_p2, texts_p2 = vectorize(native.images, native.page, settings)
        durations["raster_vectorize"] = time.perf_counter() - start
    else:
        note(f"- {tag} › raster_vectorize {p2} skipped (stop_after={stop_after})")

    vectors_p3, texts_p3 = native.vectors, []
    if last >= 2:
        start = time.perf_counter()
        with stage(f"{tag} › vector_classify {p3} ({len(native.vectors)} vectors)"):
            vectors_p3, texts_p3 = classify(native.vectors, native.page, settings)
        durations["vector_classify"] = time.perf_counter() - start
    else:
        note(f"- {tag} › vector_classify {p3} skipped (stop_after={stop_after})")

    start = time.perf_counter()
    with stage(f"{tag} › output_assemble"):
        texts, vectors = organize_outputs(
            native.texts,
            texts_p2,
            texts_p3,
            vectors_p2,
            vectors_p3,
            native.page,
        )
    durations["output_assemble"] = time.perf_counter() - start

    _LOG.info(
        "page %d: p2=%s p3=%s -> %d texts, %d vectors (%s)",
        page_index,
        p2,
        p3,
        len(texts),
        len(vectors),
        ", ".join(f"{k} {v:.2f}s" for k, v in durations.items()),
    )
    return ExtractResult(
        page=native.page.meta, texts=texts, vectors=vectors, step_durations=durations
    )
