"""The legacy pipeline: a thin wrapper that runs archive's unmodified
`raster_parser` pipeline and reshapes its output into a `PipelineResult`
so the benchmark can score it beside `current`.

Only the OCR-scored fields are meaningful here -- archive's `TextDTO`
carries no back-reference to source geometry, so `text_clusters`,
`clustering`, etc. stay empty.
"""
from __future__ import annotations

import sys
import time

from rastervec.pipelines.result import PipelineResult
from rastervec.Reader.reader import Reader

__all__ = ["run_pipeline"]


def run_pipeline(
    pdf_path: str, page_index: int = 0, *, enable_raster_pass: bool = False,
    verbose: bool = False,
) -> PipelineResult:
    from rastervec.Evaluation.Evaluate.legacy_adapter import (
        run_archive_pipeline,
        to_cluster_ocr_results,
    )

    start = time.perf_counter()
    elements = run_archive_pipeline(
        pdf_path, page_index, enable_raster_pass=enable_raster_pass,
    )
    elapsed = time.perf_counter() - start

    cor = to_cluster_ocr_results(elements, page_index=page_index)
    with Reader(pdf_path) as reader:
        page = reader.get_page(page_index)  # meta snapshot; fitz_page unused downstream

    return PipelineResult(
        page=page,
        native_words=[],
        drawing_vectors=[],
        ocr_results=[c.resolved for c in cor],
        cluster_ocr_results=cor,
        text_clusters=[],
        regrouped_clusters=[],
        clustering={},
        cluster_groups={},
        fast_dropped=[],
        ocr_failed=[],
        step_durations={"legacy": elapsed},
        engine="legacy",
    )


if __name__ == "__main__":
    from rastervec.pipelines._cli import main

    raise SystemExit(main("legacy", sys.argv[1:]))
