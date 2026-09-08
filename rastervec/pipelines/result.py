"""`PipelineResult` -- the one value every pipeline in this package
returns. Final-output fields are always populated; intermediate fields are
left `None` unless the run was `verbose=True`.

Also the new home for the small types that used to live in the old
`pipeline.py`: `GroupKey`, `ClusteringStageResult`, `FastPageResult`.

NOTE: the pipeline closes its `Reader` before returning, so `page.fitz_page`
is `None` on a returned result -- use `PipelineResult.open_page()` to reopen
the source PDF and get a live `fitz.Page` for rasterization. A
`PipelineResult` still holds numpy arrays / PIL images and must never be
pickled / sent across a process boundary. The benchmark's per-page job reads
what it needs off the result and returns only plain data (`PageResult`).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image

    from rastervec.models import (
        ClusterOcrResult,
        DrawingVector,
        Page,
        TextVectorResult,
        TextWord,
        VectorPath,
        VectorRecord,
    )
    from rastervec.output_types import NativePDFElements
    from rastervec.OCR.radon import ClusterSegmentation
    from rastervec.Vector_Classification.classification import StepResult

# (layer, color) -- one Vector.separate_by_color() bucket.
GroupKey = "tuple[str, tuple]"


@dataclass
class ClusteringStageResult:
    """One (layer, color) bucket's Vector Classification result: `steps` is
    exactly `classify_bucket()` / the old `classification.cluster()`'s
    return value. `steps[-1].categories["kept"]` is the final surviving
    clusters; every `role="dropped"` category across every step is drawing
    content."""

    steps: "list[StepResult]"


@dataclass
class FastPageResult:
    """FAST text detection's whole-page result (see
    `pipelines/_steps.detect_text_fast`)."""

    page_image: "Image.Image | None"
    page_mask: "np.ndarray | None"
    detect_seconds: float | None
    scores: dict
    passed: "list[list[VectorPath]]"
    dropped: "list[list[VectorPath]]"


@dataclass
class StepOutcome:
    """Per-step status, verbose-runs only (replaces the old `StageOutput`)."""

    name: str
    status: str  # "ok" | "error"
    error: str | None = None
    duration_seconds: float | None = None


@dataclass
class PipelineResult:
    # ---- always populated -------------------------------------------------
    page: "Page"  # `page.fitz_page` is None -- use `open_page()` for a live one
    native_words: "list[TextWord]"
    drawing_vectors: "list[DrawingVector]"
    ocr_results: "list[TextVectorResult]"
    cluster_ocr_results: "list[ClusterOcrResult]"
    text_clusters: "list[list[VectorPath]]"
    regrouped_clusters: "list[list[VectorPath]]"
    clustering: dict
    cluster_groups: dict
    fast_dropped: "list[list[VectorPath]]"
    ocr_failed: "list[list[VectorPath]]"
    step_durations: dict
    engine: str  # "current" | "legacy"

    # ---- verbose only (None unless verbose=True) -------------------------
    vector_paths: "list[VectorPath] | None" = None
    vector_records: "list[VectorRecord] | None" = None
    paths_by_layer: dict | None = None
    paths_by_layer_color: dict | None = None
    text_candidate_records: "list[VectorRecord] | None" = None
    similarity_groups: "list[list[list[VectorPath]]] | None" = None
    cluster_similarity_id: dict | None = None
    fast_passed: "list[list[VectorPath]] | None" = None
    fast_result: FastPageResult | None = None
    regrouped_cluster_similarity_id: dict | None = None
    segmentations: "list[ClusterSegmentation] | None" = None
    step_outputs: dict | None = None

    @contextmanager
    def open_page(self) -> "Iterator[Page]":
        """Reopen the source PDF and yield a live `models.Page` for this run's
        page. The pipeline closes its `Reader` before returning, so
        `self.page.fitz_page` is `None`; use this whenever you need to
        rasterize or otherwise call into `fitz`."""
        from rastervec.Reader.reader import Reader

        with Reader(self.page.doc_path) as reader:
            yield reader.get_page(self.page.meta.index)

    def to_native_pdf_elements(self) -> "NativePDFElements":
        """Serialization/export boundary -- standardized output_types.py
        DTOs from this run's native_words/drawing_vectors."""
        from rastervec.output_types import NativePDFElements

        return NativePDFElements.from_extract(
            words=self.native_words or [], drawings=self.drawing_vectors or [],
        )
