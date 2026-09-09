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

    from rastervec.models import Page, Segment, SegmentMeta, Text, UniqueSegment, Vector
    from rastervec.Vector_Classification.classification import StepResult

# (layer, color) -- one Vector.separate_by_color() bucket.
GroupKey = "tuple[str, tuple]"


@dataclass
class ClusteringStageResult:
    """One (layer, color) bucket's Vector Classification result: `steps` is
    exactly `classify_bucket()` / `classification.cluster()`'s return
    value. `steps[-1].categories["kept"]` is the final surviving (tiered)
    clusters; every `role="dropped"` category across every step is drawing
    content."""

    steps: "list[StepResult]"


@dataclass
class FastPageResult:
    """FAST text detection's whole-page result (see
    `pipelines/_steps.detect_text_fast`). `scores` is keyed by a cluster
    similarity group's own index into that step's `groups` list (the
    group's combined, min-across-members score)."""

    page_image: "Image.Image | None"
    page_mask: "np.ndarray | None"
    detect_seconds: float | None
    scores: dict[int, float]


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
    texts: "list[Text]"  # native + restored OCR text, flat
    vectors: "list[Vector]"  # drawing content: classification + FAST drops, flat
    step_durations: dict
    engine: str  # "current" | "legacy"

    # ---- verbose only (None unless verbose=True) -------------------------
    native_words: "list[Text] | None" = None
    vectors_raw: "list[Vector] | None" = None
    vectors_by_layer: dict | None = None
    vectors_by_layer_color: dict | None = None
    text_clusters: "list[list[list[Vector]]] | None" = None
    clustering: dict | None = None
    classification_dropped: "list[Vector] | None" = None
    cluster_segments: "list[Segment] | None" = None  # pre-Radon, PCA-angle,
    # one per surviving classification cluster -- similarity+FAST's input
    similarity_groups: "list[list[int]] | None" = None
    fast_result: FastPageResult | None = None
    unique_segments: "list[UniqueSegment] | None" = None  # one whole
    # representative CLUSTER per passing similarity group (its vectors)
    segment_metas: "list[SegmentMeta] | None" = None  # one per real cluster
    # occurrence (including the representative's own)
    fast_dropped_vectors: "list[Vector] | None" = None
    word_segments: "list[list[Segment]] | None" = None  # Radon's output,
    # representatives only -- word-level Segments (with `.image`), one
    # inner list per `unique_segments` entry, same order
    unique_texts: "list[list[Text]] | None" = None  # one inner list per
    # `unique_segments`/`word_segments` entry -- that representative's own
    # word-level OCR Texts, in canonical frame
    restored_texts: "list[Text] | None" = None
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
