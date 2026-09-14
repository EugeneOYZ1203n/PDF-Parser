"""`PipelineResult` -- the one value every pipeline in this package
returns. Final-output fields are always populated; intermediate fields are
left `None` unless the run was `verbose=True`.

Also the new home for the small types that used to live in the old
`pipeline.py`: `SeparationKey`, `FastPageResult`.

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

    from rastervec.commons.models import Page, Segment, Text, Vector
    from rastervec.OCR.Paddle_OCR.ocr_backend import ClusterDetection
    from rastervec.pipelines._steps import ReclassifyResult
    from rastervec.Vector_Similarity.similarity import SimilarityGroup

# (layer, color, width) -- one pipeline separation bucket.
SeparationKey = "tuple[str, tuple | None, float | None]"


@dataclass
class FastPageResult:
    """`filter_vectors_fast`'s whole-page result. `scores` is keyed by a
    Vector's own index into that step's `vectors` input list -- each Vector
    is scored and kept/dropped independently, before any clustering
    exists."""

    page_image: "Image.Image | None"  # verbose only -- downsampled by
    # `debug_image_scale` for report visualization, NOT what FastDetector
    # actually scored (that happens at full resolution inside
    # filter_vectors_fast, before this downsample)
    page_mask: "np.ndarray | None"
    detect_seconds: float | None
    scores: dict[int, float]
    # verbose-only tiling diagnostics (see pipelines/_steps.filter_vectors_fast):
    # `skipped_tiles` are page-space bboxes of tiles that overlapped no text
    # candidate and were never run; `all_tiles` is every tile in the grid
    # (run or skipped); `tile_count` is len(all_tiles); `tile_seconds` is
    # per-run-tile wall time (local runs only, else []).
    skipped_tiles: "list[tuple[float, float, float, float]] | None" = None
    all_tiles: "list[tuple[float, float, float, float]] | None" = None
    tile_count: int | None = None
    tile_seconds: "list[float] | None" = None
    # downsample factor applied to `page_image` relative to the full-res
    # render FastDetector actually scored (1.0 when not verbose, or when no
    # downsample was applied).
    debug_image_scale: float = 1.0


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
    similarity_groups: "list[SimilarityGroup] | None" = None  # vector_similarity_group's
    # output over every raw extracted Vector, run before FAST
    fast_result: FastPageResult | None = None  # filter_vectors_fast's
    # page-level result (per-Vector scores)
    fast_passed: "list[list[Vector]] | None" = None  # every passed Vector,
    # each wrapped as its own singleton list (render-layer compatibility)
    fast_dropped_vectors: "list[Vector] | None" = None
    reclassify_result: "ReclassifyResult | None" = None  # FAST's per-vector
    # verdict pulled up to a similarity-group consensus (fail -> pass only)
    separation_buckets: "list[list[Vector]] | None" = None  # one list per
    # (layer, color, width) bucket -- see `SeparationKey`
    spatial_clusters: "list[list[Vector]] | None" = None  # one per
    # (layer, color, width) bucket's union-find spatial merge, flattened
    cluster_detections: "list[ClusterDetection | None] | None" = None  # one
    # entry per spatial_clusters entry, same index -- each carries the
    # cluster's own one render (`.image`/`.dpi`, what PaddleOCR's detector
    # saw) plus its `PaddleDetection`s; `None` for an empty cluster
    reassigned_drawing: "list[Vector] | None" = None  # FAST passed these,
    # but no PaddleOCR detection was backed by them -- reassigned to drawing
    reassigned_text: "list[Vector] | None" = None  # the complementary half:
    # every Vector actually assigned to a PaddleOCR detection
    rotated_segments: "list[Segment] | None" = None  # one Segment per
    # PaddleOCR detection with an assigned vector -- no word/character
    # splitting, "just rotated clusters" (see `_steps.rotate_paddle_
    # detections`); recognition's own input
    rotation_debug: list | None = None  # one dict per detection, page
    # space: detection_bbox, resolved_theta, source_theta
    restored_texts: "list[Text] | None" = None
    step_outputs: dict | None = None

    @contextmanager
    def open_page(self) -> "Iterator[Page]":
        """Reopen the source PDF and yield a live `models.Page` for this run's
        page. The pipeline closes its `Reader` before returning, so
        `self.page.fitz_page` is `None`; use this whenever you need to
        rasterize or otherwise call into `fitz`."""
        from rastervec.P1_Reading_Native.reader import Reader

        with Reader(self.page.doc_path) as reader:
            yield reader.get_page(self.page.meta.index)
