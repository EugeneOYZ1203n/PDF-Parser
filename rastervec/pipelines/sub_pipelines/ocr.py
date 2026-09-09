"""OCR sub-pipeline: recognise the small set of `UniqueSegment`s a page's
segments deduped down to (Phase E/F), then restore one `Text` per original
segment occurrence (Phase H).

`recognize` is a thin wrapper over `ocr_backend.recognize_unique_segments`
(so `_common.py` reads as one named call per step, like every other stage);
`restore_segment_texts` is the dedup payoff -- it duplicates each
`UniqueSegment`'s single OCR `Text` across every real occurrence recorded
in `SegmentMeta`, transforming its canonical-frame geometry back onto that
occurrence's real page position/rotation.
"""
from __future__ import annotations

from dataclasses import replace

from rastervec.helpers.geometry import transform_bbox, transform_direction, transform_point
from rastervec.models import SegmentMeta, Text, UniqueSegment
from rastervec.OCR.Paddle_OCR.ocr_backend import _recognize_crops_job
from rastervec.OCR.Paddle_OCR.ocr_backend import recognize_unique_segments as _recognize_unique_segments
from rastervec.renderer.stages import render_restore  # noqa: F401 -- re-exported for callers


def recognize(
    uniques: list[UniqueSegment], *, compute=None, progress_counter=None,
) -> list[Text]:
    """Recognises every `UniqueSegment`, returning one canonical-frame
    `Text` per input (same order). `compute`, when given a shared
    compute-pool proxy (see `Reader/Parallel`), replaces the actual engine
    call with a dispatch to that pool -- the batching/orchestration in
    `recognize_unique_segments` stays local either way. `progress_counter`,
    when given (together with `compute`), is incremented by each batch's
    crop count as that batch's `compute.apply` call returns -- coarser than
    per-crop (one increment per similarity-group batch, not per crop), but
    still real, non-blocking-silent progress for the same shared counter
    `Reader/Parallel/pool.py::run_parallel` polls (see
    `OCR/fast_detect.py::detect_tiled`'s own docstring for the full
    picture)."""
    recognize_fn = None
    if compute is not None:
        def recognize_fn(crops):
            result = compute.apply(_recognize_crops_job, (crops,))
            if progress_counter is not None:
                progress_counter.value += len(crops)
            return result
    return _recognize_unique_segments(uniques, recognize_fn=recognize_fn)


def restore_segment_texts(unique_texts: list[Text], metas: list[SegmentMeta]) -> list[Text]:
    """For each `SegmentMeta`, takes `unique_texts[meta.unique_index]` (a
    canonical-frame `Text`) and transforms its `bbox`/`origin`/`direction`
    by `(meta.offset, meta.rotation)` -- the exact inverse of the Phase-E
    normalization `transform_vector` applied to that occurrence's own
    Vectors (see `models/segment.py::SegmentMeta`'s docstring) -- to place
    it back onto that occurrence's real page position. `page_index`/`seqno`
    come from the meta's own originating segment, not the canonical text
    (which carries neither meaningfully)."""
    restored: list[Text] = []
    for meta in metas:
        canonical = unique_texts[meta.unique_index]
        bbox = transform_bbox(canonical.bbox, meta.offset, meta.rotation)
        direction = transform_direction(canonical.direction, meta.rotation)
        origin = transform_point(canonical.origin, meta.offset, meta.rotation)
        restored.append(replace(
            canonical, bbox=bbox, direction=direction, origin=origin,
            page_index=meta.page_index, seqno=meta.seqno,
        ))
    return restored
