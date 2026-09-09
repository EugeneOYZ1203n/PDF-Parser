"""OCR sub-pipeline: recognise the words of the small set of representative
clusters similarity+FAST deduped a page's clusters down to (Phase E/F/G),
then restore each representative's words onto every real cluster occurrence
(Phase H).

`recognize_unique_clusters` is a thin per-representative wrapper over
`ocr_backend.recognize_segments` (so `_common.py` reads as one named call
per step, like every other stage); `restore_cluster_texts` is the dedup
payoff -- for every real cluster occurrence recorded in `SegmentMeta`, it
replays *every one* of that occurrence's representative's word-level
`Text`s through the same transform, placing them all onto that occurrence's
real page position/rotation.
"""
from __future__ import annotations

from dataclasses import replace

from rastervec.helpers.geometry import transform_bbox, transform_direction, transform_point
from rastervec.models import Segment, SegmentMeta, Text
from rastervec.OCR.Paddle_OCR.ocr_backend import _recognize_crops_job
from rastervec.OCR.Paddle_OCR.ocr_backend import recognize_segments as _recognize_segments
from rastervec.renderer.stages import render_restore  # noqa: F401 -- re-exported for callers


def recognize_unique_clusters(
    word_segments_by_unique: list[list[Segment]], *, compute=None, progress_counter=None,
) -> list[list[Text]]:
    """Recognises every representative cluster's own word-level `Segment`s,
    returning one inner list of canonical-frame `Text`s per input (same
    order/length as `word_segments_by_unique`). `compute`, when given a
    shared compute-pool proxy (see `Reader/Parallel`), replaces the actual
    engine call with a dispatch to that pool -- the batching/orchestration
    in `recognize_segments` stays local either way. `progress_counter`,
    when given (together with `compute`), is incremented by each batch's
    crop count as that batch's `compute.apply` call returns -- coarser than
    per-crop, but still real, non-blocking-silent progress for the same
    shared counter `Reader/Parallel/pool.py::run_parallel` polls (see
    `OCR/fast_detect.py::detect_tiled`'s own docstring for the full
    picture)."""
    recognize_fn = None
    if compute is not None:
        def recognize_fn(crops):
            result = compute.apply(_recognize_crops_job, (crops,))
            if progress_counter is not None:
                progress_counter.value += len(crops)
            return result
    return [
        _recognize_segments(word_segments, recognize_fn=recognize_fn)
        for word_segments in word_segments_by_unique
    ]


def restore_cluster_texts(
    word_texts_by_unique: list[list[Text]], metas: list[SegmentMeta],
) -> list[Text]:
    """For each `SegmentMeta` (one real cluster occurrence), replays every
    one of `word_texts_by_unique[meta.unique_index]` (that occurrence's
    representative cluster's own word-level `Text`s, in canonical frame)
    through `transform_bbox`/`transform_direction`/`transform_point` by
    `(meta.offset, meta.rotation)` -- the exact inverse of the cluster-level
    canonicalization applied to that occurrence's own Vectors (see
    `models/segment.py::SegmentMeta`'s docstring) -- placing every word onto
    that occurrence's real page position. `page_index`/`seqno` come from the
    meta's own originating cluster occurrence, not the canonical text (which
    carries neither meaningfully)."""
    restored: list[Text] = []
    for meta in metas:
        for canonical in word_texts_by_unique[meta.unique_index]:
            bbox = transform_bbox(canonical.bbox, meta.offset, meta.rotation)
            direction = transform_direction(canonical.direction, meta.rotation)
            origin = transform_point(canonical.origin, meta.offset, meta.rotation)
            restored.append(replace(
                canonical, bbox=bbox, direction=direction, origin=origin,
                page_index=meta.page_index, seqno=meta.seqno,
            ))
    return restored
