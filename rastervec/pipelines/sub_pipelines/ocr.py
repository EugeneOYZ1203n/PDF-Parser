"""OCR sub-pipeline: recognise every elected representative word (Phase F's
`elect_unique_segments` dedup output), then restore each representative's
`Text` onto every real word occurrence it covers (Phase H).

`recognize_unique_words` is a thin wrapper over `ocr_backend.
recognize_segments` (so `_common.py` reads as one named call per step, like
every other stage); `restore_word_texts` is the dedup payoff -- for every
real word occurrence recorded in `SegmentMeta`, it replays that occurrence's
representative's `Text` through the same transform, placing it onto that
occurrence's real page position/rotation.
"""
from __future__ import annotations

from dataclasses import replace

from rastervec.helpers.geometry import transform_bbox, transform_direction, transform_point
from rastervec.models import Segment, SegmentMeta, Text
from rastervec.OCR.Paddle_OCR.ocr_backend import _recognize_crops_job
from rastervec.OCR.Paddle_OCR.ocr_backend import recognize_segments as _recognize_segments
from rastervec.renderer.stages import render_restore  # noqa: F401 -- re-exported for callers


def recognize_unique_words(
    uniques: list[Segment], *, compute=None, progress_counter=None,
) -> list[Text]:
    """Recognises every elected representative's own word-level `Segment`,
    returning one canonical-frame `Text` per input (same order/length as
    `uniques`). `compute`, when given a shared compute-pool proxy (see
    `Reader/Parallel`), replaces the actual engine call with a dispatch to
    that pool -- the batching/orchestration in `recognize_segments` stays
    local either way. `progress_counter`, when given (together with
    `compute`), is incremented by each batch's crop count as that batch's
    `compute.apply` call returns -- coarser than per-crop, but still real,
    non-blocking-silent progress for the same shared counter
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
    return _recognize_segments(uniques, recognize_fn=recognize_fn)


def restore_word_texts(unique_texts: list[Text], metas: list[SegmentMeta]) -> list[Text]:
    """For each `SegmentMeta` (one real word occurrence), replays
    `unique_texts[meta.unique_index]` (that occurrence's representative's
    own canonical-frame `Text`) through `transform_bbox`/
    `transform_direction`/`transform_point` by `(meta.offset, meta.rotation)`
    -- the exact inverse of the canonicalization applied to that
    occurrence's own Vectors (see `models/segment.py::SegmentMeta`'s
    docstring) -- placing it onto that occurrence's real page position.
    `page_index`/`seqno` come from the meta's own originating occurrence,
    not the canonical text (which carries neither meaningfully)."""
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
