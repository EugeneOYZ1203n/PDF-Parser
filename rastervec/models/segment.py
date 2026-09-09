"""Dedup-through-OCR handoff types.

A `Segment` is one word occurrence at its real page position, produced by
`OCR/radon.py::segment_clusters` on a FAST-surviving classification
cluster -- `vectors` are that word's own `Vector`s, `angle` Radon's precise
residual skew (never rounded), `image` the word's own deskewed pixel crop
(captured directly during segmentation, so OCR never has to re-render).
This is the only `Segment` lifecycle in this pipeline: FAST (Phase D) runs
directly on plain `list[list[Vector]]` clusters *before* Radon, so it never
needs a rotation estimate at all -- there is no more pre-Radon,
PCA-estimated `Segment` lifecycle.

Segments are grouped by shape similarity (rotation/translation-normalized
using each segment's own `angle`); one member per group is elected
representative (`pipelines/_steps.py::elect_unique_segments`) and
canonicalized into a zero-angle `Segment` (`vectors` translation+rotation-
normalized, `image` carried over unchanged from the real occurrence it came
from -- already upright, no re-render needed). Every other member of a
group keeps only a `SegmentMeta` (offset + rotation back to its own real
position), so a single OCR result can be duplicated and repositioned for
every real occurrence without re-recognizing it. See `docs/PIPELINE.md` for
the full data flow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from rastervec.models.vector import Vector


@dataclass
class Segment:
    vectors: "list[Vector]"
    angle: float  # Radon's precise rotation, degrees -- NEVER rounded or
    # snapped to a quarter turn anywhere in this pipeline. 0.0 for an
    # elected representative's canonicalized copy (see `elect_unique_
    # segments`), since canonicalization already removed its own rotation.
    image: "np.ndarray | None" = None  # that word's own deskewed crop,
    # straight from segmentation (or carried over unchanged for an elected
    # representative's canonical copy).


@dataclass
class SegmentMeta:
    """One entry per real word occurrence in a passing similarity group
    (including the elected representative's own occurrence -- it is
    restored via this same math, from its own `offset`/`rotation`, not
    hardcoded to identity). `rotation` is that occurrence's own `angle`
    (Radon's precise skew); `offset` is that occurrence's own post-rotation
    bbox origin, itself rotated forward by `rotation` -- i.e. exactly the
    `(offset, rotation_deg)` pair `helpers.geometry.transform_vector`/
    `transform_bbox` expect to invert the canonical-frame normalization and
    place a representative's eventual OCR result back onto this occurrence's
    real page position: `transform_bbox(canonical_bbox, offset=meta.offset,
    rotation_deg=meta.rotation)` (see `pipelines/sub_pipelines/ocr.py::
    restore_word_texts`, which replays the representative's word-level
    `Text` through this same transform for each `SegmentMeta`).
    `page_index`/`seqno` are carried here (rather than re-derived from
    discarded original vectors) since a non-representative occurrence's real
    `Vector`s are never kept past `elect_unique_segments`."""

    unique_index: int
    offset: tuple[float, float]
    rotation: float
    page_index: int
    seqno: int
