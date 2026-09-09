"""Radon-segmentation-through-OCR handoff types.

A `Segment` is one Radon-segmented word at its real page position. Segments
are grouped by shape similarity (rotation/translation-normalized using each
segment's own precise `angle` -- never a page-wide search); one member per
group is elected representative and its normalized `vectors` becomes a
`UniqueSegment` -- the only thing actually rendered and OCR'd. Every other
member of a passing group keeps only a `SegmentMeta` (offset + rotation back
to its own real position), so the single `Text` OCR produces for the
`UniqueSegment` can be duplicated and repositioned for every real occurrence
without re-rendering or re-recognizing it. See
`docs/PIPELINE.md` for the worked dedup example.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rastervec.models.vector import Vector


@dataclass
class Segment:
    vectors: "list[Vector]"
    angle: float  # Radon's own precise skew estimate, degrees -- NEVER rounded
    # to a quarter turn anywhere in this pipeline.


@dataclass
class UniqueSegment:
    """The one representative per similarity group that gets rendered +
    OCR'd. `vectors` are `transform_vector`'d to a canonical frame:
    translated so the segment's own bbox origin is (0, 0), rotated by
    `-angle` so it's upright -- no absolute position/rotation needs
    tracking past this point."""

    vectors: "list[Vector]"


@dataclass
class SegmentMeta:
    """One entry per original segment instance in a passing similarity
    group (including the unique/representative segment's own instance --
    it is restored via this same math, from its own `offset`/`rotation`,
    not hardcoded to identity). `rotation` is that instance's own Radon
    `angle`; `offset` is that instance's own post-rotation bbox origin,
    itself rotated forward by `rotation` -- i.e. exactly the `(offset,
    rotation_deg)` pair `helpers.geometry.transform_vector`/`transform_bbox`
    expect to invert the Phase-E canonical-frame normalization and place a
    `UniqueSegment`'s eventual `Text` back onto this instance's real page
    position: `transform_bbox(canonical_bbox, offset=meta.offset,
    rotation_deg=meta.rotation)` (see `pipelines/sub_pipelines/ocr.py::
    restore_segment_texts`). `page_index`/`seqno` are carried here (rather
    than re-derived from discarded original vectors) since a
    non-representative instance's real Vectors are never kept past the
    FAST-detection step."""

    unique_index: int
    offset: tuple[float, float]
    rotation: float
    page_index: int
    seqno: int
