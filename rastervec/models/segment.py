"""Dedup-through-OCR handoff types, shared by two dedup passes at two
granularities.

A `Segment` is one shape occurrence at its real (or, for the cluster-level
pass, cluster-canonical) position, tagged with a rotation estimate. Segments
are grouped by shape similarity (rotation/translation-normalized using each
segment's own `angle`); one member per group is elected representative and
its normalized `vectors` becomes a `UniqueSegment`. Every other member of a
passing group keeps only a `SegmentMeta` (offset + rotation back to its own
real position), so a single OCR result can be duplicated and repositioned
for every real occurrence without re-rendering or re-recognizing it. See
`docs/PIPELINE.md` for the worked dedup example.

This dataclass is reused for two lifecycles rather than duplicated:

- **Cluster-level (pre-Radon)**: one `Segment` per surviving classification
  cluster, `angle` a PCA principal-axis estimate over the cluster's own
  point cloud (`pipelines/_steps.py::_cluster_angle`) -- pure vector-geometry
  math, no rendering. `image` is `None` here. Feeds `group_similar_segments`
  + `detect_text_fast`, whose `UniqueSegment`/`SegmentMeta` output at this
  level represent one *whole representative cluster* and its real cluster
  occurrences, respectively.
- **Word-level (post-Radon, representatives only)**: one `Segment` per word,
  produced by `OCR/radon.py::segment_clusters` when called on just an
  elected representative's own (cluster-canonical-frame) vectors. `angle` is
  Radon's precise residual skew for that word; `image` is the word's own
  deskewed crop (captured directly during segmentation, so OCR never has to
  re-render).
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
    angle: float  # precise (Radon) or estimated (PCA) rotation, degrees --
    # NEVER rounded to a quarter turn anywhere in this pipeline.
    image: "np.ndarray | None" = None  # word-level (post-Radon) only: that
    # word's own deskewed crop, straight from segmentation -- None for the
    # cluster-level (pre-Radon) lifecycle, which is never rendered.


@dataclass
class UniqueSegment:
    """The one representative per similarity group that gets processed
    further. `vectors` are `transform_vector`'d to a canonical frame:
    translated so the segment's own bbox origin is (0, 0), rotated by
    `-angle` so it's upright -- no absolute position/rotation needs
    tracking past this point. At the cluster-level pass, this is a whole
    representative cluster's vectors (rendered + Radon-segmented next, not
    OCR'd directly); at the word-level pass it would be one word (not
    currently produced -- word-level Segments are OCR'd directly instead,
    see `OCR/Paddle_OCR/ocr_backend.py::recognize_segments`)."""

    vectors: "list[Vector]"


@dataclass
class SegmentMeta:
    """One entry per original segment instance in a passing similarity
    group (including the unique/representative segment's own instance --
    it is restored via this same math, from its own `offset`/`rotation`,
    not hardcoded to identity). `rotation` is that instance's own `angle`
    (a real cluster occurrence's PCA estimate, at the cluster-level pass);
    `offset` is that instance's own post-rotation bbox origin, itself
    rotated forward by `rotation` -- i.e. exactly the `(offset,
    rotation_deg)` pair `helpers.geometry.transform_vector`/`transform_bbox`
    expect to invert the canonical-frame normalization and place a
    `UniqueSegment`'s eventual OCR result(s) back onto this instance's real
    page position: `transform_bbox(canonical_bbox, offset=meta.offset,
    rotation_deg=meta.rotation)` (see `pipelines/sub_pipelines/ocr.py::
    restore_cluster_texts`, which replays every one of the representative's
    word-level `Text`s through this same transform for each `SegmentMeta`).
    `page_index`/`seqno` are carried here (rather than re-derived from
    discarded original vectors) since a non-representative instance's real
    Vectors are never kept past the FAST-detection step."""

    unique_index: int
    offset: tuple[float, float]
    rotation: float
    page_index: int
    seqno: int
