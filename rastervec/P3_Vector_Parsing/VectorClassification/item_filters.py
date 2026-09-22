"""Item-level helper of the Vector Classification pipeline (see
`classify_vectors.py` for the fixed step order).

`bbox_of` -- the "union bbox of a group's member vectors" helper -- is
reused by `cluster_filters.py`'s spatial-clustering step. The pure bbox
math (`max_dimension`, `dims`) lives in `helpers/geometry.py`.
"""
from __future__ import annotations

from rastervec.commons.helpers.geometry import BBox, union_bbox
from rastervec.commons.models import Vector


def bbox_of(group: list[Vector]) -> BBox:
    """Union bbox of every member Vector's own bbox."""
    return union_bbox([v.bbox for v in group])
