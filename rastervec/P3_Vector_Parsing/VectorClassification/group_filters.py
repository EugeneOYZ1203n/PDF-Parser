"""Group-level step of the Vector Classification pipeline (see
`classify_vectors.py` for the fixed step order). A "group" is the
post-seqno-overlap-merge, pre-spatial-clustering unit -- see
docs/Glossary.md for the group/cluster distinction.

`combine_overlapping_seq` -- sorts by `seqno`, then chain-merges
consecutive vectors into groups by bbox-gap tolerance. Never drops
anything -- pure regrouping.
"""
from __future__ import annotations

from rastervec.commons.helpers.geometry import rect_gap, union_bbox
from rastervec.commons.models import Vector


def combine_overlapping_seq(
    groups: list[list[Vector]], tolerance: float,
) -> tuple[list[list[Vector]], list[list[Vector]]]:
    """Sorts every incoming Vector by `seqno`, then sweeps in that order:
    a Vector joins the current group if its bbox is within `tolerance` of
    the group's aggregate bbox so far, otherwise it starts a new group.
    Never drops anything -- pure regrouping."""
    flat = sorted((v for g in groups for v in g), key=lambda v: v.seqno)
    if not flat:
        return [], []

    result: list[list[Vector]] = []
    current = [flat[0]]
    current_bbox = flat[0].bbox
    for v in flat[1:]:
        if rect_gap(current_bbox, v.bbox) <= tolerance:
            current.append(v)
            current_bbox = union_bbox([current_bbox, v.bbox])
        else:
            result.append(current)
            current = [v]
            current_bbox = v.bbox
    result.append(current)
    return result, []
