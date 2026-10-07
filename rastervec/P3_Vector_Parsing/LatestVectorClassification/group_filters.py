"""Group-level step of the LatestVectorClassification classification chain (see
`classify_vectors.py` for the fixed step order). A "group" is the
post-seqno-overlap-merge, pre-spatial-clustering unit -- see
docs/Glossary.md for the group/cluster distinction.

`combine_overlapping_seq` -- sorts by `seqno`, then chain-merges
consecutive vectors into groups by bbox-gap tolerance. Never drops
anything -- pure regrouping.
"""
from __future__ import annotations

from rastervec.commons.helpers.geometry import rect_gap
from rastervec.commons.models import Vector


def combine_overlapping_seq(
    groups: list[list[Vector]], tolerance: float, max_area: float | None = None,
) -> tuple[list[list[Vector]], list[list[Vector]]]:
    """Sorts every incoming Vector by `seqno`, then sweeps in that order:
    a Vector joins the current group if its bbox is within `tolerance` of
    the group's aggregate bbox so far -- and, with `max_area`, the grown
    bbox's area would stay below `max_area` -- otherwise it starts a new
    group. Never drops anything -- pure regrouping."""
    flat = sorted((v for g in groups for v in g), key=lambda v: v.seqno)
    if not flat:
        return [], []

    result: list[list[Vector]] = []
    current = [flat[0]]
    current_bbox = tuple(flat[0].bbox)
    for v in flat[1:]:
        b = v.bbox
        grown = None
        if rect_gap(current_bbox, b) <= tolerance:
            grown = (
                min(current_bbox[0], b[0]), min(current_bbox[1], b[1]),
                max(current_bbox[2], b[2]), max(current_bbox[3], b[3]),
            )
            if max_area is not None and (grown[2] - grown[0]) * (grown[3] - grown[1]) >= max_area:
                grown = None
        if grown is not None:
            current.append(v)
            current_bbox = grown
        else:
            result.append(current)
            current = [v]
            current_bbox = tuple(v.bbox)
    result.append(current)
    return result, []
