"""Rotation decisions for CollinearVectorClass -- pure functions over
vector-derived parallel groups, so every rule is unit-testable without
PaddleOCR.

All angles are page space (y down), axial, folded to [0, 180) -- a line
direction, with up/down the same. `parse.py` turns a chosen angle into an
image rotation and leaves the remaining 0/180 (and 90) ambiguity to
the low-confidence retry sweep (+90/180/270).

- `pre_detect_direction` -- the rotation applied to a whole cluster render
  before PaddleOCR detect: one parallel group -> its angle; none -> 0;
  several -> the one with the longest total length.
- `final_direction` -- the per-detected-quad correction before recognition,
  from the parallel groups among the vectors under that quad:
  * one connected component: same rules as `pre_detect_direction`;
  * more than one: the Hough line angle (mod 180) snapped to the closest
    global potential angle, whatever the parallel groups are; when Hough
    finds no line, the parallel-group rules above.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rastervec.commons.models import Vector
from rastervec.P3_Vector_Parsing.CollinearVectorClass.config import (
    ANGLE_TOL_DEG,
    MIN_PARALLEL_GROUP_SIZE,
    STRAIGHT_TOL_PT,
)
from rastervec.P3_Vector_Parsing.CollinearVectorClass.line_geometry import (
    axial_distance,
    group_angle,
    group_parallel,
    group_total_length,
    split_straight,
)


@dataclass(frozen=True)
class ParallelGroup:
    """One parallel group's summary: mean axial `angle`, summed member
    `total_length` (pt) and member `count`."""

    angle: float
    total_length: float
    count: int


@dataclass(frozen=True)
class RotationDecision:
    """The chosen axial page-space `angle`, the `rule` that chose it (a
    short label, rendered as its own debug layer), and the page-space Hough
    angle when Hough ran (`None` otherwise)."""

    angle: float
    rule: str
    hough_angle: "float | None" = None


def parallel_groups(vectors: list[Vector]) -> list[ParallelGroup]:
    """Parallel groups among `vectors`' straight members -- angle groups
    with at least `MIN_PARALLEL_GROUP_SIZE` members (a lone stroke is not a
    parallel group)."""
    straight, _ = split_straight(vectors, STRAIGHT_TOL_PT)
    return [
        ParallelGroup(group_angle(g), group_total_length(g), len(g))
        for g in group_parallel(straight, ANGLE_TOL_DEG)
        if len(g) >= MIN_PARALLEL_GROUP_SIZE
    ]


def snap_axial(angle: float, candidates: list[float]) -> "float | None":
    """The candidate closest to `angle` on the axial (period-180) circle,
    or `None` when there are no candidates."""
    if not candidates:
        return None
    return min(candidates, key=lambda c: axial_distance(angle, c))


def _by_groups(groups: list[ParallelGroup], prefix: str) -> RotationDecision:
    if not groups:
        return RotationDecision(0.0, f"{prefix} no parallel group")
    if len(groups) == 1:
        return RotationDecision(groups[0].angle, f"{prefix} 1 parallel group")
    best = max(groups, key=lambda g: g.total_length)
    return RotationDecision(best.angle, f"{prefix} longest parallel group")


def pre_detect_direction(groups: list[ParallelGroup]) -> RotationDecision:
    """Direction to rotate a whole cluster render by before detect."""
    return _by_groups(groups, "cluster")


def final_direction(
    n_components: int,
    groups: list[ParallelGroup],
    hough: "Callable[[], float | None]",
    global_angles: list[float],
) -> RotationDecision:
    """Per-quad direction before recognition (see module docstring).
    `hough` is called lazily -- only for more than one component -- and
    returns the quad's page-space Hough angle or `None`. With no global
    angles at all, the raw Hough angle is used."""
    if n_components <= 1:
        return _by_groups(groups, "1cc")
    h = hough()
    if h is None:
        return _by_groups(groups, "multi-cc no hough,")
    snapped = snap_axial(h, global_angles)
    if snapped is None:
        return RotationDecision(h % 180.0, "multi-cc hough, no global angles", h)
    return RotationDecision(snapped, "multi-cc hough snapped to global angle", h)
