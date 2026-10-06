"""Pattern-lattice step: repeated motifs (stipple, hatch tiles, symbol
arrays) laid out on a regular translation lattice -> drawing.

1. **Similarity** (`group_by_similarity`): Vectors with the same item kinds
   in the same order and the same per-item lengths (`similarity_key`,
   rounded to `PATTERN_SIM_LENGTH_TOL_PT`) -- rotation-invariant: `"l"` its
   length, `"c"` its three control-polygon legs, `"re"` its sorted
   (w, h), `"qu"` its four edges in order.
2. **Ordered groups** (`ordered_groups`), only for similarity buckets with
   more than `PATTERN_MIN_BUCKET` members. Every position is a Vector's
   `anchor` (the first point of its first item). Seeds go in `seqno` order
   over the still-unassigned Vectors: v1 = seed -> nearest unassigned
   neighbour (farther than `PATTERN_MIN_STEP_PT`), v2 = seed -> nearest
   unassigned neighbour not within `ANGLE_TOL_DEG` of +-v1 (none -> a 1D
   lattice). The group is flood-filled from the seed: from every member,
   each step +-v1 (+-v2) claims every unassigned Vector within
   `PATTERN_LATTICE_TOL_FRAC * |step|` of the stepped position -- a
   connected lattice patch, never a global `p + n1 v1 + n2 v2` fit (which
   merges distant patches and, for small steps, nearly everything).
3. Groups with more than `PATTERN_MAX_GROUP` members are drawing.

Cost: O(N k) signatures + O(m log m) per similarity bucket of size m (one
static KD-tree for the flood's ball queries, each member popped once; the
neighbour tree over the unassigned set is rebuilt whenever that set has
halved, a geometric sum).
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
from scipy.spatial import cKDTree

from rastervec.commons.models import Vector
from rastervec.P3_Vector_Parsing.LatestVectorClassification.config import (
    ANGLE_TOL_DEG,
    PATTERN_KNN,
    PATTERN_LATTICE_TOL_FRAC,
    PATTERN_MAX_GROUP,
    PATTERN_MIN_BUCKET,
    PATTERN_MIN_STEP_PT,
    PATTERN_SIM_LENGTH_TOL_PT,
)


def _dist(a, b) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _item_lengths(item: tuple) -> tuple[float, ...]:
    kind = item[0]
    if kind == "l":
        return (_dist(item[1], item[2]),)
    if kind == "c":
        p = item[1:5]
        return (_dist(p[0], p[1]), _dist(p[1], p[2]), _dist(p[2], p[3]))
    if kind == "re":
        x0, y0, x1, y1 = item[1]
        w, h = abs(x1 - x0), abs(y1 - y0)
        return (min(w, h), max(w, h))
    if kind == "qu":
        q = item[1]
        return tuple(_dist(q[i], q[(i + 1) % 4]) for i in range(4))
    return ()


def similarity_key(v: Vector) -> tuple:
    """Item kinds in order + each item's lengths rounded to
    `PATTERN_SIM_LENGTH_TOL_PT` (see the module docstring)."""
    return tuple(
        (item[0], *(round(length / PATTERN_SIM_LENGTH_TOL_PT) for length in _item_lengths(item)))
        for item in v.items
    )


def anchor(v: Vector) -> tuple[float, float] | None:
    """The first point of the first item (`"re"` -> its (x0, y0), `"qu"` ->
    its ul); `None` for a Vector with no usable item."""
    if not v.items:
        return None
    item = v.items[0]
    kind = item[0]
    if kind in ("l", "c"):
        return tuple(item[1])
    if kind == "re":
        return (item[1][0], item[1][1])
    if kind == "qu":
        return tuple(item[1][0])
    return None


def group_by_similarity(vectors: list[Vector]) -> list[list[Vector]]:
    """Similarity buckets in first-seen order; Vectors without an `anchor`
    are left out."""
    buckets: dict[tuple, list[Vector]] = {}
    for v in vectors:
        if anchor(v) is None:
            continue
        buckets.setdefault(similarity_key(v), []).append(v)
    return list(buckets.values())


class _LiveTree:
    """KD-tree over the unassigned anchors, rebuilt once the unassigned set
    has halved since the last build."""

    def __init__(self, xy: np.ndarray, assigned: np.ndarray):
        self.xy = xy
        self.assigned = assigned
        self._build()

    def _build(self) -> None:
        self.idx = np.flatnonzero(~self.assigned)
        self.tree = cKDTree(self.xy[self.idx]) if len(self.idx) else None

    def _live_count(self) -> int:
        return int((~self.assigned).sum())

    def neighbours(self, i: int, *, fresh: bool = False) -> list[tuple[float, int]]:
        """`(distance, index)` of up to `PATTERN_KNN` unassigned anchors
        (excluding `i`), nearest first. Rebuilds first when the unassigned
        set has halved -- or, with `fresh`, whenever anything was assigned
        since the last build."""
        live = self._live_count()
        if live * 2 <= len(self.idx) or (fresh and live < len(self.idx)):
            self._build()
        if self.tree is None:
            return []
        k = min(PATTERN_KNN + 1, len(self.idx))
        dists, pos = self.tree.query(self.xy[i], k=k)
        dists, pos = np.atleast_1d(dists), np.atleast_1d(pos)
        out = []
        for d, p in zip(dists, pos):
            j = int(self.idx[p])
            if j != i and not self.assigned[j]:
                out.append((float(d), j))
        return out


def _angle(v: np.ndarray) -> float:
    return math.degrees(math.atan2(v[1], v[0])) % 180.0


def _lattice_basis(i: int, live: _LiveTree, xy: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """`(v1, v2)` for seed `i` (see the module docstring); `(None, None)`
    when there is no usable neighbour, `v2` `None` for a 1D lattice."""
    candidates = [(d, j) for d, j in live.neighbours(i) if d > PATTERN_MIN_STEP_PT]
    if not candidates:  # the k nearest may all be assigned already
        candidates = [(d, j) for d, j in live.neighbours(i, fresh=True) if d > PATTERN_MIN_STEP_PT]
    if not candidates:
        return None, None
    v1 = xy[candidates[0][1]] - xy[i]
    a1 = _angle(v1)
    for _d, j in candidates[1:]:
        v = xy[j] - xy[i]
        diff = abs(_angle(v) - a1) % 180.0
        if min(diff, 180.0 - diff) > ANGLE_TOL_DEG:
            return v1, v
    return v1, None


def _flood(i: int, v1: np.ndarray, v2: np.ndarray | None, tree: cKDTree,
           xy: np.ndarray, assigned: np.ndarray) -> list[int]:
    """Flood-fill seed `i`'s lattice group (marks members `assigned`), in
    visiting order."""
    steps = [v1, -v1] + ([v2, -v2] if v2 is not None else [])
    radii = [PATTERN_LATTICE_TOL_FRAC * float(np.hypot(*d)) for d in steps]
    assigned[i] = True
    members = [i]
    queue = deque([i])
    while queue:
        q = queue.popleft()
        for d, r in zip(steps, radii):
            for j in sorted(tree.query_ball_point(xy[q] + d, r)):
                if not assigned[j]:
                    assigned[j] = True
                    members.append(j)
                    queue.append(j)
    return members


def ordered_groups(similar: list[Vector]) -> list[list[Vector]]:
    """Partition one similarity bucket into lattice groups (singletons
    included), seeds in `seqno` order."""
    xy = np.asarray([anchor(v) for v in similar], dtype=float)
    assigned = np.zeros(len(similar), dtype=bool)
    tree = cKDTree(xy)
    live = _LiveTree(xy, assigned)
    groups: list[list[Vector]] = []
    for i in sorted(range(len(similar)), key=lambda k: (similar[k].seqno, k)):
        if assigned[i]:
            continue
        v1, v2 = _lattice_basis(i, live, xy)
        if v1 is None:
            assigned[i] = True
            groups.append([similar[i]])
            continue
        groups.append([similar[j] for j in _flood(i, v1, v2, tree, xy, assigned)])
    return groups


def pattern_drawing(vectors: list[Vector]) -> tuple[list[Vector], list[list[Vector]]]:
    """`(kept, drawing_groups)` -- `kept` in input order, `drawing_groups`
    every lattice group with more than `PATTERN_MAX_GROUP` members."""
    drawing_groups: list[list[Vector]] = []
    for similar in group_by_similarity(vectors):
        if len(similar) <= PATTERN_MIN_BUCKET:
            continue
        drawing_groups.extend(g for g in ordered_groups(similar) if len(g) > PATTERN_MAX_GROUP)
    dropped = {id(v) for g in drawing_groups for v in g}
    return [v for v in vectors if id(v) not in dropped], drawing_groups
