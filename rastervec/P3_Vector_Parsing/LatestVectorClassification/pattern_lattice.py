"""Pattern-lattice step: repeated motifs (stipple, hatch tiles, symbol
arrays) laid out on a regular translation lattice -> drawing.

1. **Similarity** (`group_by_similarity`): Vectors with the same item kinds
   in the same order and the same per-item lengths hashed into
   `PATTERN_SIM_LENGTH_TOL_PT` buckets (`similarity_key`) -- rotation-
   invariant: `"l"` its length, `"c"` its three control-polygon legs, `"re"`
   its sorted (w, h), `"qu"` its four edges in order. Two lengths either
   side of a bucket boundary are not similar (accepted).
2. **Ordered groups** (`ordered_groups`), only for similarity buckets with
   more than `PATTERN_MIN_BUCKET` members. Every position is a Vector's
   `anchor` (the first point of its first item). Seeds go in `seqno` order
   over the still-unassigned Vectors: v1 = seed -> nearest unassigned
   neighbour (farther than `PATTERN_MIN_STEP_PT`), v2 = seed -> nearest
   unassigned neighbour not within `ANGLE_TOL_DEG` of +-v1 (none -> a 1D
   lattice). The group is flood-filled from the seed over integer lattice
   *sites* (n1, n2), one unit step (+-v1, +-v2) at a time from sites that
   have a member: every unassigned Vector within `PATTERN_GRID_TOL_PT` of
   the **global** grid point `origin + n1 v1 + n2 v2` (and nearer that site
   than any other) is claimed at that site. A site with no such Vector is a dead end -- the flood
   never jumps a gap, so a missing n stops a 1D chain in that direction.
   One neighbour pair alone is too noisy to extrapolate far, so origin
   and basis are first least-squares prefitted to the seed's KNN neighbours
   that lie near integer sites of that pair's basis, then refitted to the
   members found so far whenever the site count doubles (sites found empty
   under the old basis may be retried after a refit).
3. A group is **drawing** only when it has more than `PATTERN_MAX_GROUP`
   members **and** every link (segment between the bbox centres of two
   lattice-adjacent members) crosses fewer than `PATTERN_LINK_FOREIGN_LIMIT`
   bboxes of foreign Vectors (the same bucket, not in the group) -- a
   repeated glyph with other text between its copies stays a text candidate.

Cost: O(N k) hashed signatures; flooding is O(m log m) per similarity
bucket (one static KD-tree for the site queries, the neighbour tree over
the unassigned set rebuilt whenever that set has halved and queried past a
seed's coincident duplicates, refits on a doubling schedule). The
in-between test runs only for groups past the size check, builds its bbox
arrays lazily once per call, queries short links against a KD-tree of small
foreign bboxes (large ones brute-forced in bounded chunks) and stops at the
first dirty link.
"""
from __future__ import annotations

import math
from collections import deque
from itertools import chain

import numpy as np
from scipy.spatial import cKDTree

from rastervec.commons.models import Vector
from rastervec.P3_Vector_Parsing.LatestVectorClassification.config import (
    ANGLE_TOL_DEG,
    PATTERN_GRID_TOL_PT,
    PATTERN_KNN,
    PATTERN_LINK_FOREIGN_LIMIT,
    PATTERN_MAX_GROUP,
    PATTERN_MIN_BUCKET,
    PATTERN_MIN_STEP_PT,
    PATTERN_SIM_LENGTH_TOL_PT,
)

_EPS = 1e-9
_LINK_CHUNK = 4096  # links per in-between-test chunk
_PAIR_CHUNK = 100_000  # link x large-bbox pairs per brute-force chunk
_FIRST_REFIT = 4  # site count at the first basis refit (then doubling)
_PREFIT_TOL = 0.25  # KNN neighbours within this of an integer site seed the prefit
_STEPS_1D = ((1, 0), (-1, 0))
_STEPS_2D = ((1, 0), (-1, 0), (0, 1), (0, -1))


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
    """Item kinds in order + each item's lengths hashed into
    `PATTERN_SIM_LENGTH_TOL_PT` buckets (see the module docstring)."""
    return tuple(
        (item[0], *(math.floor(length / PATTERN_SIM_LENGTH_TOL_PT) for length in _item_lengths(item)))
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


def _similarity_index_groups(vectors: list[Vector], anchors: list, min_size: int) -> list[np.ndarray]:
    """Similarity buckets of at least `min_size` members as index arrays,
    in first-seen order; Vectors without an anchor are left out."""
    buckets: dict[tuple, list[int]] = {}
    for i, v in enumerate(vectors):
        if anchors[i] is not None:
            buckets.setdefault(similarity_key(v), []).append(i)
    return [np.asarray(idx) for idx in buckets.values() if len(idx) >= min_size]


def group_by_similarity(vectors: list[Vector]) -> list[list[Vector]]:
    """Similarity buckets in first-seen order; Vectors without an `anchor`
    are left out."""
    anchors = [anchor(v) for v in vectors]
    return [[vectors[i] for i in g] for g in _similarity_index_groups(vectors, anchors, 1)]


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
        return len(self.assigned) - int(np.count_nonzero(self.assigned))

    def neighbours(self, i: int, *, extra: int = 0, fresh: bool = False) -> list[tuple[float, int]]:
        """`(distance, index)` of up to `PATTERN_KNN + extra` unassigned
        anchors (excluding `i`), nearest first. Rebuilds first when the
        unassigned set has halved -- or, with `fresh`, whenever anything was
        assigned since the last build."""
        live = self._live_count()
        if live * 2 <= len(self.idx) or (fresh and live < len(self.idx)):
            self._build()
        if self.tree is None:
            return []
        k = min(PATTERN_KNN + 1 + extra, len(self.idx))
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


def _lattice_basis(i: int, live: _LiveTree, xy: np.ndarray, n_dups: int = 0
                   ) -> tuple[np.ndarray | None, np.ndarray | None, list[int]]:
    """`(v1, v2, neighbours)` for seed `i` (see the module docstring);
    `v1` `None` when there is no usable neighbour, `v2` `None` for a 1D
    lattice. `neighbours` are the KNN indices considered (for the prefit).
    `n_dups` unassigned anchors coincide with the seed; the query reaches
    past them so a stack of duplicates can't hide every real neighbour."""
    candidates = [(d, j) for d, j in live.neighbours(i, extra=n_dups) if d > PATTERN_MIN_STEP_PT]
    if not candidates:  # the k nearest may all be assigned already
        candidates = [(d, j) for d, j in live.neighbours(i, extra=n_dups, fresh=True)
                      if d > PATTERN_MIN_STEP_PT]
    if not candidates:
        return None, None, []
    nbrs = [j for _d, j in candidates]
    v1 = xy[nbrs[0]] - xy[i]
    a1 = _angle(v1)
    for j in nbrs[1:]:
        v = xy[j] - xy[i]
        diff = abs(_angle(v) - a1) % 180.0
        if min(diff, 180.0 - diff) > ANGLE_TOL_DEG:
            return v1, v, nbrs
    return v1, None, nbrs


def _perp(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]], dtype=float)


class _Grid:
    """The global lattice `origin + basis @ (n1, n2)`; in 1D the second
    basis column is perp(v1) and n2 is always 0 (so the perpendicular offset
    is held to the same fraction of |v1|)."""

    def __init__(self, origin: np.ndarray, v1: np.ndarray, v2: np.ndarray | None):
        self.one_d = v2 is None
        self._set(origin, v1, _perp(v1) if v2 is None else v2)

    def _set(self, origin: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> bool:
        basis = np.column_stack([b1, b2])
        n1, n2 = float(np.hypot(*b1)), float(np.hypot(*b2))
        det = float(np.linalg.det(basis))
        if not np.isfinite(det) or abs(det) <= _EPS * max(n1 * n2, _EPS):
            return False
        self.origin = np.asarray(origin, dtype=float)
        self.basis = basis
        self.inv = np.linalg.inv(basis)
        return True

    def point(self, site: np.ndarray) -> np.ndarray:
        return self.origin + self.basis @ site

    def fits(self, pts: np.ndarray, site: np.ndarray) -> np.ndarray:
        """Within `PATTERN_GRID_TOL_PT` of the site's grid point and nearer
        that site than any other (so steps under 2x the tolerance can't
        merge neighbouring sites)."""
        near = np.hypot(*(pts - self.point(site)).T) <= PATTERN_GRID_TOL_PT + _EPS
        frac = (pts - self.origin) @ self.inv.T
        return near & np.all(np.rint(frac) == site, axis=1)

    def prefit(self, pts: np.ndarray) -> int:
        """Refit to the seed's neighbourhood: every point within
        `_PREFIT_TOL` of an integer site of the initial (one-pair) basis.
        Returns how many points the fit used (0 when it was skipped)."""
        frac = (pts - self.origin) @ self.inv.T
        sites = np.rint(frac)
        ok = np.all(np.abs(frac - sites) <= _PREFIT_TOL, axis=1)
        if self.one_d:
            ok &= sites[:, 1] == 0
        return int(ok.sum()) if self.refit(sites[ok], pts[ok]) else 0

    def refit(self, sites: np.ndarray, pts: np.ndarray) -> bool:
        cols = [np.ones(len(sites)), sites[:, 0]] + ([] if self.one_d else [sites[:, 1]])
        a = np.column_stack(cols)
        x, _res, rank, _sv = np.linalg.lstsq(a, pts, rcond=None)
        if rank < a.shape[1]:
            return False
        return self._set(x[0], x[1], _perp(x[1]) if self.one_d else x[2])


def _claim(grid: _Grid, site: tuple[int, int], tree: cKDTree, xy: np.ndarray,
           assigned: np.ndarray) -> list[int]:
    """Claim (mark `assigned`) every unassigned anchor fitting `site`."""
    target = np.asarray(site, dtype=float)
    cand = [j for j in tree.query_ball_point(grid.point(target), PATTERN_GRID_TOL_PT) if not assigned[j]]
    if not cand:
        return []
    cand = np.asarray(sorted(cand))
    hit = cand[grid.fits(xy[cand], target)]
    assigned[hit] = True
    return hit.tolist()


def _flood(i: int, v1: np.ndarray, v2: np.ndarray | None, nbrs: list[int], tree: cKDTree,
           xy: np.ndarray, assigned: np.ndarray) -> tuple[list[int], np.ndarray]:
    """Flood-fill seed `i`'s lattice group (marks members `assigned`).
    Returns the members in visiting order and the `(L, 2)` links between
    lattice-adjacent sites (one representative member per site)."""
    grid = _Grid(xy[i].copy(), v1, v2)
    prefit_n = grid.prefit(xy[[i, *nbrs]])
    steps = _STEPS_1D if grid.one_d else _STEPS_2D
    assigned[i] = True
    members = [i]
    members.extend(_claim(grid, (0, 0), tree, xy, assigned))  # coincident duplicates
    sites: dict[tuple[int, int], int] = {(0, 0): i}
    site_list: list[tuple[int, int]] = [(0, 0)]
    dead: set[tuple[int, int]] = set()
    next_refit = max(_FIRST_REFIT, 2 * prefit_n)  # never refit on fewer points than the prefit
    queue = deque([(0, 0)])
    while queue:
        n1, n2 = queue.popleft()
        for d1, d2 in steps:
            site = (n1 + d1, n2 + d2)
            if site in sites or site in dead:
                continue
            claimed = _claim(grid, site, tree, xy, assigned)
            if not claimed:  # a gap: never stepped over
                dead.add(site)
                continue
            members.extend(claimed)
            sites[site] = claimed[0]
            site_list.append(site)
            queue.append(site)
            if len(site_list) >= next_refit:
                reps = [sites[s] for s in site_list]
                grid.refit(np.asarray(site_list, dtype=float), xy[reps])
                dead.clear()  # a gap under the old basis may fit the new one
                next_refit *= 2
    forward = steps[::2]  # (1, 0) [, (0, 1)]
    links = [(rep, sites[(n1 + d1, n2 + d2)])
             for (n1, n2), rep in sites.items() for d1, d2 in forward
             if (n1 + d1, n2 + d2) in sites]
    return members, np.asarray(links, dtype=np.int64).reshape(-1, 2)


def _lattice_groups(xy: np.ndarray, seqnos: list[int]) -> list[tuple[list[int], np.ndarray]]:
    """`(members, links)` per lattice group (singletons included), seeds in
    `seqno` order; indices into `xy`."""
    assigned = np.zeros(len(xy), dtype=bool)
    tree = cKDTree(xy)
    live = _LiveTree(xy, assigned)
    no_links = np.zeros((0, 2), dtype=np.int64)
    groups: list[tuple[list[int], np.ndarray]] = []
    for i in sorted(range(len(xy)), key=lambda k: (seqnos[k], k)):
        if assigned[i]:
            continue
        dups = [j for j in tree.query_ball_point(xy[i], PATTERN_MIN_STEP_PT)
                if j != i and not assigned[j]]
        v1, v2, nbrs = _lattice_basis(i, live, xy, len(dups))
        if v1 is None:  # no lattice: the seed and its stacked duplicates stay singletons
            for j in [i, *sorted(dups)]:
                assigned[j] = True
                groups.append(([j], no_links))
            continue
        groups.append(_flood(i, v1, v2, nbrs, tree, xy, assigned))
    return groups


def ordered_groups(similar: list[Vector]) -> list[list[Vector]]:
    """Partition one similarity bucket into lattice groups (singletons
    included), seeds in `seqno` order."""
    xy = np.asarray([anchor(v) for v in similar], dtype=float)
    return [[similar[j] for j in members]
            for members, _links in _lattice_groups(xy, [v.seqno for v in similar])]


class _Foreign:
    """One bucket's bbox geometry for the in-between test, plus a reusable
    member mask (cleared after every group)."""

    def __init__(self, vectors: list[Vector]):
        raw = np.asarray([v.bbox for v in vectors], dtype=float).reshape(-1, 4)
        self.bboxes = np.column_stack([
            np.minimum(raw[:, 0], raw[:, 2]), np.minimum(raw[:, 1], raw[:, 3]),
            np.maximum(raw[:, 0], raw[:, 2]), np.maximum(raw[:, 1], raw[:, 3]),
        ])
        self.centres = (self.bboxes[:, :2] + self.bboxes[:, 2:]) * 0.5
        size = self.bboxes[:, 2:] - self.bboxes[:, :2]
        self.half_diag = 0.5 * np.hypot(size[:, 0], size[:, 1])
        self.member = np.zeros(len(vectors), dtype=bool)


def _segments_hit_boxes(p: np.ndarray, q: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Row-wise: does segment p->q touch the box (Liang-Barsky)?"""
    d = q - p
    t0 = np.zeros(len(p))
    t1 = np.ones(len(p))
    ok = np.ones(len(p), dtype=bool)
    for ax in (0, 1):
        lo, hi = boxes[:, ax], boxes[:, ax + 2]
        pa, da = p[:, ax], d[:, ax]
        flat = np.abs(da) < _EPS
        ok &= ~flat | ((pa >= lo) & (pa <= hi))
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = np.where(flat, -np.inf, (lo - pa) / da)
            tb = np.where(flat, np.inf, (hi - pa) / da)
        t0 = np.maximum(t0, np.minimum(ta, tb))
        t1 = np.minimum(t1, np.maximum(ta, tb))
    return ok & (t0 <= t1)


def _links_clean(members: np.ndarray, links: np.ndarray, fg: _Foreign) -> bool:
    """True when every link crosses fewer than `PATTERN_LINK_FOREIGN_LIMIT`
    foreign bboxes (`members`/`links` index the bucket's Vectors)."""
    if len(links) == 0:
        return True
    gb = fg.bboxes[members]
    gx0, gy0 = gb[:, 0].min(), gb[:, 1].min()
    gx1, gy1 = gb[:, 2].max(), gb[:, 3].max()
    fg.member[members] = True
    b = fg.bboxes
    near = ~fg.member & (b[:, 0] <= gx1) & (b[:, 2] >= gx0) & (b[:, 1] <= gy1) & (b[:, 3] >= gy0)
    fg.member[members] = False
    foreign = np.flatnonzero(near)
    if len(foreign) < PATTERN_LINK_FOREIGN_LIMIT:
        return True
    a, c = fg.centres[links[:, 0]], fg.centres[links[:, 1]]
    seg_len = np.hypot(*(c - a).T)
    reach = 2.0 * float(seg_len.max())
    small_mask = fg.half_diag[foreign] <= reach
    small, large = foreign[small_mask], foreign[~small_mask]
    tree = cKDTree(fg.centres[small]) if len(small) else None
    large_chunk = max(1, _PAIR_CHUNK // max(len(large), 1))
    for s in range(0, len(links), _LINK_CHUNK):
        ca, cc, cl = a[s:s + _LINK_CHUNK], c[s:s + _LINK_CHUNK], seg_len[s:s + _LINK_CHUNK]
        counts = np.zeros(len(ca), dtype=np.int64)
        if tree is not None:
            # a small box touching the segment has its centre within
            # half_diag <= reach of it, so within len/2 + reach of its midpoint
            hits = tree.query_ball_point((ca + cc) * 0.5, cl * 0.5 + reach)
            lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
            total = int(lens.sum())
            if total:
                li = np.repeat(np.arange(len(ca)), lens)
                fi = small[np.fromiter(chain.from_iterable(hits), dtype=np.int64, count=total)]
                hit = _segments_hit_boxes(ca[li], cc[li], fg.bboxes[fi])
                counts += np.bincount(li[hit], minlength=len(ca))
        for t in range(0, len(ca) if len(large) else 0, large_chunk):
            k = min(large_chunk, len(ca) - t)
            li = np.repeat(np.arange(t, t + k), len(large))
            fi = np.tile(large, k)
            hit = _segments_hit_boxes(ca[li], cc[li], fg.bboxes[fi])
            counts += np.bincount(li[hit], minlength=len(ca))
        if (counts >= PATTERN_LINK_FOREIGN_LIMIT).any():
            return False
    return True


def pattern_drawing(vectors: list[Vector]) -> tuple[list[Vector], list[list[Vector]]]:
    """`(kept, drawing_groups)` -- `kept` in input order, `drawing_groups`
    every lattice group with more than `PATTERN_MAX_GROUP` members whose
    links are all clean of foreign content (see the module docstring)."""
    anchors = [anchor(v) for v in vectors]
    fg: _Foreign | None = None
    dropped = np.zeros(len(vectors), dtype=bool)
    drawing_groups: list[list[Vector]] = []
    for sim in _similarity_index_groups(vectors, anchors, PATTERN_MIN_BUCKET + 1):
        xy = np.asarray([anchors[i] for i in sim], dtype=float)
        for members, links in _lattice_groups(xy, [vectors[i].seqno for i in sim]):
            if len(members) <= PATTERN_MAX_GROUP:
                continue
            if fg is None:
                fg = _Foreign(vectors)
            group = sim[members]
            if _links_clean(group, sim[links], fg):
                dropped[group] = True
                drawing_groups.append([vectors[i] for i in group])
    if not drawing_groups:
        return list(vectors), drawing_groups
    return [v for v, d in zip(vectors, dropped) if not d], drawing_groups
