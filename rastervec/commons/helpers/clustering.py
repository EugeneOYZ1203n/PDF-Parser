"""Spatial connected-components clustering.

Two functions here, both plain spatial-hash-grid + union-find (no
scikit-learn/scipy dependency), which stay fast even on pages with tens of
thousands of path items (`cluster_spatial` is near-linear unless many items
exceed its per-item cell cap, see there): `cluster_spatial` (bbox-*gap*-based, used by the
Vector Classification pipeline's `Vector_Classification/cluster_filters.py::
cluster_spatial_groups` and `pipeline.py::_run_spatial_regroup`) and
`group_by_overlap` (pure bbox *overlap/touch*, no distance threshold at
all).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np

from rastervec.commons.helpers.geometry import bboxes_intersect, rect_gap
from rastervec.commons.logging_setup import get_logger

_LOG = get_logger("clustering")

# A single bbox spanning more than this many grid cells is kept out of the
# grid (so one huge item -- e.g. an unfiltered background rect -- can't blow
# up the grid's memory/time): `cluster_spatial` checks such items against
# every item directly instead; `group_by_overlap` (whose cells are sized
# from the largest item, so this never triggers there in practice) falls
# back to a centre-cell bucket.
_MAX_CELLS_PER_ITEM = 2000


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _group_by_root(items: list, uf: _UnionFind) -> list[list]:
    buckets: dict[int, list] = defaultdict(list)
    for index, item in enumerate(items):
        buckets[uf.find(index)].append(item)
    return list(buckets.values())


def cluster_spatial(
    items: list,
    get_bbox: Callable[[Any], tuple],
    *,
    threshold: float,
    extra_close: Callable[[Any, Any], bool] | None = None,
    max_union_area: float | None = None,
) -> list[list]:
    """Connected-components clustering by bbox gap: items whose boxes are
    within `threshold` of some other item in the same cluster end up
    together (single-linkage). Uses a spatial hash grid so this stays close
    to linear time on large item counts. `extra_close`, if given, is an
    additional required condition (e.g. "similar max side length") checked
    on top of the bbox-gap rule -- two items only union when both the gap
    and `extra_close` pass. `max_union_area`, if given, caps cluster size:
    two clusters never merge when the bbox of their union would have an
    area of at least `max_union_area` (each cluster's running bbox is
    tracked at its union-find root), so the result depends on pair order
    only where that cap bites.

    An item spanning more than `_MAX_CELLS_PER_ITEM` grid cells (~447 x 447
    pt at a 10 pt threshold) is not put in the grid; after the grid pass,
    each such item is compared against every item whose bbox meets its own
    expanded by `threshold` (one numpy mask -- a necessary condition for a
    gap within `threshold`), with the same exact gap/`extra_close`/cap test.
    That costs O(big items x N), a handful of masks per page; a degenerate
    `threshold` near 0 makes every item "big" and the call O(N^2) -- correct,
    but `group_by_overlap` is the fast tool for pure overlap."""
    if not items:
        return []

    cell = max(threshold, 1e-6)
    bboxes = [tuple(get_bbox(item)) for item in items]
    uf = _UnionFind(len(items))
    root_bbox = list(bboxes) if max_union_area is not None else None
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    cell_spans: list[tuple[int, int, int, int] | None] = []
    big: list[int] = []

    def _try_union(i: int, j: int) -> None:
        if rect_gap(bboxes[i], bboxes[j]) > threshold:
            return
        if extra_close is not None and not extra_close(items[i], items[j]):
            return
        if root_bbox is None:
            uf.union(i, j)
            return
        ra, rb = uf.find(i), uf.find(j)
        if ra == rb:
            return
        a, b = root_bbox[ra], root_bbox[rb]
        merged = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
        if (merged[2] - merged[0]) * (merged[3] - merged[1]) >= max_union_area:
            return
        uf.union(ra, rb)
        root_bbox[uf.find(rb)] = merged

    for index, (x0, y0, x1, y1) in enumerate(bboxes):
        cx0, cy0 = int(x0 // cell), int(y0 // cell)
        cx1, cy1 = int(x1 // cell), int(y1 // cell)
        if (cx1 - cx0 + 1) * (cy1 - cy0 + 1) > _MAX_CELLS_PER_ITEM:
            # Too big for the grid: checked directly below.
            big.append(index)
            cell_spans.append(None)
            continue
        cell_spans.append((cx0, cy0, cx1, cy1))
        for gx in range(cx0, cx1 + 1):
            for gy in range(cy0, cy1 + 1):
                grid[(gx, gy)].append(index)

    # Grid pass: every pair of grid items in neighbouring cells.
    for index, span in enumerate(cell_spans):
        if span is None:
            continue
        cx0, cy0, cx1, cy1 = span
        neighbor_indices: set[int] = set()
        for gx in range(cx0 - 1, cx1 + 2):
            for gy in range(cy0 - 1, cy1 + 2):
                neighbor_indices.update(grid.get((gx, gy), ()))
        for other in sorted(neighbor_indices):
            if other > index:
                _try_union(index, other)

    # Big-item pass: each big item against every item near it (grid items
    # and later big items; earlier big items already checked this pair).
    if big:
        boxes = np.asarray(bboxes, dtype=float)
        is_big = np.zeros(len(items), dtype=bool)
        is_big[big] = True
        for b in big:
            x0, y0, x1, y1 = bboxes[b]
            near = (
                (boxes[:, 0] <= x1 + threshold) & (boxes[:, 2] >= x0 - threshold)
                & (boxes[:, 1] <= y1 + threshold) & (boxes[:, 3] >= y0 - threshold)
            )
            near[b] = False
            near[: b + 1] &= ~is_big[: b + 1]
            for other in np.flatnonzero(near):
                i, j = (b, int(other)) if b < other else (int(other), b)
                _try_union(i, j)

    clusters = _group_by_root(items, uf)
    _LOG.debug(
        "cluster_spatial: %d item(s) -> %d cluster(s) (threshold=%s)",
        len(items),
        len(clusters),
        threshold,
    )
    return clusters


def group_by_overlap(items: list, get_bbox: Callable[[Any], tuple]) -> list[list]:
    """Connected-components clustering by pure bbox overlap/touch, no
    distance threshold at all -- items whose boxes intersect or share an
    edge (`bboxes_intersect`) end up in the same group (single-linkage).

    Prefer this over `cluster_spatial(items, get_bbox, threshold=0.0)`,
    which gives the same groups but slowly: `cluster_spatial`'s grid cell
    size is `max(threshold, 1e-6)`, so a `threshold=0.0` collapses to a
    1e-6-sized cell -- every realistically-sized bbox then spans millions
    of cells, exceeds `_MAX_CELLS_PER_ITEM` and goes through its O(N)
    per-item direct check, O(N^2) overall. This function instead derives
    its own grid cell size from the items' own bbox extents (the largest
    width/height across every item, floored at 1.0) so the grid is always
    sized appropriately for whatever's actually being clustered, then uses
    the exact `bboxes_intersect` test (not a gap-vs-threshold comparison)
    for the real match criterion -- grid granularity only ever affects
    which candidate pairs get checked, never correctness."""
    if not items:
        return []

    bboxes = [tuple(get_bbox(item)) for item in items]
    max_dim = max((max(x1 - x0, y1 - y0) for x0, y0, x1, y1 in bboxes), default=0.0)
    cell = max(max_dim, 1.0)
    uf = _UnionFind(len(items))
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    cell_spans: list[tuple[int, int, int, int]] = []

    for index, (x0, y0, x1, y1) in enumerate(bboxes):
        cx0, cy0 = int(x0 // cell), int(y0 // cell)
        cx1, cy1 = int(x1 // cell), int(y1 // cell)
        span = (cx1 - cx0 + 1) * (cy1 - cy0 + 1)
        if span > _MAX_CELLS_PER_ITEM:
            cx = int(((x0 + x1) / 2.0) // cell)
            cy = int(((y0 + y1) / 2.0) // cell)
            cx0 = cx1 = cx
            cy0 = cy1 = cy
        cell_spans.append((cx0, cy0, cx1, cy1))
        for gx in range(cx0, cx1 + 1):
            for gy in range(cy0, cy1 + 1):
                grid[(gx, gy)].append(index)

    for index, (cx0, cy0, cx1, cy1) in enumerate(cell_spans):
        neighbor_indices: set[int] = set()
        for gx in range(cx0 - 1, cx1 + 2):
            for gy in range(cy0 - 1, cy1 + 2):
                neighbor_indices.update(grid.get((gx, gy), ()))
        for other in neighbor_indices:
            if other <= index:
                continue
            if bboxes_intersect(bboxes[index], bboxes[other]):
                uf.union(index, other)

    clusters = _group_by_root(items, uf)
    _LOG.debug("group_by_overlap: %d item(s) -> %d cluster(s)", len(items), len(clusters))
    return clusters
