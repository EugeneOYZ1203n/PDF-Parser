"""Spatial connected-components clustering.

`cluster_spatial`: single-linkage clustering by bbox *gap*, on a spatial hash grid +
union-find (no scikit-learn / scipy), near-linear even on pages with tens of thousands of
items. Used by `phase_vector_classify/v2_vector_class/cluster_filters.py`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable

import numpy as np

from raster_parser.commons.helpers.geometry import rect_gap
from raster_parser.commons.logging_setup import get_logger

# per-call bucket counts: DEBUG is opt-in (see `get_logger`)
_LOG = get_logger("clustering", default_level=logging.INFO)

# A single bbox spanning more than this many grid cells is kept out of the
# grid (so one huge item -- e.g. an unfiltered background rect -- can't blow
# up the grid's memory/time): `cluster_spatial` checks such items against
# every item directly instead.
_MAX_CELLS_PER_ITEM = 2000


class UnionFind:
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


def _group_by_root(items: list, uf: UnionFind) -> list[list]:
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
    `threshold` near 0 makes every item "big" and the call O(N^2)."""
    if not items:
        return []

    cell = max(threshold, 1e-6)
    bboxes = [tuple(get_bbox(item)) for item in items]
    uf = UnionFind(len(items))
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
                (boxes[:, 0] <= x1 + threshold)
                & (boxes[:, 2] >= x0 - threshold)
                & (boxes[:, 1] <= y1 + threshold)
                & (boxes[:, 3] >= y0 - threshold)
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
