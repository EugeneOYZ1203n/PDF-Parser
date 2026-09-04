"""Spatial connected-components clustering.

`cluster_spatial` is the one function here -- used by the Vector
Classification pipeline (`Vector_Classification/clusters/cluster_filters.py`'s
`cluster_spatial_groups` and `pipeline.py`'s `_run_spatial_regroup`). No
scikit-learn/scipy dependency: it uses a plain spatial hash grid +
union-find, which stays fast even on pages with tens of thousands of path
items.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from rastervec.helpers.geometry import rect_gap
from rastervec.logging_setup import get_logger

_LOG = get_logger("clustering")

# A single bbox spanning more than this many grid cells falls back to a
# single center-cell bucket, so one huge item (e.g. an unfiltered
# background rect) can't blow up the grid's memory/time.
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
) -> list[list]:
    """Connected-components clustering by bbox gap: items whose boxes are
    within `threshold` of some other item in the same cluster end up
    together (single-linkage). Uses a spatial hash grid so this stays close
    to linear time on large item counts. `extra_close`, if given, is an
    additional required condition (e.g. "similar max side length") checked
    on top of the bbox-gap rule -- two items only union when both the gap
    and `extra_close` pass."""
    if not items:
        return []

    cell = max(threshold, 1e-6)
    bboxes = [tuple(get_bbox(item)) for item in items]
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
            if rect_gap(bboxes[index], bboxes[other]) <= threshold:
                if extra_close is None or extra_close(items[index], items[other]):
                    uf.union(index, other)

    clusters = _group_by_root(items, uf)
    _LOG.debug(
        "cluster_spatial: %d item(s) -> %d cluster(s) (threshold=%s)",
        len(items),
        len(clusters),
        threshold,
    )
    return clusters
