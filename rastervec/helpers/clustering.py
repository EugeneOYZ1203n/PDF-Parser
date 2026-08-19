"""Clustering helper.

cluster_spatial/cluster_by_dimension/cluster_by_seq/group_by_overlap are
used by Vector's classification pipeline (spatial closeness -> seq-number
proximity -> overlap grouping -> dimension similarity, applied twice: once
per-path early on via cluster_by_dimension's generic T support -- no
longer used that way by Vector directly, kept for reuse -- and once
per-group late in the pipeline). cluster_hsv (Raster.separate_by_color's
HSV pixel clustering) is not implemented yet -- that's a Raster-phase
concern, deferred along with numpy/image-library imports until Raster is
actually built.

No scikit-learn/scipy dependency: cluster_spatial uses a plain spatial
hash grid + union-find, which stays fast even on pages with tens of
thousands of path items without adding a new third-party dependency.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from rastervec.geometry import rect_gap
from rastervec.logging_setup import get_logger

_LOG = get_logger("clustering")

# A single bbox spanning more than this many grid cells falls back to a
# single center-cell bucket, so one huge item (e.g. an unfiltered
# background rect) can't blow up the grid's memory/time.
_MAX_CELLS_PER_ITEM = 2000

# Above this many items, cluster_by_dimension/cluster_by_seq's O(k^2)
# within-group pass is skipped for that group (kept as one cluster) to
# bound worst-case runtime on pathological pages.
_MAX_GROUP_SIZE_FOR_PAIRWISE = 4000


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


def _bbox_fully_contains(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    """True if a and b's bboxes overlap and one fully contains (or equals)
    the other. No intersection returns False -- see
    Clustering.group_by_overlap, which never merges a contained pair
    regardless of gap tolerance (a fully-enclosed item is a distinct thing,
    not part of the same shape)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter_area = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(ax1 - ax0, 0.0) * max(ay1 - ay0, 0.0)
    area_b = max(bx1 - bx0, 0.0) * max(by1 - by0, 0.0)
    if area_a <= 0.0 or area_b <= 0.0:
        return False
    return inter_area >= area_a - 1e-9 or inter_area >= area_b - 1e-9


def _bboxes_close_or_overlapping(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    tolerance: float,
) -> bool:
    """True if a and b's bboxes overlap/touch/are within `tolerance` of each
    other, UNLESS one fully contains (or equals) the other -- a
    fully-enclosed item never merges, regardless of tolerance. rect_gap
    already returns 0.0 for overlapping/touching boxes, so this one check
    covers both "actually touching" and "merely nearby"."""
    if _bbox_fully_contains(a, b):
        return False
    return rect_gap(a, b) <= tolerance


class Clustering:
    """Generic clustering utilities reused by both the Vector and Raster
    stages."""

    def cluster_spatial(
        self,
        items: list,
        get_bbox: Callable[[Any], tuple],
        *,
        threshold: float = 3.0,
    ) -> list[list]:
        """Connected-components clustering by bbox gap: items whose boxes
        are within `threshold` of some other item in the same cluster end
        up together (single-linkage). Uses a spatial hash grid so this
        stays close to linear time on large item counts."""
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
                    uf.union(index, other)

        clusters = _group_by_root(items, uf)
        _LOG.debug(
            "cluster_spatial: %d item(s) -> %d cluster(s) (threshold=%s)",
            len(items),
            len(clusters),
            threshold,
        )
        return clusters

    def _split_group_pairwise(
        self,
        group: list,
        close_fn: Callable[[Any, Any], bool],
    ) -> list[list]:
        n = len(group)
        if n <= 1:
            return [group]
        if n > _MAX_GROUP_SIZE_FOR_PAIRWISE:
            _LOG.warning(
                "group of %d item(s) too large for pairwise clustering; "
                "keeping as one cluster",
                n,
            )
            return [group]

        uf = _UnionFind(n)
        for i in range(n):
            for j in range(i + 1, n):
                if close_fn(group[i], group[j]):
                    uf.union(i, j)
        return _group_by_root(group, uf)

    def cluster_by_dimension(
        self,
        groups: list[list],
        get_bbox: Callable[[Any], tuple],
        *,
        tolerance: float = 0.35,
    ) -> list[list]:
        """Split each spatial cluster further by height/width similarity
        (relative difference <= tolerance on both axes)."""

        def close(a, b) -> bool:
            ax0, ay0, ax1, ay1 = get_bbox(a)
            bx0, by0, bx1, by1 = get_bbox(b)
            aw, ah = ax1 - ax0, ay1 - ay0
            bw, bh = bx1 - bx0, by1 - by0
            dw = abs(aw - bw) / max(aw, bw, 1e-6)
            dh = abs(ah - bh) / max(ah, bh, 1e-6)
            return dw <= tolerance and dh <= tolerance

        result: list[list] = []
        for group in groups:
            result.extend(self._split_group_pairwise(group, close))

        _LOG.debug(
            "cluster_by_dimension: %d group(s) -> %d cluster(s) (tolerance=%s)",
            len(groups),
            len(result),
            tolerance,
        )
        return result

    def cluster_by_seq(
        self,
        groups: list[list],
        get_seq: Callable[[Any], int],
        *,
        max_gap: int = 3,
    ) -> list[list]:
        """Split each dimension cluster further by drawing sequence-number
        proximity: consecutive (sorted by seq) items more than `max_gap`
        apart in draw order start a new cluster."""
        result: list[list] = []

        for group in groups:
            if len(group) <= 1:
                result.append(group)
                continue

            ordered = sorted(group, key=get_seq)
            current = [ordered[0]]
            for prev, item in zip(ordered, ordered[1:]):
                if get_seq(item) - get_seq(prev) <= max_gap:
                    current.append(item)
                else:
                    result.append(current)
                    current = [item]
            result.append(current)

        _LOG.debug(
            "cluster_by_seq: %d group(s) -> %d cluster(s) (max_gap=%s)",
            len(groups),
            len(result),
            max_gap,
        )
        return result

    def cluster_by_pairwise_distance(
        self,
        groups: list[list],
        distance_fn: Callable[[Any, Any], float],
        *,
        threshold: float,
    ) -> list[list]:
        """Generic pairwise merge: within each incoming group, items with
        `distance_fn(a, b) <= threshold` end up together (single-linkage,
        via the same O(k^2) union-find pass as cluster_by_dimension/
        group_by_overlap -- see _split_group_pairwise). Those two methods
        are themselves expressible as this with a specific distance_fn, but
        keep their own hardcoded bbox-similarity/overlap logic unchanged
        (own tests, own callers); this is the metric-driven classification
        engine's (rastervec/vector_classification/) single generic entry
        point instead of duplicating _split_group_pairwise's wiring a third
        time."""

        def close(a, b) -> bool:
            return distance_fn(a, b) <= threshold

        result: list[list] = []
        for group in groups:
            result.extend(self._split_group_pairwise(group, close))

        _LOG.debug(
            "cluster_by_pairwise_distance: %d group(s) -> %d cluster(s) (threshold=%s)",
            len(groups), len(result), threshold,
        )
        return result

    def group_by_overlap(
        self,
        groups: list[list],
        get_bbox: Callable[[Any], tuple],
        *,
        tolerance: float = 0.0,
    ) -> list[list]:
        """Merge items whose bboxes overlap, touch, or are within
        `tolerance` of each other into the same group; items where one bbox
        fully contains/equals the other are left un-merged (a
        fully-enclosed item, e.g. text sitting inside a drawing's frame, is
        a distinct thing, not part of the same shape) regardless of
        tolerance. Applied per incoming group, same calling convention as
        cluster_by_dimension/cluster_by_seq -- items in different incoming
        groups are never compared against each other."""

        def close(a, b) -> bool:
            return _bboxes_close_or_overlapping(get_bbox(a), get_bbox(b), tolerance)

        result: list[list] = []
        for group in groups:
            result.extend(self._split_group_pairwise(group, close))

        _LOG.debug(
            "group_by_overlap: %d group(s) -> %d cluster(s) (tolerance=%s)",
            len(groups), len(result), tolerance,
        )
        return result

    def cluster_hsv(self, pixels: "np.ndarray") -> list["np.ndarray"]:
        """Cluster image pixels by HSV value into an unknown number of
        color groups, using a predetermined distance threshold.

        Not implemented yet -- Raster-phase concern."""
        raise NotImplementedError
