"""Helpers for `pattern_texture_lab.ipynb` -- experimental texture-fill
signals (stipple dot fields, dense parallel hatching), kept out of
`P3_Vector_Parsing/` until one of them proves useful. Builds on
`_vector_probe_helpers` (extraction, bucketing, geometry, colour, the
`DebugReport` viewer folder).

The unit here is an **element**, not a whole `Vector`: a CAD hatch or
stipple fill is often one `get_drawings()` drawing holding hundreds of
disconnected sub-paths, so `explode_elements` splits each Vector's items
wherever the pen lifts (an item's start != the previous item's end; every
"re"/"qu" is its own element). `Element.vector` is a copy of the parent
Vector holding only that element's items, so debug layers draw exactly it.

- **stipple** -- small elements (`size <= dot_max`). Per candidate, over
  its k nearest candidates (`neighbourhood_stats`): NN-spacing CV,
  isotropy (PCA eigenvalue ratio, 2-D field vs 1-D dotted line), size /
  ink-length CV, same-shape fraction. `grow_regions` links candidates
  within `link_factor` x the local median NN distance into regions, scored
  by the Clark-Evans index R (> 1 regular, ~1 random, < 1 clumped).
- **hatching** -- straight elements, grouped by angle into families
  (`_chain_angles`), collinear pieces merged into tracks (a dashed hatch
  line is one track), then tracks greedily chained into side-by-side runs
  (`hatch_runs`): perpendicular gap <= `max_spacing` *and* overlapping
  along the line direction -- the along-overlap is what keeps an
  end-to-end dashed line from looking like a hatch. Per run: spacing CV,
  density ratio (median spacing / median track length), endpoint-line
  residual (clipped-boundary straightness), orientation dominance (share
  of local ink along the run's -- or its cross-hatch partners' -- angle).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from rastervec.commons.models import Vector
from rastervec.notebooks._vector_probe_helpers import (
    LineFit,
    _anchored_1d,
    _axis_diff,
    _chain_angles,
    _mean_axis_angle,
    straight_line,
    vector_segments,
)

Point = tuple[float, float]
BBox = tuple[float, float, float, float]

_JOIN_EPS = 0.01  # pt: an item starting closer than this to the previous end continues the sub-path


# --------------------------------------------------------------------------
# Elements: one pen-down sub-path of a Vector
# --------------------------------------------------------------------------

def _item_points(item: tuple) -> list[Point]:
    kind = item[0]
    if kind == "l":
        return [tuple(item[1]), tuple(item[2])]
    if kind == "c":
        return [tuple(p) for p in item[1:5]]
    if kind == "re":
        x0, y0, x1, y1 = item[1]
        return [(x0, y0), (x1, y1)]
    if kind == "qu":
        return [tuple(p) for p in item[1]]
    return []


def _item_ends(item: tuple) -> tuple[Point, Point] | None:
    """(start, end) of a path-continuing item, `None` for a closed "re"/"qu"."""
    if item[0] == "l":
        return tuple(item[1]), tuple(item[2])
    if item[0] == "c":
        return tuple(item[1]), tuple(item[4])
    return None


@dataclass
class Element:
    """One sub-path. `size` = longer bbox side (0 for a zero-length
    round-cap dot), `ink_len` = flattened path length, `line` = its
    `LineFit` when straight, `shape` = translation-invariant shape key."""

    parent: Vector
    vector: Vector
    bbox: BBox
    centroid: Point
    size: float
    ink_len: float
    line: LineFit | None
    shape: tuple


def _shape_key(items: list[tuple], origin: Point, q: float = 0.1) -> tuple:
    ox, oy = origin
    return tuple(
        (item[0], tuple((round((x - ox) / q), round((y - oy) / q)) for x, y in _item_points(item)))
        for item in items
    )


def make_element(parent: Vector, items: list[tuple], straight_tol: float = 0.25) -> Element:
    pts = np.asarray([p for item in items for p in _item_points(item)], dtype=float).reshape(-1, 2)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    bbox = (float(x0), float(y0), float(x1), float(y1))
    vec = replace(parent, items=list(items), rect=bbox)
    segs = np.asarray(vector_segments(vec), dtype=float).reshape(-1, 4)
    ink = float(np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]).sum())
    return Element(
        parent=parent, vector=vec, bbox=bbox,
        centroid=((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
        size=max(bbox[2] - bbox[0], bbox[3] - bbox[1]),
        ink_len=ink,
        line=straight_line(vec, straight_tol),
        shape=_shape_key(items, (bbox[0], bbox[1])),
    )


def explode_elements(v: Vector, straight_tol: float = 0.25) -> list[Element]:
    """Split `v.items` into pen-down sub-paths: a new element starts at
    every "re"/"qu" and wherever an "l"/"c" doesn't start where the
    previous item ended."""
    groups: list[list[tuple]] = []
    current: list[tuple] = []
    prev_end: Point | None = None
    for item in v.items:
        ends = _item_ends(item)
        if ends is None:
            if current:
                groups.append(current)
            current, prev_end = [], None
            if _item_points(item):
                groups.append([item])
            continue
        start, end = ends
        if current and (prev_end is None or math.hypot(start[0] - prev_end[0], start[1] - prev_end[1]) > _JOIN_EPS):
            groups.append(current)
            current = []
        current.append(item)
        prev_end = end
    if current:
        groups.append(current)
    return [make_element(v, g, straight_tol) for g in groups]


def explode_all(vectors: list[Vector], straight_tol: float = 0.25) -> list[Element]:
    return [e for v in vectors for e in explode_elements(v, straight_tol)]


def _cv(values) -> float:
    """Coefficient of variation; 0 when the mean is 0, nan when empty."""
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return math.nan
    m = float(a.mean())
    return 0.0 if m == 0 else float(a.std() / m)


def _isotropy(points: np.ndarray) -> float:
    """lambda_min / lambda_max of the points' covariance: ~1 for a 2-D
    blob, ~0 for points on a line."""
    if len(points) < 3:
        return 0.0
    ev = np.linalg.eigvalsh(np.cov(points.T))
    return 0.0 if ev[-1] <= 0 else float(max(ev[0], 0.0) / ev[-1])


def union_bbox(boxes) -> BBox:
    a = np.asarray(list(boxes), dtype=float).reshape(-1, 4)
    return (float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 2].max()), float(a[:, 3].max()))


# --------------------------------------------------------------------------
# A. Stipple / dot fields
# --------------------------------------------------------------------------

def stipple_candidates(elements: list[Element], dot_max: float = 2.0) -> list[Element]:
    return [e for e in elements if e.size <= dot_max]


@dataclass
class NeighbourStats:
    """Per stipple candidate, over itself + its k nearest candidates."""

    nn_dist: float          # distance to its own nearest candidate
    local_median_nn: float  # median `nn_dist` over the neighbourhood
    nn_cv: float            # CV of `nn_dist` over the neighbourhood
    isotropy: float         # PCA eigenvalue ratio of the neighbourhood's centroids
    size_cv: float
    ink_cv: float
    same_shape_frac: float  # share of the neighbourhood with this element's exact shape key


def neighbourhood_stats(cands: list[Element], k: int = 6) -> list[NeighbourStats]:
    from scipy.spatial import cKDTree

    n = len(cands)
    if n < 2:
        return [NeighbourStats(math.inf, math.inf, math.nan, 0.0, math.nan, math.nan, 1.0) for _ in cands]
    pts = np.asarray([e.centroid for e in cands], dtype=float)
    tree = cKDTree(pts)
    kk = min(k, n - 1)
    dist, idx = tree.query(pts, k=kk + 1)
    # Column 0 is normally the point itself; with duplicate centroids it may
    # be a twin, so take the NN distance as the smallest distance to a
    # *different* index.
    nn = np.empty(n)
    for i in range(n):
        others = dist[i][idx[i] != i]
        nn[i] = others[0] if len(others) else math.inf
    sizes = np.asarray([e.size for e in cands])
    inks = np.asarray([e.ink_len for e in cands])
    out = []
    for i in range(n):
        hood = np.unique(np.append(idx[i], i))
        out.append(NeighbourStats(
            nn_dist=float(nn[i]),
            local_median_nn=float(np.median(nn[hood])),
            nn_cv=_cv(nn[hood]),
            isotropy=_isotropy(pts[hood]),
            size_cv=_cv(sizes[hood]),
            ink_cv=_cv(inks[hood]),
            same_shape_frac=float(np.mean([cands[j].shape == cands[i].shape for j in hood])),
        ))
    return out


@dataclass
class StippleRegion:
    members: list[int]      # indices into the candidate list
    bbox: BBox
    count: int
    area: float             # convex-hull area of member centroids (0 if collinear)
    clark_evans: float      # R; 0 when the area is 0 (a 1-D arrangement)
    isotropy: float         # mean of members' local isotropy
    size_cv: float
    shape_frac: float       # share of members carrying the most common shape key
    is_stipple: bool = False


def _hull_area(pts: np.ndarray) -> float:
    if len(pts) < 3:
        return 0.0
    from scipy.spatial import ConvexHull, QhullError

    try:
        return float(ConvexHull(pts).volume)  # 2-D "volume" is the area
    except (QhullError, ValueError):
        return 0.0


def clark_evans(pts: np.ndarray, area: float) -> float:
    """R = observed mean NN distance / expected (0.5 / sqrt(density)) under
    complete spatial randomness. No edge correction (a hull underestimates
    the field, which biases R slightly upwards)."""
    from scipy.spatial import cKDTree

    n = len(pts)
    if n < 2 or area <= 0:
        return 0.0
    d, _ = cKDTree(pts).query(pts, k=2)
    return float(d[:, 1].mean() / (0.5 / math.sqrt(n / area)))


def grow_regions(
    cands: list[Element], stats: list[NeighbourStats], *, link_factor: float = 2.5,
    min_dots: int = 20, r_min: float = 1.0, iso_min: float = 0.3, size_cv_max: float = 0.3,
) -> list[StippleRegion]:
    """Union-find: i and j link when their distance is <= `link_factor` x
    min(local median NN of i, of j) -- a dense field never links into the
    sparse stuff around it. Every component (singletons included) becomes a
    `StippleRegion`; `is_stipple` = count >= `min_dots` and R >= `r_min`
    and isotropy >= `iso_min` and size CV <= `size_cv_max`."""
    from scipy.spatial import cKDTree

    n = len(cands)
    if n == 0:
        return []
    pts = np.asarray([e.centroid for e in cands], dtype=float)
    med = np.asarray([s.local_median_nn for s in stats])
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree = cKDTree(pts)
    for i in range(n):
        if not math.isfinite(med[i]):
            continue
        for j in tree.query_ball_point(pts[i], link_factor * med[i] + 1e-9):
            if j > i and math.hypot(*(pts[i] - pts[j])) <= link_factor * min(med[i], med[j]) + 1e-9:
                parent[find(i)] = find(j)

    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    regions = []
    for members in comps.values():
        p = pts[members]
        area = _hull_area(p)
        shapes: dict[tuple, int] = {}
        for i in members:
            shapes[cands[i].shape] = shapes.get(cands[i].shape, 0) + 1
        r = StippleRegion(
            members=members,
            bbox=union_bbox(cands[i].bbox for i in members),
            count=len(members),
            area=area,
            clark_evans=clark_evans(p, area),
            isotropy=float(np.mean([stats[i].isotropy for i in members])),
            size_cv=_cv([cands[i].size for i in members]),
            shape_frac=max(shapes.values()) / len(members),
        )
        r.is_stipple = (r.count >= min_dots and r.clark_evans >= r_min
                        and r.isotropy >= iso_min and r.size_cv <= size_cv_max)
        regions.append(r)
    return regions


# --------------------------------------------------------------------------
# B. Hatching: side-by-side parallel runs
# --------------------------------------------------------------------------

@dataclass
class Track:
    """Collinear straight elements (one hatch line, possibly dashed), in a
    family's (u, n) frame: `rho` = mean perpendicular offset, `t0..t1` =
    extent along the line."""

    members: list[Element]
    rho: float
    t0: float
    t1: float

    @property
    def span(self) -> float:
        return self.t1 - self.t0


@dataclass
class HatchRun:
    family: int
    angle: float
    tracks: list[Track]
    bbox: BBox
    gaps: list[float]
    spacing_cv: float
    density_ratio: float
    endpoint_residual: float
    dominance: float = math.nan
    partners: list[int] = field(default_factory=list)  # cross-hatch run indices
    is_hatch: bool = False

    @property
    def count(self) -> int:
        return len(self.tracks)

    @property
    def elements(self) -> list[Element]:
        return [e for t in self.tracks for e in t.members]


def parallel_families(elements: list[Element], angle_tol: float = 1.0) -> list[list[Element]]:
    straight = [e for e in elements if e.line is not None]
    return [[straight[i] for i in g] for g in _chain_angles([e.line.angle for e in straight], angle_tol)]


def family_tracks(family: list[Element], collinear_tol: float = 0.5) -> tuple[float, list[Track]]:
    """`(mean angle, tracks)` -- members merged into collinear tracks by
    anchored grouping on `rho`, sorted by `rho`."""
    angle = _mean_axis_angle([e.line.angle for e in family])
    th = math.radians(angle)
    ux, uy, nx, ny = math.cos(th), math.sin(th), -math.sin(th), math.cos(th)
    rhos = [e.line.midpoint[0] * nx + e.line.midpoint[1] * ny for e in family]
    tracks = []
    for g in _anchored_1d(rhos, collinear_tol):
        t0s, t1s = [], []
        for i in g:
            e = family[i]
            tm = e.line.midpoint[0] * ux + e.line.midpoint[1] * uy
            t0s.append(tm - e.line.length / 2.0)
            t1s.append(tm + e.line.length / 2.0)
        tracks.append(Track([family[i] for i in g], float(np.mean([rhos[i] for i in g])), min(t0s), max(t1s)))
    tracks.sort(key=lambda t: t.rho)
    return angle, tracks


def _overlap_frac(a: Track, b: Track) -> float:
    inter = min(a.t1, b.t1) - max(a.t0, b.t0)
    shorter = min(a.span, b.span)
    if shorter <= 0:
        return 1.0 if inter >= 0 else 0.0
    return max(0.0, inter) / shorter


def chain_runs(tracks: list[Track], max_spacing: float = 6.0, min_overlap: float = 0.3) -> list[list[Track]]:
    """Greedy, in `rho` order: each track joins the open run whose last
    track is within `max_spacing` in rho and overlaps it along the line by
    >= `min_overlap` of the shorter span (nearest rho wins); else it opens
    a new run. Runs fall behind (closed) once `max_spacing` is exceeded, so
    two same-angle hatches elsewhere on the page interleave harmlessly."""
    open_runs: list[list[Track]] = []
    closed: list[list[Track]] = []
    for t in tracks:
        still = []
        for run in open_runs:
            (still if t.rho - run[-1].rho <= max_spacing else closed).append(run)
        open_runs = still
        best = None
        for run in open_runs:
            if _overlap_frac(run[-1], t) >= min_overlap:
                gap = t.rho - run[-1].rho
                if best is None or gap < best[0]:
                    best = (gap, run)
        if best is None:
            open_runs.append([t])
        else:
            best[1].append(t)
    return closed + open_runs


def _residual(xs: list[float], ys: list[float]) -> float:
    """Std of ys about the least-squares line through (xs, ys)."""
    if len(xs) < 3:
        return math.nan
    coef = np.polyfit(xs, ys, 1)
    return float(np.std(np.asarray(ys) - np.polyval(coef, xs)))


def make_run(family: int, angle: float, tracks: list[Track]) -> HatchRun:
    rhos = [t.rho for t in tracks]
    gaps = list(np.diff(rhos)) if len(rhos) > 1 else []
    spans = [t.span for t in tracks]
    med_gap = float(np.median(gaps)) if gaps else math.nan
    return HatchRun(
        family=family,
        angle=angle,
        tracks=tracks,
        bbox=union_bbox(e.bbox for t in tracks for e in t.members),
        gaps=[float(g) for g in gaps],
        spacing_cv=_cv(gaps) if len(gaps) >= 2 else math.nan,
        density_ratio=med_gap / float(np.median(spans)) if gaps and np.median(spans) > 0 else math.nan,
        endpoint_residual=float(np.nanmean([_residual(rhos, [t.t0 for t in tracks]),
                                            _residual(rhos, [t.t1 for t in tracks])]))
        if len(tracks) >= 3 else math.nan,
    )


def hatch_runs(
    elements: list[Element], *, angle_tol: float = 1.0, collinear_tol: float = 0.5,
    max_spacing: float = 6.0, min_overlap: float = 0.3,
) -> list[HatchRun]:
    """Every side-by-side run (singleton tracks included) over the straight
    elements of one bucket/cluster; `HatchRun.family` indexes
    `parallel_families`' order."""
    runs = []
    for fi, fam in enumerate(parallel_families(elements, angle_tol)):
        angle, tracks = family_tracks(fam, collinear_tol)
        runs.extend(make_run(fi, angle, r) for r in chain_runs(tracks, max_spacing, min_overlap))
    return runs


def _bbox_area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _bbox_overlap(a: BBox, b: BBox) -> float:
    return _bbox_area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def pair_cross_hatch(runs: list[HatchRun], *, min_count: int = 5, min_angle: float = 10.0,
                     min_overlap: float = 0.5) -> None:
    """Fill `partners`: runs with >= `min_count` tracks, >= `min_angle`
    apart, whose bbox overlap is >= `min_overlap` of the smaller bbox."""
    big = [i for i, r in enumerate(runs) if r.count >= min_count]
    for a_i, i in enumerate(big):
        for j in big[a_i + 1:]:
            a, b = runs[i], runs[j]
            if _axis_diff(a.angle, b.angle) < min_angle:
                continue
            smaller = min(_bbox_area(a.bbox), _bbox_area(b.bbox))
            if smaller > 0 and _bbox_overlap(a.bbox, b.bbox) >= min_overlap * smaller:
                a.partners.append(j)
                b.partners.append(i)


def score_dominance(runs: list[HatchRun], elements: list[Element], angle_tol: float = 1.0,
                    min_count: int = 3) -> None:
    """`dominance` = ink of straight elements (centroid inside the run's
    bbox) along the run's own angle or a cross-hatch partner's, over all
    ink inside the bbox. Runs with < `min_count` tracks are left `nan`
    (scoring every singleton is quadratic on a busy page)."""
    if not elements:
        return
    c = np.asarray([e.centroid for e in elements], dtype=float)
    ink = np.asarray([e.ink_len for e in elements], dtype=float)
    ang = [e.line.angle if e.line is not None else None for e in elements]
    for r in runs:
        if r.count < min_count:
            continue
        x0, y0, x1, y1 = r.bbox
        inside = np.nonzero((c[:, 0] >= x0) & (c[:, 0] <= x1) & (c[:, 1] >= y0) & (c[:, 1] <= y1))[0]
        total = float(ink[inside].sum())
        if total <= 0:
            r.dominance = math.nan
            continue
        dirs = [r.angle] + [runs[p].angle for p in r.partners]
        along = sum(ink[i] for i in inside
                    if ang[i] is not None and any(_axis_diff(ang[i], d) <= angle_tol for d in dirs))
        r.dominance = float(along / total)


def judge_hatch(runs: list[HatchRun], *, min_lines: int = 5, spacing_cv_max: float = 0.25,
                density_max: float = 0.5, dominance_min: float = 0.7) -> None:
    for r in runs:
        r.is_hatch = (r.count >= min_lines
                      and r.spacing_cv <= spacing_cv_max
                      and r.density_ratio <= density_max
                      and r.dominance >= dominance_min)
