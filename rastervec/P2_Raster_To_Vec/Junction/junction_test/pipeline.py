"""The centerline tracer Junction runs on each component crop:

    binarize -> skeletonize + distance transform -> chains (barb pruning)
             -> Douglas-Peucker per chain -> regularize

`run(gray, params) -> PipelineResult` keeps the intermediates for debugging.
Always native resolution (no downscale). Text is OCR'd and erased before
this runs (see `Junction/adapter.py`), so every ink pixel handed in is
traced -- there is no text/graphics separation here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize

from rastervec.commons.logging_setup import get_logger

from .geom import angle_gap, dist, heading_deg
from .simplify import approximate_rdp
from .skeleton_graph import build_graph
from .types_ import Graph, PipelineResult, Point, Segment

_LOG = get_logger("P2.Junction.trace")


@dataclass
class Params:
    # binarisation: Otsu OR gray < soft_ink_thresh; ink blobs smaller than
    # min_ink_area_px (8-connected) are specks and dropped
    soft_ink_thresh: int = 245
    min_ink_area_px: int = 4
    # skeleton graph: leaf branches off a junction shorter than this are barbs
    barb_min_px: float = 9.0
    # Douglas-Peucker tolerance per chain
    rdp_eps_px: float = 1.6
    # regularisation
    snap_px: float = 4.0
    collinear_deg: float = 8.0
    min_segment_px: float = 3.0


# ----------------------------------------------------------------------- stages


def binarize(gray: np.ndarray, p: Params) -> np.ndarray:
    """Grayscale crop -> bool ink mask. Specks are removed by connected-
    component area, not by a morphological open (a 2x2 open erases every
    1-px-wide line, including thin diagonals)."""
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink = (otsu > 0) | (gray < p.soft_ink_thresh)
    if p.min_ink_area_px > 1 and ink.any():
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
        small = stats[:, cv2.CC_STAT_AREA] < p.min_ink_area_px
        small[0] = False  # background label
        if small.any():
            ink &= ~small[labels]
    return ink


def skeleton_and_dt(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = mask.astype(bool)
    sk = skeletonize(m)
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    return sk, dt


def _chain_width(chain: list[Point], dt: np.ndarray) -> float:
    """Stroke width from the distance transform along the centreline. A
    stroke W px wide has DT ~ (W + 1) / 2 at its centre pixel (distance to
    the nearest *background pixel centre*), hence `2 * DT - 1`, not the
    `2 * DT` the original spike used (which overestimated by ~1 px)."""
    pts = np.round(np.asarray(chain, float)).astype(int)
    h, w = dt.shape
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    vals = dt[pts[:, 1], pts[:, 0]]
    return max(1.0, float(np.median(vals) * 2.0 - 1.0)) if len(vals) else 1.0


def _anchor_loop(chain: list[Point]) -> list[Point]:
    """A closed chain (first == last) starts wherever the skeleton walk
    happened to begin, and Douglas-Peucker always keeps that start -- an
    arbitrary vertex that leaves a slanted stub near a corner. Re-start the
    loop at its point farthest from the loop's centroid (a genuine extreme,
    e.g. a rectangle corner) instead -- the same idea as
    `Evaluation/CadFont/graph.py::_force_protect_junctionless_components`
    anchoring a junction-free component at its farthest points."""
    if len(chain) < 4 or chain[0] != chain[-1]:
        return chain
    pts = np.asarray(chain[:-1], float)
    k = int(np.argmax(np.hypot(*(pts - pts.mean(axis=0)).T)))
    ring = chain[:-1]
    rotated = ring[k:] + ring[:k]
    return rotated + [rotated[0]]


def vectorize(graph: Graph, dt: np.ndarray, p: Params) -> list[Segment]:
    """Each chain -> Douglas-Peucker polyline -> straight segments carrying
    the chain's median stroke width."""
    segments: list[Segment] = []
    for chain in graph.chains:
        if len(chain) < 2:
            continue
        poly = approximate_rdp(_anchor_loop(chain), p.rdp_eps_px)
        width = _chain_width(chain, dt)
        for q0, q1 in zip(poly, poly[1:]):
            if dist(q0, q1) >= p.min_segment_px:
                segments.append(Segment(p0=q0, p1=q1, width=width))
    return segments


def regularize(segments: list[Segment], p: Params) -> list[Segment]:
    """Snap endpoints, drop too-short segments and isolated spurs, merge
    collinear pairs sharing an endpoint. Every neighbour query goes through
    a cKDTree (one component can hold thousands of segments at native
    resolution; the original all-pairs loops took minutes)."""
    from scipy.spatial import cKDTree

    if not segments:
        return []

    # snap endpoints
    endpoints = [ep for s in segments for ep in (s.p0, s.p1)]
    snapped = _dedup_points_kd(endpoints, p.snap_px)
    snap_tree = cKDTree(np.asarray(snapped, float))

    def nearest(pt):
        return snapped[int(snap_tree.query(pt)[1])]

    segs = [Segment(nearest(s.p0), nearest(s.p1), s.width) for s in segments]
    segs = [s for s in segs if dist(s.p0, s.p1) >= p.min_segment_px]

    # drop short isolated noise spurs: a segment shorter than barb_min_px
    # with neither endpoint shared by another segment
    if segs:
        ep_tree = cKDTree(np.asarray([ep for s in segs for ep in (s.p0, s.p1)], float))

        def _shared_end(pt, idx):
            return any(n // 2 != idx for n in ep_tree.query_ball_point(pt, p.snap_px))

        segs = [
            s for idx, s in enumerate(segs)
            if dist(s.p0, s.p1) >= p.barb_min_px
            or _shared_end(s.p0, idx) or _shared_end(s.p1, idx)
        ]

    return _merge_collinear(segs, p)


def _dedup_points_kd(points: list[Point], tol: float) -> list[Point]:
    """Greedy point clustering: each unused point in order absorbs every
    still-unused point within `tol`; a cluster becomes its mean."""
    from scipy.spatial import cKDTree

    if not points:
        return []
    arr = np.asarray(points, float)
    tree = cKDTree(arr)
    used = np.zeros(len(points), dtype=bool)
    out: list[Point] = []
    for i in range(len(points)):
        if used[i]:
            continue
        members = [j for j in tree.query_ball_point(arr[i], tol) if not used[j]]
        used[members] = True
        c = arr[members].mean(axis=0)
        out.append((float(c[0]), float(c[1])))
    return out


def _shared_point(a: Segment, b: Segment, tol: float) -> "Point | None":
    for pa in (a.p0, a.p1):
        for pb in (b.p0, b.p1):
            if dist(pa, pb) <= tol:
                return pa
    return None


def _away_from(s: Segment, at: Point) -> tuple[Point, Point]:
    """`s` oriented to start at whichever endpoint is nearer `at`."""
    return (s.p0, s.p1) if dist(s.p0, at) <= dist(s.p1, at) else (s.p1, s.p0)


def _try_merge(a: Segment, b: Segment, p: Params) -> "Segment | None":
    """Merge `a` and `b` into one segment if they share an endpoint and
    continue straight through it: `a`'s heading *into* the shared point and
    `b`'s heading *out of* it agree within `collinear_deg`. (The original
    spike compared both headings *away* from the shared point, which merged
    segments folding back on each other and never merged a real straight
    continuation.)"""
    shared = _shared_point(a, b, p.snap_px)
    if shared is None:
        return None
    a_near, a_far = _away_from(a, shared)
    b_near, b_far = _away_from(b, shared)
    heading_in = heading_deg(a_far, a_near)
    heading_out = heading_deg(b_near, b_far)
    if angle_gap(heading_in, heading_out, 360.0) > p.collinear_deg:
        return None
    return Segment(a_far, b_far, (a.width + b.width) / 2)


def _merge_collinear(segs: list[Segment], p: Params) -> list[Segment]:
    """Merge near-collinear pairs sharing an endpoint until none remain.
    Pass-based: each pass indexes all endpoints in a cKDTree and merges many
    disjoint pairs (lowest index first, each segment at most once per
    pass)."""
    from scipy.spatial import cKDTree

    segs = list(segs)
    while len(segs) > 1:
        tree = cKDTree(np.asarray([ep for s in segs for ep in (s.p0, s.p1)], float))
        touched = [False] * len(segs)
        alive = [True] * len(segs)
        merged_any = False
        for i in range(len(segs)):
            if touched[i] or not alive[i]:
                continue
            a = segs[i]
            cands = sorted({
                n // 2 for ep in (a.p0, a.p1) for n in tree.query_ball_point(ep, p.snap_px)
            })
            for j in cands:
                if j <= i or touched[j] or not alive[j]:
                    continue
                m = _try_merge(a, segs[j], p)
                if m is not None:
                    segs[i] = m
                    alive[j] = False
                    touched[i] = touched[j] = True
                    merged_any = True
                    break
        segs = [s for s, keep in zip(segs, alive) if keep]
        if not merged_any:
            break
    return segs


# ----------------------------------------------------------------------- driver


def run(gray: np.ndarray, params: Params | None = None) -> PipelineResult:
    p = params or Params()
    t: dict[str, float] = {}

    def _t(name, fn):
        s = time.perf_counter()
        r = fn()
        t[name] = time.perf_counter() - s
        return r

    _LOG.debug("trace: %dx%d px crop", gray.shape[1], gray.shape[0])
    ink = _t("binarize", lambda: binarize(gray, p))
    _LOG.debug("  binarize: %d ink px (%.4fs)", int(ink.sum()), t["binarize"])
    skeleton, dist_map = _t("skeleton", lambda: skeleton_and_dt(ink))
    _LOG.debug("  skeleton: %d skeleton px (%.4fs)", int(skeleton.sum()), t["skeleton"])
    graph = _t("graph", lambda: build_graph(skeleton, p.barb_min_px))
    _LOG.debug(
        "  graph: %d node(s), %d chain(s) (%.4fs)",
        len(graph.nodes), len(graph.chains), t["graph"],
    )
    segments = _t("vectorize", lambda: vectorize(graph, dist_map, p))
    _LOG.debug("  vectorize: %d segment(s) (%.4fs)", len(segments), t["vectorize"])
    segments = _t("regularize", lambda: regularize(segments, p))
    _LOG.debug("  regularize: %d segment(s) (%.4fs)", len(segments), t["regularize"])
    return PipelineResult(
        params=p, ink=ink, skeleton=skeleton, dist_map=dist_map,
        graph=graph, segments=segments, timings=t,
    )
