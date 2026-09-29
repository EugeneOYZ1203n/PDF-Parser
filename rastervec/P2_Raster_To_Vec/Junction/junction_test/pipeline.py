"""The classical raster->vector pipeline (Dosch 2000 Sec 2-3, no 3D).

run(gray, params) -> PipelineResult  keeps every intermediate for the notebook.

Text is no longer separated here: the Fletcher & Kasturi connected-component
text/graphics split (and the dashed-dash reclaim that undid its worst
mistakes) was removed -- it silently discarded every small compact blob,
including isolated stair treads, ticks and dots. Text is now OCR'd and
erased *before* this runs (see `Junction/adapter.py`), so every ink pixel
handed to `run` is traced.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import medial_axis, reconstruction, skeletonize

from . import dashed, polyapprox, staircase, symbols
from .geom import angle_gap, classify_junction, dist, heading_deg
from .skeleton_graph import build_graph
from .types_ import Arc, Graph, Junction, PipelineResult, Point, Segment, StaircaseRegion, SymbolInstance


@dataclass
class Params:
    # working resolution: if set, downscale so the longer side <= this before
    # analysis. Off by default -- fine detail (stair treads, thin hatching)
    # needs native resolution; `Junction/components.py` keeps the per-call
    # size down instead by tracing one spatial component at a time.
    max_work_px: int | None = None
    # binarisation
    soft_ink_thresh: int = 245
    # thick / thin
    thick_min_px: int | None = None      # None -> auto from distance transform
    # centerline / primitive extraction (JUNCTION_ABLATION.md Sec 2 "S")
    centerline_method: str = "skeleton"  # skeleton | lsd | hough
    skeleton_method: str = "skeletonize"  # skeletonize (S1) | medial_axis (S2)
    junction_repair: bool = False        # S3: erase disk at deg>=3 nodes, re-pair stubs
    lsd_min_len_px: float = 12.0
    hough_thresh: int = 40
    hough_min_len_px: float = 20.0
    hough_max_gap_px: float = 6.0
    # skeleton graph
    barb_min_px: float = 9.0
    # polygonal approximation
    approx_method: str = "rosin_west"    # "rosin_west" | "rdp"
    rdp_eps_px: float = 1.6
    rosin_min_significance: float = 0.05
    # arc detection
    arc_angle_tol_deg: float = 18.0
    arc_fit_tol_px: float = 2.4
    arc_min_radius: float = 6.0
    # dashed lines
    dash_max_len: float = 22.0
    dash_max_gap: float = 26.0
    dash_min_count: int = 3
    # regularisation
    snap_px: float = 4.0
    collinear_deg: float = 8.0
    min_segment_px: float = 3.0
    # remainder
    remainder_dilate_px: int = 3
    # staircase detection (Sec 3.2)
    detect_staircases: bool = True
    stair_tread_min_len: float = 10.0
    stair_tread_max_len: float = 90.0
    stair_angle_tol_deg: float = 10.0
    stair_len_ratio_tol: float = 0.35
    stair_spacing_min: float = 6.0
    stair_spacing_max: float = 40.0
    stair_lateral_tol: float = 6.0
    stair_min_treads: int = 5
    stair_max_treads: int = 30
    # symbol recognition (Sec 3.3)
    detect_symbols: bool = True
    symbol_families: tuple[str, ...] = ("door", "window")
    symbol_max_error: float = 1.0


# ----------------------------------------------------------------------- stages


def binarize(gray: np.ndarray, p: Params) -> np.ndarray:
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    soft = (gray < p.soft_ink_thresh).astype(np.uint8) * 255
    ink = cv2.bitwise_or(otsu, soft)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return ink > 0


def thick_thin(graphics: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray, int]:
    g = graphics.astype(bool)
    if not g.any():
        return np.zeros_like(g), np.zeros_like(g), 0
    _, dt = medial_axis(g, return_distance=True)
    ridge = dt[g & (dt > 0)]
    auto_w = int(np.clip(np.median(ridge) * 2.0, 3, 40)) if ridge.size else 3
    w = p.thick_min_px if p.thick_min_px is not None else auto_w
    w = max(3, int(w))
    n = w // 2
    if n < 1:
        return np.zeros_like(g), g, w
    b_n = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * n + 1, 2 * n + 1))
    seed = cv2.erode(g.astype(np.uint8), b_n).astype(bool)
    thick = reconstruction(seed, g, method="dilation").astype(bool)
    thin = g & ~thick
    return thick, thin, w


def skeleton_and_dt(mask: np.ndarray, method: str = "skeletonize") -> tuple[np.ndarray, np.ndarray]:
    m = mask.astype(bool)
    if method == "medial_axis":            # S2: subpixel-width medial axis
        sk = medial_axis(m)
    else:                                  # S1: Zhang-Suen / Lee thinning
        sk = skeletonize(m)
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    return sk, dt


def repair_junctions(skeleton: np.ndarray, dist_map: np.ndarray, p: Params) -> np.ndarray:
    """S3: erase a disk (r ~ local stroke half-width) around every degree>=3
    skeleton node so the graph builder re-pairs the stubs by direction instead
    of collapsing the crossing to one distorted node."""
    from .skeleton_graph import _degree_map
    sk = skeleton.astype(bool).copy()
    deg = _degree_map(sk)
    ys, xs = np.nonzero(sk & (deg >= 3))
    h, w = sk.shape
    for y, x in zip(ys.tolist(), xs.tolist()):
        r = int(max(2, round(dist_map[y, x])))
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        sk[y0:y1, x0:x1] &= ((yy - y) ** 2 + (xx - x) ** 2) > r * r
    return sk


def _seg_width(p0: Point, p1: Point, dt: np.ndarray) -> float:
    n = max(2, int(dist(p0, p1)))
    xs = np.linspace(p0[0], p1[0], n)
    ys = np.linspace(p0[1], p1[1], n)
    h, w = dt.shape
    xi = np.clip(np.round(xs).astype(int), 0, w - 1)
    yi = np.clip(np.round(ys).astype(int), 0, h - 1)
    v = dt[yi, xi]
    v = v[v > 0]
    return float(np.median(v) * 2.0) if v.size else 1.0


def detect_segments_lsd(graphics: np.ndarray, dt: np.ndarray, p: Params) -> list[Segment]:
    """S8: LSD straight-segment detector directly on the raster (skip skeleton)."""
    g = (graphics.astype(np.uint8)) * 255
    try:
        lsd = cv2.createLineSegmentDetector()
        lines = lsd.detect(g)[0]
    except Exception:
        lines = None
    segs: list[Segment] = []
    if lines is not None:
        for ln in lines.reshape(-1, 4):
            p0, p1 = (float(ln[0]), float(ln[1])), (float(ln[2]), float(ln[3]))
            if dist(p0, p1) >= p.lsd_min_len_px:
                segs.append(Segment(p0=p0, p1=p1, width=_seg_width(p0, p1, dt)))
    else:                                  # LSD unavailable in this opencv build
        segs = detect_lines_hough(graphics, dt, p)
    return segs


def detect_lines_hough(graphics: np.ndarray, dt: np.ndarray, p: Params) -> list[Segment]:
    """S9: progressive probabilistic Hough lines."""
    g = (graphics.astype(np.uint8)) * 255
    lines = cv2.HoughLinesP(g, 1, np.pi / 360, p.hough_thresh,
                            minLineLength=p.hough_min_len_px, maxLineGap=p.hough_max_gap_px)
    segs: list[Segment] = []
    if lines is not None:
        for ln in lines.reshape(-1, 4):
            p0, p1 = (float(ln[0]), float(ln[1])), (float(ln[2]), float(ln[3]))
            segs.append(Segment(p0=p0, p1=p1, width=_seg_width(p0, p1, dt)))
    return segs


def detect_circles_hough(graphics: np.ndarray, p: Params) -> list[Arc]:
    g = (graphics.astype(np.uint8)) * 255
    circ = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.5, minDist=20,
                            param1=120, param2=40, minRadius=int(p.arc_min_radius), maxRadius=0)
    arcs: list[Arc] = []
    if circ is not None:
        for cx, cy, r in circ.reshape(-1, 3):
            t = np.linspace(0, 2 * np.pi, 64)
            poly = [(float(cx + r * np.cos(a)), float(cy + r * np.sin(a))) for a in t]
            arcs.append(Arc(center=(float(cx), float(cy)), radius=float(r),
                            a0=0.0, a1=360.0, polyline=poly, closed=True))
    return arcs


def _polyline_of(chain: list[Point], p: Params) -> list[Point]:
    if p.approx_method == "rdp":
        return polyapprox.approximate_rdp(chain, p.rdp_eps_px)
    return polyapprox.approximate_rosin_west(chain, p.rosin_min_significance)


def _chain_width(chain: list[Point], dt: np.ndarray) -> float:
    pts = np.round(np.asarray(chain, float)).astype(int)
    h, w = dt.shape
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    vals = dt[pts[:, 1], pts[:, 0]]
    return float(np.median(vals) * 2.0) if len(vals) else 1.0


def vectorize(graph: Graph, dt: np.ndarray, thick_mask: np.ndarray, p: Params):
    segments: list[Segment] = []
    arcs: list[Arc] = []
    polylines: list[list[Point]] = []
    for chain in graph.chains:
        if len(chain) < 2:
            continue
        poly = _polyline_of(chain, p)
        polylines.append(poly)
        width = _chain_width(chain, dt)
        is_thick = _fraction_in_mask(chain, thick_mask) > 0.5
        chain_arcs, straights = polyapprox.detect_arcs(
            poly, chain,
            angle_tol_deg=p.arc_angle_tol_deg,
            fit_tol_px=p.arc_fit_tol_px,
            min_radius=p.arc_min_radius,
        )
        for a in chain_arcs:
            a.width = width
            arcs.append(a)
        for (q0, q1) in straights:
            if dist(q0, q1) >= p.min_segment_px:
                segments.append(Segment(p0=q0, p1=q1, width=width, thick=is_thick))
    return segments, arcs, polylines


def _fraction_in_mask(chain: list[Point], mask: np.ndarray) -> float:
    if mask is None or not mask.any():
        return 0.0
    pts = np.round(np.asarray(chain, float)).astype(int)
    h, w = mask.shape
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    if not inside.any():
        return 0.0
    pts = pts[inside]
    return float(mask[pts[:, 1], pts[:, 0]].mean())


def regularize(segments: list[Segment], p: Params) -> tuple[list[Segment], list[Junction]]:
    if not segments:
        return [], []
    # Every neighbour query below goes through a cKDTree: at native
    # resolution one component can hold thousands of segments, and the
    # original all-pairs loops (O(n^2) snapping/spur checks, an O(n^3)
    # restart-after-every-merge collinear merge) took minutes per page.
    from scipy.spatial import cKDTree

    # snap endpoints
    endpoints = [ep for s in segments for ep in (s.p0, s.p1)]
    snapped = _dedup_points_kd(endpoints, p.snap_px)
    snap_tree = cKDTree(np.asarray(snapped, float))

    def nearest(pt):
        return snapped[int(snap_tree.query(pt)[1])]

    segs = [Segment(nearest(s.p0), nearest(s.p1), s.width, s.thick, s.dashed) for s in segments]
    segs = [s for s in segs if dist(s.p0, s.p1) >= p.min_segment_px]

    # drop short isolated noise spurs: a non-dashed segment shorter than
    # barb_min_px with neither endpoint shared by another segment
    if segs:
        ep_tree = cKDTree(np.asarray([ep for s in segs for ep in (s.p0, s.p1)], float))

        def _shared_end(pt, idx):
            return any(n // 2 != idx for n in ep_tree.query_ball_point(pt, p.snap_px))

        segs = [
            s for idx, s in enumerate(segs)
            if s.dashed
            or dist(s.p0, s.p1) >= p.barb_min_px
            or _shared_end(s.p0, idx) or _shared_end(s.p1, idx)
        ]

    segs = _merge_collinear(segs, p)

    # junctions = snapped points where >=2 non-collinear segment ends meet
    incidence: dict[Point, list[float]] = {}
    for s in segs:
        if s.dashed:
            continue
        for ep, other in ((s.p0, s.p1), (s.p1, s.p0)):
            key = nearest(ep)
            incidence.setdefault(key, []).append(heading_deg(ep, other))
    junctions = []
    for k, headings in incidence.items():
        if len(headings) >= 3 or (
            len(headings) == 2 and angle_gap(headings[0], headings[1], 360.0) > p.collinear_deg
        ):
            junctions.append(Junction(
                xy=k, directions=headings,
                jtype=classify_junction(headings, p.collinear_deg * 2.5),
                arm_angles=sorted(float(h) % 360.0 for h in headings),
            ))
    return segs, junctions


def _dedup_points_kd(points: list[Point], tol: float) -> list[Point]:
    """`geom.dedup_points` (greedy: each unused point in order absorbs every
    still-unused point within `tol`, cluster -> its mean), with the
    neighbour scan done by a cKDTree instead of all pairs. Same result."""
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


def _try_merge(a: Segment, b: Segment, p: Params) -> "Segment | None":
    """The original pairwise collinear-merge rule for one (a, b) pair."""
    if a.dashed != b.dashed:
        return None
    shared = _shared_point(a, b, p.snap_px)
    if shared is None:
        return None
    ha = heading_deg(*_oriented(a, shared))
    hb = heading_deg(*_oriented(b, shared, away=True))
    if angle_gap(ha, hb, 360.0) > p.collinear_deg:
        return None
    pa = a.p1 if _close(a.p0, shared, p.snap_px) else a.p0
    pb = b.p1 if _close(b.p0, shared, p.snap_px) else b.p0
    return Segment(pa, pb, (a.width + b.width) / 2, a.thick or b.thick, a.dashed)


def _merge_collinear(segs: list[Segment], p: Params) -> list[Segment]:
    """Merge near-collinear pairs sharing an endpoint until none remain.
    Pass-based: each pass indexes all endpoints in a cKDTree and merges
    many disjoint pairs (lowest index first, each segment at most once per
    pass), instead of the original restart-from-scratch after every single
    merge."""
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


def _close(a, b, tol):
    return dist(a, b) <= tol


def _shared_point(a: Segment, b: Segment, tol):
    for pa in (a.p0, a.p1):
        for pb in (b.p0, b.p1):
            if dist(pa, pb) <= tol:
                return pa
    return None


def _oriented(s: Segment, at: Point, away: bool = False):
    if _close(s.p0, at, 1e6) and dist(s.p0, at) <= dist(s.p1, at):
        p0, p1 = s.p0, s.p1
    else:
        p0, p1 = s.p1, s.p0
    return (p0, p1) if not away else (p0, p1)


def detect_staircase_regions(segments: list[Segment], p: Params) -> list[StaircaseRegion]:
    if not p.detect_staircases:
        return []
    return staircase.detect(
        segments,
        tread_min_len=p.stair_tread_min_len, tread_max_len=p.stair_tread_max_len,
        angle_tol_deg=p.stair_angle_tol_deg, len_ratio_tol=p.stair_len_ratio_tol,
        spacing_min=p.stair_spacing_min, spacing_max=p.stair_spacing_max,
        lateral_tol=p.stair_lateral_tol, min_treads=p.stair_min_treads, max_treads=p.stair_max_treads,
    )


def recognize_symbols(segments: list[Segment], arcs: list[Arc], p: Params) -> list[SymbolInstance]:
    if not p.detect_symbols:
        return []
    return symbols.recognize(segments, arcs, families=p.symbol_families, max_error=p.symbol_max_error)


def render_geometry(shape, segments, arcs, p: Params) -> np.ndarray:
    canvas = np.zeros(shape, np.uint8)
    for s in segments:
        cv2.line(canvas, _i(s.p0), _i(s.p1), 255, max(1, int(round(s.width))))
    for a in arcs:
        pts = np.round(np.asarray(a.polyline)).astype(np.int32)
        cv2.polylines(canvas, [pts], a.closed, 255, max(1, int(round(a.width))))
    return canvas > 0


def _i(pt):
    return (int(round(pt[0])), int(round(pt[1])))


def extract_remainder(ink, segments, arcs, p: Params) -> np.ndarray:
    covered = render_geometry(ink.shape, segments, arcs, p).astype(np.uint8)
    if p.remainder_dilate_px > 0:
        k = 2 * p.remainder_dilate_px + 1
        covered = cv2.dilate(covered, np.ones((k, k), np.uint8))
    return ink & ~(covered > 0)


# ----------------------------------------------------------------------- driver


def run(gray: np.ndarray, params: Params | None = None) -> PipelineResult:
    p = params or Params()
    t = {}
    long_side = max(gray.shape)
    if p.max_work_px and long_side > p.max_work_px:
        s = p.max_work_px / long_side
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    def _t(name, fn):
        s = time.perf_counter()
        r = fn()
        t[name] = time.perf_counter() - s
        return r

    ink = _t("binarize", lambda: binarize(gray, p))
    graphics_mask = ink
    thick_mask, thin_mask, _w = _t("thick_thin", lambda: thick_thin(graphics_mask, p))
    skeleton, dist_map = _t("skeleton", lambda: skeleton_and_dt(graphics_mask, p.skeleton_method))
    if p.junction_repair:
        skeleton = _t("junction_repair", lambda: repair_junctions(skeleton, dist_map, p))

    if p.centerline_method == "skeleton":
        graph = _t("graph", lambda: build_graph(skeleton, p.barb_min_px))
        segments, arcs, polylines = _t("vectorize", lambda: vectorize(graph, dist_map, thick_mask, p))
    elif p.centerline_method == "lsd":
        graph = Graph(nodes=[], chains=[])
        segments = _t("lsd", lambda: detect_segments_lsd(graphics_mask, dist_map, p))
        arcs = _t("circles", lambda: detect_circles_hough(graphics_mask, p))
        polylines = [[s.p0, s.p1] for s in segments]
    elif p.centerline_method == "hough":
        graph = Graph(nodes=[], chains=[])
        segments = _t("hough", lambda: detect_lines_hough(graphics_mask, dist_map, p))
        arcs = _t("circles", lambda: detect_circles_hough(graphics_mask, p))
        polylines = [[s.p0, s.p1] for s in segments]
    else:
        raise ValueError(f"unknown centerline_method: {p.centerline_method}")
    segments = _t("dashed", lambda: dashed.detect(
        segments,
        dash_max_len=p.dash_max_len,
        dash_max_gap=p.dash_max_gap,
        dash_min_count=p.dash_min_count,
    ))
    segments, junctions = _t("regularize", lambda: regularize(segments, p))
    staircases = _t("staircases", lambda: detect_staircase_regions(segments, p))
    symbols_found = _t("symbols", lambda: recognize_symbols(segments, arcs, p))
    remainder = _t("remainder", lambda: extract_remainder(ink, segments, arcs, p))

    return PipelineResult(
        params=p, gray=gray, ink=ink, graphics_mask=graphics_mask,
        thick_mask=thick_mask, thin_mask=thin_mask, skeleton=skeleton, dist_map=dist_map,
        graph=graph, polylines=polylines, segments=segments, arcs=arcs, junctions=junctions,
        remainder=remainder, timings=t,
        staircases=staircases, symbols=symbols_found,
    )
