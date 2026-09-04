"""The classical raster->vector pipeline (Dosch 2000 Sec 2-3, no 3D).

run(gray, params) -> PipelineResult  keeps every intermediate for the notebook.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import medial_axis, reconstruction, skeletonize

from . import dashed, polyapprox, staircase, symbols
from .geom import angle_gap, classify_junction, dedup_points, dist, heading_deg
from .skeleton_graph import build_graph
from .types_ import Arc, Graph, Junction, PipelineResult, Point, Segment, StaircaseRegion, SymbolInstance


@dataclass
class Params:
    # working resolution: downscale so the longer side <= this before analysis
    # (the pure-python skeleton graph is O(pixels); large scans are otherwise slow)
    max_work_px: int = 1600
    # binarisation
    soft_ink_thresh: int = 245
    # text / graphics separation (Fletcher & Kasturi)
    text_max_dim: int = 34
    text_min_dim: int = 3
    text_min_fill: float = 0.18
    text_max_fill: float = 0.98
    text_max_aspect: float = 8.0
    run_ocr: bool = False
    # thick / thin
    thick_min_px: int | None = None      # None -> auto from distance transform
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


def separate_text_graphics(ink: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    text = np.zeros_like(ink)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        long_dim = max(w, h)
        short_dim = max(min(w, h), 1)
        fill = area / float(w * h)
        aspect = long_dim / float(short_dim)
        if (
            p.text_min_dim <= long_dim <= p.text_max_dim
            and short_dim >= 1
            and p.text_min_fill <= fill <= p.text_max_fill
            and aspect <= p.text_max_aspect
        ):
            text |= labels == i
    graphics = ink & ~text
    return text, graphics


def reclaim_dashed_from_text(
    text_mask: np.ndarray, graphics_mask: np.ndarray, p: Params
) -> tuple[np.ndarray, np.ndarray, list[Segment]]:
    """Dosch 2000 Sec 3.1 note: Fletcher & Kasturi dumps dashed-line dashes into
    the text layer. Recover them: treat small text CCs as proto-segments, run the
    Dov Dori run-grower, and move any CC that lands in a >=DASH_MIN_COUNT
    collinear evenly-spaced run back into graphics (emitting one dashed Segment)."""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(text_mask.astype(np.uint8), 8)
    protos: list[Segment] = []
    proto_labels: list[int] = []
    small = [i for i in range(1, n) if max(stats[i][2], stats[i][3]) <= p.dash_max_len]
    if len(small) > 400:
        return text_mask, graphics_mask, []   # text-dense page -> skip (see below)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if max(w, h) > p.dash_max_len or area < 2:
            continue
        cx, cy = cents[i]
        axis = np.array([1.0, 0.0]) if w >= h else np.array([0.0, 1.0])
        half = max(w, h) / 2.0
        protos.append(Segment(p0=(cx - axis[0] * half, cy - axis[1] * half),
                              p1=(cx + axis[0] * half, cy + axis[1] * half),
                              width=float(min(w, h) or 1)))
        proto_labels.append(i)

    if len(protos) < p.dash_min_count:
        return text_mask, graphics_mask, []

    collapsed = dashed.detect(
        protos, dash_max_len=p.dash_max_len, dash_max_gap=p.dash_max_gap,
        dash_min_count=p.dash_min_count,
    )
    dashed_segs = [s for s in collapsed if s.dashed]
    if not dashed_segs:
        return text_mask, graphics_mask, []

    # which proto CCs were consumed into a dashed run?
    from .geom import point_to_segment_dist
    tm = text_mask.copy()
    gm = graphics_mask.copy()
    for ds in dashed_segs:
        for proto, lab in zip(protos, proto_labels):
            mid = ((proto.p0[0] + proto.p1[0]) / 2, (proto.p0[1] + proto.p1[1]) / 2)
            if point_to_segment_dist(mid, ds.p0, ds.p1) <= 4.0:
                comp = labels == lab
                gm |= comp
                tm &= ~comp
    return tm, gm, dashed_segs


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


def skeleton_and_dt(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = mask.astype(bool)
    sk = skeletonize(m)
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    return sk, dt


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
    # snap endpoints
    endpoints = [ep for s in segments for ep in (s.p0, s.p1)]
    snapped = dedup_points(endpoints, p.snap_px)

    def nearest(pt):
        d = [dist(pt, q) for q in snapped]
        return snapped[int(np.argmin(d))]

    segs = [Segment(nearest(s.p0), nearest(s.p1), s.width, s.thick, s.dashed) for s in segments]
    segs = [s for s in segs if dist(s.p0, s.p1) >= p.min_segment_px]

    # drop short isolated noise spurs: a non-dashed segment shorter than
    # barb_min_px with neither endpoint shared by another segment
    def _shared_end(pt, exclude):
        return any(
            o is not exclude and (dist(pt, o.p0) <= p.snap_px or dist(pt, o.p1) <= p.snap_px)
            for o in segs
        )

    segs = [
        s for s in segs
        if s.dashed
        or dist(s.p0, s.p1) >= p.barb_min_px
        or _shared_end(s.p0, s) or _shared_end(s.p1, s)
    ]

    # merge near-collinear pairs sharing an endpoint
    merged = True
    while merged:
        merged = False
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                a, b = segs[i], segs[j]
                if a.dashed != b.dashed:
                    continue
                shared = _shared_point(a, b, p.snap_px)
                if shared is None:
                    continue
                ha = heading_deg(*_oriented(a, shared))
                hb = heading_deg(*_oriented(b, shared, away=True))
                if angle_gap(ha, hb, 360.0) <= p.collinear_deg:
                    pa = a.p1 if _close(a.p0, shared, p.snap_px) else a.p0
                    pb = b.p1 if _close(b.p0, shared, p.snap_px) else b.p0
                    segs[i] = Segment(pa, pb, (a.width + b.width) / 2,
                                      a.thick or b.thick, a.dashed)
                    segs.pop(j)
                    merged = True
                    break
            if merged:
                break

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


def ocr_text_boxes(gray: np.ndarray, text_mask: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Optional: PaddleOCR box detection over the page (used to mask text)."""
    try:
        from rastervec.OCR.Paddle_OCR.render_ocr import RenderOCR  # type: ignore
        from PIL import Image
    except Exception:
        return []
    img = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    det = RenderOCR().backend.detect(np.array(img))
    out = []
    for b in det.boxes:
        xs = [c[0] for c in b.corners]
        ys = [c[1] for c in b.corners]
        out.append((min(xs), min(ys), max(xs), max(ys)))
    return out


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
    text_mask, graphics_mask = _t("text_graphics", lambda: separate_text_graphics(ink, p))
    text_mask, graphics_mask, reclaimed_dashed = _t(
        "reclaim_dashed", lambda: reclaim_dashed_from_text(text_mask, graphics_mask, p)
    )
    thick_mask, thin_mask, _w = _t("thick_thin", lambda: thick_thin(graphics_mask, p))
    skeleton, dist_map = _t("skeleton", lambda: skeleton_and_dt(graphics_mask))
    graph = _t("graph", lambda: build_graph(skeleton, p.barb_min_px))
    segments, arcs, polylines = _t("vectorize", lambda: vectorize(graph, dist_map, thick_mask, p))
    segments = _t("dashed", lambda: dashed.detect(
        segments,
        dash_max_len=p.dash_max_len,
        dash_max_gap=p.dash_max_gap,
        dash_min_count=p.dash_min_count,
    ))
    segments, junctions = _t("regularize", lambda: regularize(segments, p))
    if reclaimed_dashed:
        existing = {(round(s.p0[0]), round(s.p0[1])) for s in segments if s.dashed}
        for ds in reclaimed_dashed:
            if (round(ds.p0[0]), round(ds.p0[1])) not in existing:
                segments.append(ds)
    staircases = _t("staircases", lambda: detect_staircase_regions(segments, p))
    symbols_found = _t("symbols", lambda: recognize_symbols(segments, arcs, p))
    remainder = _t("remainder", lambda: extract_remainder(ink, segments, arcs, p))
    ocr_boxes = ocr_text_boxes(gray, text_mask) if p.run_ocr else []

    return PipelineResult(
        params=p, gray=gray, ink=ink, text_mask=text_mask, graphics_mask=graphics_mask,
        thick_mask=thick_mask, thin_mask=thin_mask, skeleton=skeleton, dist_map=dist_map,
        graph=graph, polylines=polylines, segments=segments, arcs=arcs, junctions=junctions,
        remainder=remainder, ocr_boxes=ocr_boxes, timings=t,
        staircases=staircases, symbols=symbols_found,
    )
