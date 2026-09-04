"""Procedural architectural-ish drawings with exact ground truth.

Simpler than junction_cnn/synthetic.py; reuses its idea of an angle hierarchy
favouring 90 / 45 degrees. Returns (uint8 raster, GroundTruth).
"""
from __future__ import annotations

import numpy as np

from .geom import dedup_points, dist, heading_deg, point_to_segment_dist, segments_intersection
from .types_ import Arc, GroundTruth, Junction, Segment, StaircaseRegion, SymbolInstance

_ANGLES = [0.0, 90.0, 45.0, 135.0, 22.5, 67.5]

# AEC line-weight ladder (mm) -> px at a given DPI (SYNTHETIC_DATA.md).
_WEIGHTS_MM = (0.13, 0.18, 0.25, 0.35, 0.50, 0.70, 1.00)

# dash_style -> on/off run pattern (px, before DPI scaling); solid handled separately.
_DASH_PATTERNS = {
    "dashed": (10.0, 8.0),
    "hidden": (6.0, 4.0),
    "center": (18.0, 4.0, 3.0, 4.0),           # dash-dot
    "phantom": (24.0, 4.0, 3.0, 4.0, 3.0, 4.0),  # dash-dot-dot
}

# named colour layers (RGB). near-black default plus MEP/annotation hues.
_LAYERS = {
    "ink": (40, 40, 40),
    "mep_supply": (30, 90, 200),
    "mep_return": (200, 40, 40),
    "site": (30, 150, 60),
    "annotation": (170, 40, 170),
}


def mm_to_px(mm: float, dpi: int) -> float:
    return mm / 25.4 * dpi


def _gray_of(rgb: tuple[int, int, int]) -> int:
    """ITU-R 601 luma; the spike raster is single-channel so coloured ink is
    flattened here while the true RGB is kept in ground truth."""
    r, g, b = rgb
    return int(round(0.299 * r + 0.587 * g + 0.114 * b))


def _weights_px(rng, dpi: int) -> tuple[int, int, int]:
    """heavy / medium / light stroke widths, px, three distinct ladder rungs."""
    idx = sorted(rng.choice(len(_WEIGHTS_MM), size=3, replace=False).tolist())
    w = [max(1, int(round(mm_to_px(_WEIGHTS_MM[i], dpi)))) for i in idx]
    if w[0] == w[1]:
        w[1] += 1
    if w[1] >= w[2]:
        w[2] = w[1] + 1
    return w[2], w[1], w[0]


def _line(img, p0, p1, thickness, val: int = 40):
    import cv2
    cv2.line(img, (int(round(p0[0])), int(round(p0[1]))),
             (int(round(p1[0])), int(round(p1[1]))), val, thickness, cv2.LINE_AA)


def _styled_dash(img, p0, p1, thickness, pattern, val: int = 40):
    """Render an arbitrary on/off dash pattern along p0->p1."""
    p0 = np.array(p0, float)
    p1 = np.array(p1, float)
    total = float(np.hypot(*(p1 - p0)))
    if total < 1e-6:
        return
    d = (p1 - p0) / total
    s, k, on = 0.0, 0, True
    while s < total:
        run = pattern[k % len(pattern)]
        e = min(s + run, total)
        if on:
            _line(img, p0 + d * s, p0 + d * e, thickness, val)
        s = e
        k += 1
        on = not on


def _punch_holes(img, rng, segs, level: float):
    """Simulate imperfect text removal: white axis-aligned rectangles through
    strokes (bridge targets) + stray short strokes not in GT (reject targets)."""
    if level <= 0:
        return
    import cv2
    h, w = img.shape
    n_holes = int(level * 8)
    for _ in range(n_holes):
        if not segs:
            break
        s = segs[int(rng.integers(len(segs)))]
        t = rng.random()
        cx = s.p0[0] + t * (s.p1[0] - s.p0[0])
        cy = s.p0[1] + t * (s.p1[1] - s.p0[1])
        bw = int(rng.integers(8, 22))
        bh = int(rng.integers(8, 16))
        x = int(np.clip(cx - bw / 2, 0, w - 1))
        y = int(np.clip(cy - bh / 2, 0, h - 1))
        img[y:y + bh, x:x + bw] = 255
    for _ in range(int(level * 6)):                 # glyph-fragment speckle
        gx = int(rng.integers(0, w - 12))
        gy = int(rng.integers(0, h - 12))
        a = (gx, gy)
        b = (gx + int(rng.integers(-6, 7)), gy + int(rng.integers(-6, 7)))
        _line(img, a, b, 1)


def _dashed(img, p0, p1, thickness, dash=10, gap=8):
    p0 = np.array(p0, float)
    p1 = np.array(p1, float)
    total = np.hypot(*(p1 - p0))
    d = (p1 - p0) / total
    s = 0.0
    while s < total:
        a = p0 + d * s
        b = p0 + d * min(s + dash, total)
        _line(img, a, b, thickness)
        s += dash + gap


def _arc(img, center, radius, a0, a1, thickness):
    import cv2
    cv2.ellipse(img, (int(center[0]), int(center[1])), (int(radius), int(radius)),
                0, a0, a1, 40, thickness, cv2.LINE_AA)


def _staircase(img, origin, direction_deg, n_treads, tread_len, spacing, thickness):
    """Draw n_treads short ticks perpendicular to direction_deg, evenly spaced
    along direction_deg starting at origin. Returns list of tread endpoint
    (p0, p1) pairs."""
    d = np.radians(direction_deg)
    axis = np.array([np.cos(d), np.sin(d)])
    perp = np.array([-axis[1], axis[0]])
    treads = []
    for i in range(n_treads):
        c = np.array(origin, float) + axis * (i * spacing)
        a = c - perp * tread_len / 2
        b = c + perp * tread_len / 2
        _line(img, a, b, thickness)
        treads.append(((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
    return treads


def _door_leaf(img, hinge, heading_deg_, length, thickness):
    """Companion straight leaf segment for a door swing arc -- shares the
    hinge point with the arc's start."""
    h = np.radians(heading_deg_)
    far = (hinge[0] + length * np.cos(h), hinge[1] + length * np.sin(h))
    _line(img, hinge, far, thickness)
    return far


def _window(img, wall_p0, wall_p1, t_along, gap, jamb_len, thin_t, wall_t):
    """Split a wall segment with a gap and draw two perpendicular jamb ticks
    at the gap edges. Returns (wall_piece_a, wall_piece_b, jamb1, jamb2), each
    a (p0, p1) tuple."""
    p0, p1 = np.array(wall_p0, float), np.array(wall_p1, float)
    length = np.hypot(*(p1 - p0))
    axis = (p1 - p0) / (length + 1e-9)
    perp = np.array([-axis[1], axis[0]])
    center = p0 + axis * t_along
    ga, gb = center - axis * gap / 2, center + axis * gap / 2
    _line(img, p0, ga, wall_t)
    _line(img, gb, p1, wall_t)
    j1a, j1b = ga - perp * jamb_len / 2, ga + perp * jamb_len / 2
    j2a, j2b = gb - perp * jamb_len / 2, gb + perp * jamb_len / 2
    _line(img, j1a, j1b, thin_t)
    _line(img, j2a, j2b, thin_t)
    to_pt = lambda v: (float(v[0]), float(v[1]))
    return ((to_pt(p0), to_pt(ga)), (to_pt(gb), to_pt(p1)),
            (to_pt(j1a), to_pt(j1b)), (to_pt(j2a), to_pt(j2b)))


def generate(seed: int = 0, size: int = 512, noise: float = 3.0,
             rotate: bool = False, *,
             dpi: int = 150,
             weight_ladder: bool = False,
             color_layers: bool = False,
             dash_styles: bool = False,
             forced_crossings: int = 0,
             coincident_unrelated: int = 0,
             curved_walls: bool = False,
             residue_level: float = 0.0,
             archetype: str = "floor_plan") -> tuple[np.ndarray, GroundTruth]:
    """Procedural AEC drawing + exact ground truth.

    The keyword-only knobs add the SYNTHETIC_DATA.md difficulty axes; all default
    to off/neutral so existing callers (smoke.py) get the original archetype #1.
    """
    import cv2
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 255, np.uint8)

    if weight_ladder:
        wall_t, mid_t, thin_t = _weights_px(rng, dpi)
    else:
        wall_t = int(rng.integers(5, 9))
        thin_t = int(rng.integers(1, 3))
        mid_t = max(thin_t + 1, wall_t - 2)
    m = int(size * 0.08)

    gt_segments: list[Segment] = []
    gt_arcs: list[Arc] = []

    # outer rectangle (thick walls)
    x0, y0, x1, y1 = m, m, size - m, size - m
    rect = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
            ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
    for a, b in rect:
        _line(img, a, b, wall_t)
        gt_segments.append(Segment(p0=a, p1=b, width=float(wall_t), thick=True))

    # interior walls (thick), axis-aligned, spanning to the outer walls
    gt_staircases: list[StaircaseRegion] = []
    gt_symbols: list[SymbolInstance] = []
    n_interior_walls = int(rng.integers(2, 4))
    window_wall_idx = 0 if (seed % 2 == 0) else None  # force a window on even seeds
    for wall_i in range(n_interior_walls):
        if rng.random() < 0.5:
            xv = int(rng.integers(x0 + 40, x1 - 40))
            ya = y0
            yb = int(rng.integers(y0 + 40, y1 - 20))
            a, b = (xv, ya), (xv, yb)
        else:
            yv = int(rng.integers(y0 + 40, y1 - 40))
            xa = int(rng.integers(x0 + 20, x1 - 40))
            xb = x1
            a, b = (xa, yv), (xb, yv)
        wall_len = dist(a, b)
        if wall_i == window_wall_idx and wall_len > 80:
            gap = float(rng.uniform(22.0, 34.0))
            jamb_len = float(wall_t + 6)
            t_along = wall_len / 2.0
            (wp0, wp1), (wp2, wp3), (j1a, j1b), (j2a, j2b) = _window(
                img, a, b, t_along, gap, jamb_len, thin_t, wall_t)
            gt_segments.append(Segment(p0=wp0, p1=wp1, width=float(wall_t), thick=True))
            gt_segments.append(Segment(p0=wp2, p1=wp3, width=float(wall_t), thick=True))
            jamb1 = Segment(p0=j1a, p1=j1b, width=float(thin_t))
            jamb2 = Segment(p0=j2a, p1=j2b, width=float(thin_t))
            gt_segments.append(jamb1)
            gt_segments.append(jamb2)
            cx = (j1a[0] + j1b[0] + j2a[0] + j2b[0]) / 4.0
            cy = (j1a[1] + j1b[1] + j2a[1] + j2b[1]) / 4.0
            xs = [j1a[0], j1b[0], j2a[0], j2b[0]]
            ys = [j1a[1], j1b[1], j2a[1], j2b[1]]
            gt_symbols.append(SymbolInstance(
                family="window", features=[jamb1, jamb2], anchor=(cx, cy),
                bbox=(min(xs), min(ys), max(xs), max(ys)), error=0.0))
        else:
            _line(img, a, b, wall_t)
            gt_segments.append(Segment(p0=a, p1=b, width=float(wall_t), thick=True))

    # thin fixture lines (some diagonal)
    for _ in range(int(rng.integers(3, 7))):
        ang = np.radians(rng.choice(_ANGLES) + rng.normal(0, 3))
        length = rng.integers(30, 110)
        cx = rng.integers(x0 + 20, x1 - 20)
        cy = rng.integers(y0 + 20, y1 - 20)
        a = (cx, cy)
        b = (cx + length * np.cos(ang), cy + length * np.sin(ang))
        _line(img, a, b, thin_t)
        gt_segments.append(Segment(p0=a, p1=(float(b[0]), float(b[1])), width=float(thin_t)))

    # dashed lines (optionally with the full hidden/center/phantom vocabulary
    # and coloured MEP layers)
    style_pool = list(_DASH_PATTERNS) if dash_styles else ["dashed"]
    for _ in range(int(rng.integers(1, 3)) + (2 if dash_styles else 0)):
        horiz = rng.random() < 0.5
        if horiz:
            yv = int(rng.integers(y0 + 30, y1 - 30))
            a, b = (x0 + 15, yv), (x1 - 15, yv)
        else:
            xv = int(rng.integers(x0 + 30, x1 - 30))
            a, b = (xv, y0 + 15), (xv, y1 - 15)
        style = str(rng.choice(style_pool))
        pattern = tuple(round(v * dpi / 150.0, 1) for v in _DASH_PATTERNS[style])
        layer = "ink"
        color = _LAYERS["ink"]
        if color_layers and rng.random() < 0.6:
            layer = str(rng.choice(["mep_supply", "mep_return", "site", "annotation"]))
            color = _LAYERS[layer]
        _styled_dash(img, a, b, thin_t, pattern, val=_gray_of(color))
        gt_segments.append(Segment(
            p0=(float(a[0]), float(a[1])), p1=(float(b[0]), float(b[1])),
            width=float(thin_t), dashed=True, dash_style=style,
            dash_array=pattern, color=color, layer=layer, role="hidden_edge"))

    # door arcs (quarter circles) + hinge leaf
    for _ in range(int(rng.integers(1, 4))):
        r = int(rng.integers(28, 52))
        cx = int(rng.integers(x0 + r + 10, x1 - r - 10))
        cy = int(rng.integers(y0 + r + 10, y1 - r - 10))
        a0 = float(rng.choice([0, 90, 180, 270]))
        a1 = a0 + 90.0
        _arc(img, (cx, cy), r, a0, a1, thin_t)
        t = np.radians(np.linspace(a0, a1, 24))
        poly = [(float(cx + r * np.cos(tt)), float(cy + r * np.sin(tt))) for tt in t]
        arc = Arc(center=(float(cx), float(cy)), radius=float(r),
                  a0=a0, a1=a1, polyline=poly, width=float(thin_t))
        gt_arcs.append(arc)

        # leaf: straight segment from the hinge (arc start point) radially
        # outward, length == radius -- shares the hinge point with the arc.
        hinge = poly[0]
        radial_heading = heading_deg((float(cx), float(cy)), hinge)
        far = _door_leaf(img, hinge, radial_heading, float(r), thin_t)
        leaf = Segment(p0=hinge, p1=(float(far[0]), float(far[1])), width=float(thin_t))
        gt_segments.append(leaf)
        xs = [hinge[0], far[0]] + [p[0] for p in poly]
        ys = [hinge[1], far[1]] + [p[1] for p in poly]
        gt_symbols.append(SymbolInstance(
            family="door", features=[leaf, arc], anchor=hinge,
            bbox=(min(xs), min(ys), max(xs), max(ys)), error=0.0))

    # staircase (forced on even seeds so smoke output has a non-flaky signal)
    if seed % 2 == 0:
        n_treads = int(rng.integers(6, 12))
        tread_len = float(rng.uniform(30.0, 50.0))
        spacing = float(rng.uniform(12.0, 20.0))
        direction = float(rng.choice([0.0, 90.0, 180.0, 270.0]))
        run_len = n_treads * spacing
        clearance = tread_len / 2 + 10.0
        origin = None
        for _ in range(20):
            cand = (float(rng.uniform(x0 + clearance, x1 - clearance - run_len)),
                    float(rng.uniform(y0 + clearance, y1 - clearance)))
            if all(point_to_segment_dist(cand, s.p0, s.p1) > clearance for s in gt_segments):
                origin = cand
                break
        if origin is not None:
            tread_pairs = _staircase(img, origin, direction, n_treads, tread_len, spacing, thin_t)
            treads = [Segment(p0=p0, p1=p1, width=float(thin_t)) for p0, p1 in tread_pairs]
            gt_segments.extend(treads)
            mids = np.array([((s.p0[0] + s.p1[0]) / 2, (s.p0[1] + s.p1[1]) / 2) for s in treads])
            c = mids.mean(0)
            axis_vec = np.linalg.svd(mids - c, full_matrices=False)[2][0]
            tproj = (mids - c) @ axis_vec
            axis_lo = tuple((c + axis_vec * tproj.min()).tolist())
            axis_hi = tuple((c + axis_vec * tproj.max()).tolist())
            hull_pts = np.array([ep for s in treads for ep in (s.p0, s.p1)], float)
            import cv2 as _cv2
            hull = _cv2.convexHull(hull_pts.astype(np.float32)).reshape(-1, 2)
            gt_staircases.append(StaircaseRegion(
                polygon=[tuple(map(float, pt)) for pt in hull],
                treads=treads, axis=(axis_lo, axis_hi),
                spacing=spacing, n_treads=n_treads,
            ))

    # ---- forced X crossings: two long strokes that both continue through ----
    cu_points: list[tuple[float, float]] = []
    for _ in range(forced_crossings):
        cx = float(rng.uniform(x0 + 60, x1 - 60))
        cy = float(rng.uniform(y0 + 60, y1 - 60))
        a1 = np.radians(float(rng.choice(_ANGLES)))
        a2 = a1 + np.radians(float(rng.uniform(50, 130)))
        L = float(rng.uniform(50, 110))
        for ang in (a1, a2):
            dvec = np.array([np.cos(ang), np.sin(ang)])
            p = (cx - dvec[0] * L, cy - dvec[1] * L)
            q = (cx + dvec[0] * L, cy + dvec[1] * L)
            _line(img, p, q, mid_t)
            gt_segments.append(Segment(p0=(float(p[0]), float(p[1])),
                                       p1=(float(q[0]), float(q[1])),
                                       width=float(mid_t), role="structure"))

    # ---- coincident-but-unrelated: an endpoint just grazes another stroke ----
    for _ in range(coincident_unrelated):
        host = gt_segments[int(rng.integers(len(gt_segments)))]
        t = float(rng.uniform(0.3, 0.7))
        hx = host.p0[0] + t * (host.p1[0] - host.p0[0])
        hy = host.p0[1] + t * (host.p1[1] - host.p0[1])
        ang = np.radians(float(rng.uniform(0, 360)))
        far = (hx + 40 * np.cos(ang), hy + 40 * np.sin(ang))
        _line(img, (hx, hy), far, thin_t)
        gt_segments.append(Segment(p0=(float(hx), float(hy)),
                                   p1=(float(far[0]), float(far[1])),
                                   width=float(thin_t), role="annotation"))
        cu_points.append((float(hx), float(hy)))

    # ---- large-radius curved wall, tangent to a straight wall stub ----
    if curved_walls:
        r = float(rng.uniform(size * 0.35, size * 0.6))
        cx = float(rng.uniform(x0, x1))
        cy = y1 + r * 0.4                       # centre below the sheet -> gentle arc across it
        a0 = float(np.degrees(np.arctan2(y0 - cy, x0 - cx)))
        a1d = float(np.degrees(np.arctan2(y0 - cy, x1 - cx)))
        lo, hi = sorted((a0, a1d))
        _arc(img, (cx, cy), r, lo, hi, wall_t)
        t = np.radians(np.linspace(lo, hi, 40))
        poly = [(float(cx + r * np.cos(tt)), float(cy + r * np.sin(tt))) for tt in t]
        gt_arcs.append(Arc(center=(cx, cy), radius=r, a0=lo, a1=hi, polyline=poly,
                           width=float(wall_t), role="curved_wall"))
        # tangent straight stub at the arc start
        p_start = np.array(poly[0])
        tang = p_start - np.array(poly[1])
        tang = tang / (np.hypot(*tang) + 1e-9)
        stub_end = p_start + tang * 60.0
        _line(img, tuple(p_start), tuple(stub_end), wall_t)
        gt_segments.append(Segment(p0=(float(p_start[0]), float(p_start[1])),
                                   p1=(float(stub_end[0]), float(stub_end[1])),
                                   width=float(wall_t), thick=True, role="wall"))

    _punch_holes(img, rng, gt_segments, residue_level)

    if rotate:
        ang = float(rng.uniform(-8, 8))
        m2 = cv2.getRotationMatrix2D((size / 2, size / 2), ang, 1.0)
        img = cv2.warpAffine(img, m2, (size, size), borderValue=255)
        gt_segments = [_rot_seg(s, m2) for s in gt_segments]
        gt_arcs = [_rot_arc(a, m2) for a in gt_arcs]
        gt_staircases = [
            StaircaseRegion(
                polygon=[_rot_pt(p, m2) for p in r.polygon],
                treads=[_rot_seg(s, m2) for s in r.treads],
                axis=(_rot_pt(r.axis[0], m2), _rot_pt(r.axis[1], m2)),
                spacing=r.spacing, n_treads=r.n_treads,
            )
            for r in gt_staircases
        ]
        gt_symbols = [
            SymbolInstance(
                family=sym.family,
                features=[_rot_seg(f, m2) if isinstance(f, Segment) else _rot_arc(f, m2) for f in sym.features],
                anchor=_rot_pt(sym.anchor, m2),
                bbox=_rot_bbox(sym.bbox, m2),
                error=sym.error,
            )
            for sym in gt_symbols
        ]

    if noise > 0:
        img = np.clip(img.astype(float) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)

    if rotate and cu_points:
        cu_points = [_rot_pt(p, m2) for p in cu_points]
    gt_junctions = _ground_truth_junctions(gt_segments, size, cu_points)
    meta = {
        "archetype": archetype, "dpi": dpi, "seed": seed,
        "weight_ladder": weight_ladder, "color_layers": color_layers,
        "dash_styles": dash_styles, "forced_crossings": forced_crossings,
        "coincident_unrelated": coincident_unrelated, "curved_walls": curved_walls,
        "residue_level": residue_level, "rotate": rotate,
    }
    return img, GroundTruth(segments=gt_segments, arcs=gt_arcs,
                            junctions=gt_junctions, size=(size, size),
                            staircases=gt_staircases, symbols=gt_symbols, meta=meta)


def _rot_pt(p, m):
    return (float(m[0, 0] * p[0] + m[0, 1] * p[1] + m[0, 2]),
            float(m[1, 0] * p[0] + m[1, 1] * p[1] + m[1, 2]))


def _rot_seg(s: Segment, m) -> Segment:
    return Segment(_rot_pt(s.p0, m), _rot_pt(s.p1, m), s.width, s.thick, s.dashed,
                   s.color, s.dash_style, s.dash_array, s.role, s.layer)


def _rot_arc(a: Arc, m) -> Arc:
    return Arc(_rot_pt(a.center, m), a.radius, a.a0, a.a1,
              [_rot_pt(p, m) for p in a.polyline], a.closed, a.width,
              a.color, a.dash_style, a.role, a.layer)


def _rot_bbox(bbox, m):
    x0, y0, x1, y1 = bbox
    pts = [_rot_pt(p, m) for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _ground_truth_junctions(segs: list[Segment], size: int,
                            cu_points: list[tuple[float, float]] | None = None) -> list[Junction]:
    from .geom import classify_junction, heading_deg, point_to_segment_dist

    cu_points = cu_points or []
    walls = [(i, s) for i, s in enumerate(segs) if not s.dashed]
    cand = []
    for _, s in walls:
        cand.extend([s.p0, s.p1])
    for a in range(len(walls)):
        for b in range(a + 1, len(walls)):
            x = segments_intersection(walls[a][1].p0, walls[a][1].p1,
                                      walls[b][1].p0, walls[b][1].p1)
            if x is not None:
                cand.append(x)
    cand = [p for p in cand if 0 <= p[0] < size and 0 <= p[1] < size]
    merged = dedup_points(cand, 5.0)

    out = []
    for p in merged:
        arms: list[float] = []
        members: list[int] = []
        for idx, s in walls:
            if point_to_segment_dist(p, s.p0, s.p1) > 4.0:
                continue
            near0 = dist(p, s.p0) <= 6.0
            near1 = dist(p, s.p1) <= 6.0
            if near0 and not near1:
                arms.append(heading_deg(p, s.p1))
            elif near1 and not near0:
                arms.append(heading_deg(p, s.p0))
            else:                                   # p sits in the interior -> passes through
                arms.append(heading_deg(p, s.p0))
                arms.append(heading_deg(p, s.p1))
            members.append(idx)
        if len(members) < 2 and len(arms) < 3:
            continue
        is_cu = any(dist(p, c) <= 6.0 for c in cu_points)
        out.append(Junction(
            xy=p, directions=list(arms), arm_angles=sorted(arms),
            jtype="coincident_unrelated" if is_cu else classify_junction(arms),
            members=members, is_true_connection=not is_cu,
        ))
    # CU points that produced no wall-wall candidate (endpoint grazing an interior)
    have = {(round(j.xy[0]), round(j.xy[1])) for j in out}
    for c in cu_points:
        if (round(c[0]), round(c[1])) not in have and all(dist(c, j.xy) > 6.0 for j in out):
            out.append(Junction(xy=c, directions=[], arm_angles=[],
                                jtype="coincident_unrelated", is_true_connection=False))
    return out
