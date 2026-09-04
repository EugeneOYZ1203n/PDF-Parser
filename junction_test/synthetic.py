"""Procedural architectural-ish drawings with exact ground truth.

Simpler than junction_cnn/synthetic.py; reuses its idea of an angle hierarchy
favouring 90 / 45 degrees. Returns (uint8 raster, GroundTruth).
"""
from __future__ import annotations

import numpy as np

from .geom import dedup_points, dist, heading_deg, point_to_segment_dist, segments_intersection
from .types_ import Arc, GroundTruth, Junction, Segment, StaircaseRegion, SymbolInstance

_ANGLES = [0.0, 90.0, 45.0, 135.0, 22.5, 67.5]


def _line(img, p0, p1, thickness):
    import cv2
    cv2.line(img, (int(round(p0[0])), int(round(p0[1]))),
             (int(round(p1[0])), int(round(p1[1]))), 40, thickness, cv2.LINE_AA)


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
             rotate: bool = False) -> tuple[np.ndarray, GroundTruth]:
    import cv2
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 255, np.uint8)

    wall_t = int(rng.integers(5, 9))
    thin_t = int(rng.integers(1, 3))
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

    # dashed lines
    for _ in range(int(rng.integers(1, 3))):
        horiz = rng.random() < 0.5
        if horiz:
            yv = int(rng.integers(y0 + 30, y1 - 30))
            a, b = (x0 + 15, yv), (x1 - 15, yv)
        else:
            xv = int(rng.integers(x0 + 30, x1 - 30))
            a, b = (xv, y0 + 15), (xv, y1 - 15)
        _dashed(img, a, b, thin_t)
        gt_segments.append(Segment(p0=(float(a[0]), float(a[1])),
                                   p1=(float(b[0]), float(b[1])),
                                   width=float(thin_t), dashed=True))

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

    gt_junctions = _ground_truth_junctions(gt_segments, size)
    return img, GroundTruth(segments=gt_segments, arcs=gt_arcs,
                            junctions=gt_junctions, size=(size, size),
                            staircases=gt_staircases, symbols=gt_symbols)


def _rot_pt(p, m):
    return (float(m[0, 0] * p[0] + m[0, 1] * p[1] + m[0, 2]),
            float(m[1, 0] * p[0] + m[1, 1] * p[1] + m[1, 2]))


def _rot_seg(s: Segment, m) -> Segment:
    return Segment(_rot_pt(s.p0, m), _rot_pt(s.p1, m), s.width, s.thick, s.dashed)


def _rot_arc(a: Arc, m) -> Arc:
    return Arc(_rot_pt(a.center, m), a.radius, a.a0, a.a1,
              [_rot_pt(p, m) for p in a.polyline], a.closed, a.width)


def _rot_bbox(bbox, m):
    x0, y0, x1, y1 = bbox
    pts = [_rot_pt(p, m) for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _ground_truth_junctions(segs: list[Segment], size: int) -> list[Junction]:
    from .geom import classify_junction, heading_deg, point_to_segment_dist

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
        out.append(Junction(
            xy=p, directions=list(arms), arm_angles=sorted(arms),
            jtype=classify_junction(arms), members=members,
        ))
    return out
