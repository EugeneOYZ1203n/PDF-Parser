"""Procedural architectural-ish drawings with exact ground truth.

Simpler than junction_cnn/synthetic.py; reuses its idea of an angle hierarchy
favouring 90 / 45 degrees. Returns (uint8 raster, GroundTruth).
"""
from __future__ import annotations

import numpy as np

from .geom import dedup_points, segments_intersection
from .types_ import Arc, GroundTruth, Junction, Segment

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
    for _ in range(int(rng.integers(2, 4))):
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

    # door arcs (quarter circles)
    for _ in range(int(rng.integers(1, 4))):
        r = int(rng.integers(28, 52))
        cx = int(rng.integers(x0 + r + 10, x1 - r - 10))
        cy = int(rng.integers(y0 + r + 10, y1 - r - 10))
        a0 = float(rng.choice([0, 90, 180, 270]))
        a1 = a0 + 90.0
        _arc(img, (cx, cy), r, a0, a1, thin_t)
        t = np.radians(np.linspace(a0, a1, 24))
        poly = [(float(cx + r * np.cos(tt)), float(cy + r * np.sin(tt))) for tt in t]
        gt_arcs.append(Arc(center=(float(cx), float(cy)), radius=float(r),
                           a0=a0, a1=a1, polyline=poly, width=float(thin_t)))

    if rotate:
        ang = float(rng.uniform(-8, 8))
        m2 = cv2.getRotationMatrix2D((size / 2, size / 2), ang, 1.0)
        img = cv2.warpAffine(img, m2, (size, size), borderValue=255)
        gt_segments = [_rot_seg(s, m2) for s in gt_segments]
        gt_arcs = [_rot_arc(a, m2) for a in gt_arcs]

    if noise > 0:
        img = np.clip(img.astype(float) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)

    gt_junctions = _ground_truth_junctions(gt_segments, size)
    return img, GroundTruth(segments=gt_segments, arcs=gt_arcs,
                            junctions=gt_junctions, size=(size, size))


def _rot_pt(p, m):
    return (float(m[0, 0] * p[0] + m[0, 1] * p[1] + m[0, 2]),
            float(m[1, 0] * p[0] + m[1, 1] * p[1] + m[1, 2]))


def _rot_seg(s: Segment, m) -> Segment:
    return Segment(_rot_pt(s.p0, m), _rot_pt(s.p1, m), s.width, s.thick, s.dashed)


def _rot_arc(a: Arc, m) -> Arc:
    return Arc(_rot_pt(a.center, m), a.radius, a.a0, a.a1,
              [_rot_pt(p, m) for p in a.polyline], a.closed, a.width)


def _ground_truth_junctions(segs: list[Segment], size: int) -> list[Junction]:
    from .geom import point_to_segment_dist

    walls = [s for s in segs if not s.dashed]
    cand = []
    for s in walls:
        cand.extend([s.p0, s.p1])
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            x = segments_intersection(walls[i].p0, walls[i].p1, walls[j].p0, walls[j].p1)
            if x is not None:
                cand.append(x)
    cand = [p for p in cand if 0 <= p[0] < size and 0 <= p[1] < size]
    merged = dedup_points(cand, 5.0)
    # keep a point only if >=2 wall segments actually meet / cross there
    out = []
    for p in merged:
        touching = sum(1 for s in walls if point_to_segment_dist(p, s.p0, s.p1) <= 4.0)
        if touching >= 2:
            out.append(Junction(xy=p, directions=[]))
    return out
