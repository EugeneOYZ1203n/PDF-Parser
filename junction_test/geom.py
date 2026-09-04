"""Small geometry helpers shared across the spike modules."""
from __future__ import annotations

import numpy as np

Point = tuple[float, float]


def dist(a: Point, b: Point) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def seg_length(p0: Point, p1: Point) -> float:
    return dist(p0, p1)


def seg_midpoint(p0: Point, p1: Point) -> Point:
    return ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)


def angle_deg(p0: Point, p1: Point) -> float:
    """Direction of p0->p1 in degrees, 0..180 (undirected line orientation)."""
    a = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    return float(a % 180.0)


def heading_deg(p0: Point, p1: Point) -> float:
    """Directed heading p0->p1 in degrees, 0..360."""
    return float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])) % 360.0)


def angle_gap(a: float, b: float, period: float = 180.0) -> float:
    d = abs(a - b) % period
    return min(d, period - d)


def point_to_segment_dist(p: Point, a: Point, b: Point) -> float:
    ap = np.array(p, float) - np.array(a, float)
    ab = np.array(b, float) - np.array(a, float)
    denom = float(ab @ ab)
    if denom < 1e-9:
        return dist(p, a)
    t = float(np.clip((ap @ ab) / denom, 0.0, 1.0))
    proj = np.array(a, float) + t * ab
    return float(np.hypot(*(np.array(p, float) - proj)))


def polyline_length(pts: np.ndarray) -> float:
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return 0.0
    return float(np.hypot(*np.diff(pts, axis=0).T).sum())


def segments_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> Point | None:
    """Proper intersection point of segments (p1,p2) and (p3,p4), or None."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom
    if -0.02 <= t <= 1.02 and -0.02 <= u <= 1.02:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


def dedup_points(points: list[Point], tol: float) -> list[Point]:
    """Merge points within `tol` (greedy, average of a cluster)."""
    out: list[Point] = []
    used = [False] * len(points)
    for i, p in enumerate(points):
        if used[i]:
            continue
        cluster = [p]
        used[i] = True
        for j in range(i + 1, len(points)):
            if not used[j] and dist(p, points[j]) <= tol:
                cluster.append(points[j])
                used[j] = True
        cx = float(np.mean([c[0] for c in cluster]))
        cy = float(np.mean([c[1] for c in cluster]))
        out.append((cx, cy))
    return out


def classify_junction(arm_angles: list[float], collinear_deg: float = 20.0) -> str:
    """Junction type from the set of arm headings (degrees, 0..360, one per arm
    pointing away from the node). L|T|X|Y|star|endpoint (SYNTHETIC_DATA.md)."""
    a = sorted(float(x) % 360.0 for x in arm_angles)
    n = len(a)
    if n <= 1:
        return "endpoint"
    if n == 2:
        return "L"

    def _has_opposite(i):
        return any(i != j and abs(angle_gap(a[i], a[j], 360.0) - 180.0) <= collinear_deg
                   for j in range(n))

    opp = sum(1 for i in range(n) if _has_opposite(i))
    if n == 3:
        return "T" if opp >= 2 else "Y"
    if n == 4:
        return "X" if opp == 4 else "star"
    return "star"


def taubin_circle_fit(pts: np.ndarray) -> tuple[float, float, float, float]:
    """Algebraic circle fit (Taubin). Returns (cx, cy, r, rms_radial_residual)."""
    pts = np.asarray(pts, float)
    x = pts[:, 0]
    y = pts[:, 1]
    n = len(pts)
    if n < 3:
        return (0.0, 0.0, 0.0, 1e9)
    x_m = x.mean()
    y_m = y.mean()
    u = x - x_m
    v = y - y_m
    z = u * u + v * v
    z_m = z.mean()
    zc = z - z_m
    m = np.column_stack([zc, u, v])
    # normalise
    _, s, vt = np.linalg.svd(m, full_matrices=False)
    a = vt[-1]  # [A, B, C] for A*z + B*u + C*v + D = 0, with 2*A*z_m + ... layout
    a0, b0, c0 = a
    if abs(a0) < 1e-12:
        return (0.0, 0.0, 0.0, 1e9)
    # centre in (u,v) space: (-B/(2A), -C/(2A)); D = -A*z_m - B*u_m - C*v_m, u_m=v_m=0
    uc = -b0 / (2 * a0)
    vc = -c0 / (2 * a0)
    d0 = -a0 * z_m
    r2 = uc * uc + vc * vc - d0 / a0
    if r2 <= 0:
        return (0.0, 0.0, 0.0, 1e9)
    r = float(np.sqrt(r2))
    cx = uc + x_m
    cy = vc + y_m
    radial = np.hypot(x - cx, y - cy) - r
    rms = float(np.sqrt(np.mean(radial ** 2)))
    return (float(cx), float(cy), r, rms)
