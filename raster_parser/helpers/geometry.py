from __future__ import annotations

import math
from typing import Any
from raster_parser.models import Vector

_JOIN_EPS = 1e-3


def _pt(p: Any) -> tuple[float, float]:
    if hasattr(p, "x"):
        return (float(p.x), float(p.y))
    if isinstance(p, (list, tuple)) and len(p) >= 2:
        return (float(p[0]), float(p[1]))
    raise TypeError(f"not a point: {p!r}")


def _item_points(item: Any) -> list[tuple[float, float]]:
    """Sample points along one drawing op (chord approx for beziers)."""
    op = item[0]
    if op == "l":
        return [_pt(item[1]), _pt(item[2])]
    if op == "c":
        p0, p1, p2, p3 = _pt(item[1]), _pt(item[2]), _pt(item[3]), _pt(item[4])
        return [p0, p1, p2, p3]
    if op == "re":
        r = item[1]
        if hasattr(r, "x0"):
            x0, y0, x1, y1 = float(r.x0), float(r.y0), float(r.x1), float(r.y1)
        else:
            x0, y0, x1, y1 = map(float, r[:4])
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    if op == "qu":
        q = item[1]
        pts = [_pt(q[i]) for i in range(4)]
        pts.append(pts[0])
        return pts
    return []


def _split_loops(items: list) -> list[list[tuple[float, float]]]:
    """Split drawing items into continuous closed/open polylines."""
    if not items:
        return []
    loops: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []

    def flush() -> None:
        nonlocal cur
        if len(cur) >= 2:
            loops.append(cur)
        cur = []

    for it in items:
        op = it[0]
        if op in ("re", "qu"):
            flush()
            pts = _item_points(it)
            if pts:
                loops.append(pts)
            continue
        pts = _item_points(it)
        if not pts:
            flush()
            continue
        if not cur:
            cur = pts[:]
            continue
        prev_end = cur[-1]
        start = pts[0]
        dx = start[0] - prev_end[0]
        dy = start[1] - prev_end[1]
        if dx * dx + dy * dy > _JOIN_EPS * _JOIN_EPS:
            flush()
            cur = pts[:]
        else:
            cur.extend(pts[1:])
    flush()
    return loops


def _perimeter(pts: list[tuple[float, float]]) -> float:
    if len(pts) < 2:
        return 0.0
    p = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        p += math.hypot(dx, dy)
    dx = pts[0][0] - pts[-1][0]
    dy = pts[0][1] - pts[-1][1]
    if dx * dx + dy * dy > _JOIN_EPS * _JOIN_EPS:
        p += math.hypot(dx, dy)
    return p


def _area(pts: list[tuple[float, float]]) -> float:
    """Shoelace area (absolute); closes the ring if needed."""
    if len(pts) < 3:
        return 0.0
    ring = pts
    if abs(pts[0][0] - pts[-1][0]) > _JOIN_EPS or abs(pts[0][1] - pts[-1][1]) > _JOIN_EPS:
        ring = pts + [pts[0]]
    a = 0.0
    for i in range(len(ring) - 1):
        x0, y0 = ring[i]
        x1, y1 = ring[i + 1]
        a += x0 * y1 - x1 * y0
    return abs(a) * 0.5


def isoperimetric_quotient(v: Vector) -> float:
    """``Q = 4*pi*A / P^2`` for ``v`` (area-weighted mean over continuous loops).

    Returns 0 if geometry is degenerate. Circle -> 1; thin stroke ribbon -> near 0.
    """
    loops = _split_loops(v.items or [])
    if not loops:
        x0, y0, x1, y1 = v.rect
        w = max(0.0, float(x1) - float(x0))
        h = max(0.0, float(y1) - float(y0))
        a = w * h
        p = 2.0 * (w + h)
        if p <= 0 or a <= 0:
            return 0.0
        return float(4.0 * math.pi * a / (p * p))

    total_a = 0.0
    weighted_q = 0.0
    for pts in loops:
        a = _area(pts)
        p = _perimeter(pts)
        if a <= 0 or p <= 0:
            continue
        q = 4.0 * math.pi * a / (p * p)
        weighted_q += a * q
        total_a += a
    if total_a <= 0:
        return 0.0
    return float(weighted_q / total_a)
