"""Small geometry helpers for the tracer."""
from __future__ import annotations

import numpy as np

Point = tuple[float, float]


def dist(a: Point, b: Point) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def heading_deg(p0: Point, p1: Point) -> float:
    """Directed heading p0->p1 in degrees, 0..360."""
    return float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0])) % 360.0)


def angle_gap(a: float, b: float, period: float = 180.0) -> float:
    d = abs(a - b) % period
    return min(d, period - d)
