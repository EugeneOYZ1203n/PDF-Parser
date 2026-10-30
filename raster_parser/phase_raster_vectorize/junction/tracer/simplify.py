"""Douglas-Peucker polyline fit -- every traced stroke and fill outline goes
through `fit_polyline` (`pipeline.vectorize`, `fills.outlines`); the debug
chain display (`adapter._accumulate_trace_geometry`) thins raw chains with
`approximate_rdp` directly.

Both chain ends are always kept, so a chain's junction/end nodes never move
-- the same contract as `Evaluation/CadFont/graph.py::_rdp_chain` (not
imported: P2 backends don't depend on Evaluation tooling). A closed loop
(first point == last point) works too: the zero-length chord makes the first
split the farthest point from the start. No endpoint snapping, no
short-segment drop and no collinear merge: those damaged real geometry
before (see git history, commit fcd677c)."""

from __future__ import annotations

import numpy as np

from raster_parser.models.junction import Point


def _max_dev(pts: np.ndarray, i: int, j: int) -> tuple[float, int]:
    """Max perpendicular distance of pts[i..j] from the chord pts[i]-pts[j]."""
    a, b = pts[i], pts[j]
    ab = b - a
    n = np.hypot(*ab)
    if n < 1e-9:
        d = np.hypot(*(pts[i : j + 1] - a).T)
    else:
        rel = pts[i : j + 1] - a
        d = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / n
    k = int(np.argmax(d))
    return float(d[k]), i + k


def approximate_rdp(chain: list[Point], eps: float) -> list[Point]:
    """Iterative Douglas-Peucker with tolerance `eps` (px)."""
    pts = np.asarray(chain, float)
    if len(pts) < 3:
        return [tuple(p) for p in pts]
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        d, k = _max_dev(pts, i, j)
        if d > eps:
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return [(float(p[0]), float(p[1])) for p in pts[keep]]


def anchor_loop(chain: list[Point]) -> list[Point]:
    """A closed chain (first == last) starts wherever the skeleton walk
    happened to begin, and Douglas-Peucker always keeps that start -- an
    arbitrary vertex that leaves a slanted stub near a corner. Re-start the
    loop at its point farthest from the loop's centroid (a genuine extreme,
    e.g. a rectangle corner) instead."""
    if len(chain) < 4 or chain[0] != chain[-1]:
        return chain
    pts = np.asarray(chain[:-1], float)
    k = int(np.argmax(np.hypot(*(pts - pts.mean(axis=0)).T)))
    ring = list(chain[:-1])
    rotated = ring[k:] + ring[:k]
    return rotated + [rotated[0]]


def fit_polyline(chain: list[Point], eps: float) -> list[Point]:
    """`anchor_loop` + `approximate_rdp`; `eps <= 0` returns the chain as is."""
    if eps <= 0:
        return [(float(p[0]), float(p[1])) for p in chain]
    return approximate_rdp(anchor_loop(chain), eps)
