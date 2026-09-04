"""Staircase detection (Dosch 2000 Sec 3.2).

The paper delegates to G. Sanchez's structural polygon-texture segmentation
(external, unavailable here) plus a crude filter keeping regions with 5..30
texture elements. `junction_test` has no polygon-extraction stage to feed a
real texture segmenter, so this reimplements the idea directly on vectorized
segments: a stair tread run is structurally a dashed line rotated 90 degrees
-- short segments regularly spaced ALONG a walking axis, each oriented
PERPENDICULAR to that axis, rather than dashes lying end-to-end ALONG their
own line. Reuses `dashed._find_regular_runs` with each candidate's heading
rotated +90 deg so the alignment test checks "across the line" instead of
"with the line".
"""
from __future__ import annotations

import cv2
import numpy as np

from .dashed import _find_regular_runs
from .geom import dist
from .types_ import Segment, StaircaseRegion


def detect(
    segments: list[Segment],
    *,
    tread_min_len: float = 10.0,
    tread_max_len: float = 90.0,
    angle_tol_deg: float = 10.0,
    len_ratio_tol: float = 0.35,
    spacing_min: float = 6.0,
    spacing_max: float = 40.0,
    lateral_tol: float = 6.0,
    min_treads: int = 5,
    max_treads: int = 30,
    max_candidates: int = 400,
) -> list[StaircaseRegion]:
    segs = list(segments)
    lengths = [dist(s.p0, s.p1) for s in segs]
    mids = [((s.p0[0] + s.p1[0]) / 2, (s.p0[1] + s.p1[1]) / 2) for s in segs]
    # rotate each candidate's heading 90 deg: a tread run's own segments are
    # PERPENDICULAR to the walking axis, so the axis-alignment test in
    # _find_regular_runs (which normally checks "is this dash aligned WITH
    # the line") must check "is this tread aligned ACROSS the line" instead.
    from .geom import heading_deg
    axis_headings = [(heading_deg(s.p0, s.p1) + 90.0) % 180.0 for s in segs]
    is_key = [not s.dashed and tread_min_len <= lengths[i] <= tread_max_len
              for i, s in enumerate(segs)]

    runs = _find_regular_runs(
        mids, axis_headings, lengths, is_key,
        max_len=tread_max_len, max_gap=spacing_max, min_count=min_treads,
        max_count=max_treads, angle_tol_deg=angle_tol_deg, lateral_tol=lateral_tol,
        require_key_fraction=0.0, max_candidates=max_candidates,
    )

    regions: list[StaircaseRegion] = []
    for run in runs:
        run_lengths = [lengths[k] for k in run]
        if (max(run_lengths) / max(min(run_lengths), 1e-6)) - 1.0 > len_ratio_tol:
            continue
        pts = np.array([mids[k] for k in run], float)
        c = pts.mean(0)
        axis = np.linalg.svd(pts - c, full_matrices=False)[2][0]
        t = (pts - c) @ axis
        p_lo = tuple((c + axis * t.min()).tolist())
        p_hi = tuple((c + axis * t.max()).tolist())
        spacing = float(np.mean(np.diff(np.sort(t)))) if len(run) > 1 else 0.0
        if not (spacing_min <= spacing <= spacing_max):
            continue
        hull_pts = np.array([ep for k in run for ep in (segs[k].p0, segs[k].p1)], float)
        hull = cv2.convexHull(hull_pts.astype(np.float32)).reshape(-1, 2)
        regions.append(StaircaseRegion(
            polygon=[tuple(map(float, pt)) for pt in hull],
            treads=[segs[k] for k in run],
            axis=(p_lo, p_hi),
            spacing=spacing,
            n_treads=len(run),
        ))
    return regions
