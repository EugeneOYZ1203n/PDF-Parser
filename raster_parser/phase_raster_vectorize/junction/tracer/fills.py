"""Solid filled regions -> outlines only.

Skeletonizing a solid blob (a filled arrowhead, poche wall, solid column,
filled dot) gives a medial-axis tree of branches that describe nothing
real. So solid regions are split off by *thickness* before skeletonizing and
only their boundary is kept:

  * core   = ink pixels deeper than r = min_width / 2 (distance transform
             > r). A stroke W px wide has DT ~ (W + 1) / 2 at its centre, so
             only ink at least `min_width` px wide survives -- every thinner
             line vanishes, including the crossing of two of them. Core
             components smaller than `min_width / 2` px (a blobby junction)
             are dropped.
  * fill   = core grown back by ceil(r * sqrt 2) + 1, clipped to the ink.
             That restores the convex corners a plain opening rounds off, and
             only enters attached lines by a few px.
  * outline = `cv2.findContours` (outer ring + one ring per hole), each ring
             Douglas-Peucker fitted.

`pipeline.run` skeletonizes `ink & ~dilate(fill)` -- everything else -- and
snaps free stroke ends within `snap_px` onto the fill, so a line attached to
a fill still touches its outline."""

from __future__ import annotations


import cv2
import numpy as np

from raster_parser.models.junction import FillSplit, Outline, Point
from .simplify import fit_polyline


# Fill cores smaller than this (px) are noise, whatever the fill width.
_MIN_CORE_AREA_PX = 3.0
# Free stroke ends within the fill's regrow distance plus this (px) snap onto the fill.
_SNAP_MARGIN_PX = 3


def split_fills(
    ink: np.ndarray, dt: np.ndarray, min_width_px: float, rdp_eps_px: float
) -> "FillSplit | None":
    """See module docstring. `None` when there are no solid regions."""
    r = min_width_px / 2.0
    core = (dt > r).astype(np.uint8)
    if not core.any():
        return None
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    small = stats[:, cv2.CC_STAT_AREA] < max(_MIN_CORE_AREA_PX, min_width_px / 2.0)
    small[0] = True
    core[small[lab]] = 0
    if not core.any():
        return None
    grow = int(np.ceil(r * np.sqrt(2.0))) + 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    fill = (cv2.dilate(core, k) > 0) & ink.astype(bool)
    return FillSplit(
        fill=fill, outlines=outlines(fill, rdp_eps_px), snap_px=float(grow + _SNAP_MARGIN_PX)
    )


def outlines(fill: np.ndarray, rdp_eps_px: float) -> list[Outline]:
    """One `Outline` per outer contour of `fill`, its holes as extra rings."""
    contours, hier = cv2.findContours(fill.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return []
    hier = hier[0]
    out: list[Outline] = []
    for i, c in enumerate(contours):
        if hier[i][3] != -1:  # a hole -- collected with its parent
            continue
        rings = [_ring(c, rdp_eps_px)]
        child = hier[i][2]
        while child != -1:
            rings.append(_ring(contours[child], rdp_eps_px))
            child = hier[child][0]
        rings = [rg for rg in rings if len(rg) >= 3]
        if rings:
            out.append(Outline(rings=rings))
    return out


def _ring(contour: np.ndarray, eps: float) -> list[Point]:
    pts = [(float(x), float(y)) for x, y in contour.reshape(-1, 2)]
    if len(pts) < 2:
        return pts
    return fit_polyline(pts + [pts[0]], eps)


def snap_free_ends(strokes: list[list[Point]], free_end, split: FillSplit) -> list[list[Point]]:
    """Extend every free stroke end (`free_end(point) -> bool`) lying within
    `split.snap_px` of the fill to its nearest fill pixel."""
    from scipy.ndimage import distance_transform_edt

    dist, (iy, ix) = distance_transform_edt(~split.fill, return_indices=True)
    h, w = dist.shape

    def target(p: Point) -> "Point | None":
        x = min(max(int(round(p[0])), 0), w - 1)
        y = min(max(int(round(p[1])), 0), h - 1)
        if 0 < dist[y, x] <= split.snap_px:
            return (float(ix[y, x]), float(iy[y, x]))
        return None

    out = []
    for s in strokes:
        if len(s) >= 2 and s[0] != s[-1]:
            s = list(s)
            if free_end(s[0]) and (t := target(s[0])) is not None:
                s.insert(0, t)
            if free_end(s[-1]) and (t := target(s[-1])) is not None:
                s.append(t)
        out.append(s)
    return out
