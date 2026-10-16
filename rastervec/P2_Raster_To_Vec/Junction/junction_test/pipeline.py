"""The centerline tracer Junction runs on each component crop:

    binarize (specks dropped, pinholes filled) -> distance transform
      -> solid fills split off, outlined only (`fills.py`)
      -> skeletonize the rest -> node/edge graph (`skeleton_graph.py`:
         junction-pixel clustering, barb pruning, junction-zone collapse)
      -> good continuation through junctions -> Douglas-Peucker per stroke

`run(gray, params) -> PipelineResult` keeps the intermediates for debugging.
Always native resolution (no downscale). Text is OCR'd and erased before
this runs (see `Junction/adapter.py`), so every ink pixel handed in is
traced -- there is no text/graphics separation here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize

from rastervec.commons.logging_setup import get_logger

from . import fills
from .simplify import fit_polyline
from .skeleton_graph import build_graph, continue_through_junctions
from .types_ import Graph, PipelineResult, Point, Polyline

_LOG = get_logger("P2.Junction.trace")


@dataclass
class Params:
    # binarisation: Otsu OR gray < soft_ink_thresh; ink blobs smaller than
    # min_ink_area_px (8-connected) are specks and dropped; enclosed paper
    # holes up to hole_max_area_px are filled (they skeletonize into bubble
    # loops: two junctions + two chains each)
    soft_ink_thresh: int = 245
    min_ink_area_px: int = 4
    hole_max_area_px: int = 6
    # skeleton graph: leaf branches off a junction shorter than this are barbs
    barb_min_px: float = 9.0
    # two junction nodes joined by an edge shorter than this x the local
    # stroke width merge (a thick X crossing skeletonizes as two Ys); 0 = off
    junction_merge_factor: float = 1.0
    # good continuation: join chains continuing within continuation_deg of
    # straight through a junction, tangents measured over continuation_len_px
    # just outside the junction zone
    merge_through_junctions: bool = True
    continuation_deg: float = 20.0
    continuation_len_px: float = 10.0
    # Douglas-Peucker tolerance per stroke / fill ring (px); 0 = raw pixels
    rdp_eps_px: float = 1.0
    # ink at least this wide (px) is a solid fill, outlined instead of
    # skeletonized; None/0 = off (the adapter sets it from FILL_MIN_WIDTH_PT)
    fill_min_width_px: "float | None" = None


# ----------------------------------------------------------------------- stages


def binarize(gray: np.ndarray, p: Params) -> np.ndarray:
    """Grayscale crop -> bool ink mask. Specks are removed by connected-
    component area, not by a morphological open (a 2x2 open erases every
    1-px-wide line, including thin diagonals)."""
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink = (otsu > 0) | (gray < p.soft_ink_thresh)
    ink = _drop_specks(ink, p.min_ink_area_px)
    return _fill_holes(ink, p.hole_max_area_px)


def _drop_specks(ink: np.ndarray, min_area: int) -> np.ndarray:
    if min_area > 1 and ink.any():
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
        small = stats[:, cv2.CC_STAT_AREA] < min_area
        small[0] = False  # background label
        if small.any():
            ink = ink & ~small[labels]
    return ink


def _fill_holes(ink: np.ndarray, max_area: int) -> np.ndarray:
    """Fill enclosed (not touching the crop border) paper components of at
    most `max_area` px, 4-connected."""
    if max_area <= 0 or not ink.any():
        return ink
    n, labels, stats, _ = cv2.connectedComponentsWithStats((~ink).astype(np.uint8), connectivity=4)
    x, y, w, h, area = (stats[:, i] for i in range(5))
    hw, hh = ink.shape[1], ink.shape[0]
    hole = (area <= max_area) & (x > 0) & (y > 0) & (x + w < hw) & (y + h < hh)
    hole[0] = False  # label 0 is ink here
    return ink | hole[labels] if hole.any() else ink


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    return skeletonize(mask.astype(bool))


def vectorize(strokes: list[list[Point]], p: Params) -> list[Polyline]:
    """Each stroke -> one Douglas-Peucker polyline (`simplify.fit_polyline`:
    ends fixed, closed loops re-anchored at an extreme point). No stroke-width
    estimate -- `Polyline.width` stays at its 1.0 default."""
    out = []
    for s in strokes:
        if len(s) >= 2:
            pts = fit_polyline(s, p.rdp_eps_px)
            if len(pts) >= 2:
                out.append(Polyline(points=pts))
    return out


def strokes_of(graph: Graph, p: Params) -> list[list[Point]]:
    if p.merge_through_junctions:
        return continue_through_junctions(graph, p.continuation_deg, p.continuation_len_px)
    return [list(c) for c in graph.chains]


# ----------------------------------------------------------------------- driver


def run(gray: np.ndarray, params: Params | None = None) -> PipelineResult:
    p = params or Params()
    t: dict[str, float] = {}

    def _t(name, fn):
        s = time.perf_counter()
        r = fn()
        t[name] = time.perf_counter() - s
        return r

    _LOG.debug("trace: %dx%d px crop", gray.shape[1], gray.shape[0])
    ink = _t("binarize", lambda: binarize(gray, p))
    _LOG.debug("  binarize: %d ink px (%.4fs)", int(ink.sum()), t["binarize"])
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    split = None
    line_ink = ink
    if p.fill_min_width_px:
        split = _t("fills", lambda: fills.split_fills(ink, dt, p.fill_min_width_px, p.rdp_eps_px))
        if split is not None:
            grown = cv2.dilate(split.fill.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            line_ink = _drop_specks(ink & ~grown, p.min_ink_area_px)
        _LOG.debug("  fills: %d outline(s) (%.4fs)", len(split.outlines) if split else 0, t["fills"])
    skeleton = _t("skeleton", lambda: skeletonize_mask(line_ink))
    _LOG.debug("  skeleton: %d skeleton px (%.4fs)", int(skeleton.sum()), t["skeleton"])
    graph = _t("graph", lambda: build_graph(
        skeleton, p.barb_min_px, dt=dt, junction_merge_factor=p.junction_merge_factor))
    _LOG.debug(
        "  graph: %d node(s), %d chain(s) (%.4fs)",
        len(graph.nodes), len(graph.chains), t["graph"],
    )

    def _vectorize():
        strokes = strokes_of(graph, p)
        if split is not None:
            nodes = set(graph.nodes)
            strokes = fills.snap_free_ends(strokes, lambda q: q not in nodes, split)
        return vectorize(strokes, p)

    polylines = _t("vectorize", _vectorize)
    _LOG.debug("  vectorize: %d polyline(s) (%.4fs)", len(polylines), t["vectorize"])
    return PipelineResult(
        params=p, ink=ink, skeleton=skeleton,
        graph=graph, polylines=polylines, timings=t,
        fill=split.fill if split else None, outlines=split.outlines if split else [],
    )
