"""The centerline tracer Junction runs on each component crop:

    binarize -> skeletonize -> chains (barb pruning)
             -> each chain emitted as a raw polyline (one vertex per skeleton pixel)

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

from .skeleton_graph import build_graph
from .types_ import Graph, PipelineResult, Polyline

_LOG = get_logger("P2.Junction.trace")


@dataclass
class Params:
    # binarisation: Otsu OR gray < soft_ink_thresh; ink blobs smaller than
    # min_ink_area_px (8-connected) are specks and dropped
    soft_ink_thresh: int = 245
    min_ink_area_px: int = 4
    # skeleton graph: leaf branches off a junction shorter than this are barbs
    barb_min_px: float = 9.0


# ----------------------------------------------------------------------- stages


def binarize(gray: np.ndarray, p: Params) -> np.ndarray:
    """Grayscale crop -> bool ink mask. Specks are removed by connected-
    component area, not by a morphological open (a 2x2 open erases every
    1-px-wide line, including thin diagonals)."""
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink = (otsu > 0) | (gray < p.soft_ink_thresh)
    if p.min_ink_area_px > 1 and ink.any():
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
        small = stats[:, cv2.CC_STAT_AREA] < p.min_ink_area_px
        small[0] = False  # background label
        if small.any():
            ink &= ~small[labels]
    return ink


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    return skeletonize(mask.astype(bool))


def vectorize(graph: Graph) -> list[Polyline]:
    """Each chain -> one raw polyline through every one of its skeleton
    pixels (no simplification, no regularisation, no stroke-width estimate
    -- `Polyline.width` stays at its 1.0 default). A closed loop keeps its
    repeated first/last point."""
    return [Polyline(points=list(chain)) for chain in graph.chains if len(chain) >= 2]


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
    skeleton = _t("skeleton", lambda: skeletonize_mask(ink))
    _LOG.debug("  skeleton: %d skeleton px (%.4fs)", int(skeleton.sum()), t["skeleton"])
    graph = _t("graph", lambda: build_graph(skeleton, p.barb_min_px))
    _LOG.debug(
        "  graph: %d node(s), %d chain(s) (%.4fs)",
        len(graph.nodes), len(graph.chains), t["graph"],
    )
    polylines = _t("vectorize", lambda: vectorize(graph))
    _LOG.debug("  vectorize: %d polyline(s) (%.4fs)", len(polylines), t["vectorize"])
    return PipelineResult(
        params=p, ink=ink, skeleton=skeleton,
        graph=graph, polylines=polylines, timings=t,
    )
