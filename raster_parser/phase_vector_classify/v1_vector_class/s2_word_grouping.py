"""Step 2 -- word grouping: fill-glyph candidates in content-stream order (`seqno`), chained into
words while each glyph is close to the previous one; each word group gets an orientation.
Glyphs are the shared `Vector`s themselves (their `rect` is the glyph box).

Copied from `ocr/wordgrouping.py`.
"""

from __future__ import annotations

from typing import List

import numpy as np

from raster_parser.commons.helpers.geometry import union_bbox
from raster_parser.models.vector import Vector
from raster_parser.models.vector_classify import Orientation, WordGroup

# Two glyphs side by side join when the gap along the reading direction is under the mean
# of their widths (half the summed widths), and the cross-direction offset is under half the
# taller glyph.
GLYPH_GAP_FRAC = 0.5
# Centre offsets whose longer axis is less than this many times the shorter one are diagonal
# (neither side by side nor stacked); such glyphs join only when their top-left corners are
# within one glyph width + height of each other.
DIAGONAL_MAX_AXIS_RATIO = 1.35


def build_word_group(glyphs: List[Vector], page_rotation: int) -> WordGroup:
    return WordGroup(
        vectors=glyphs,
        bbox=union_bbox([g.rect for g in glyphs]),
        orientation=orientation_of_word_group(glyphs, page_rotation),
    )


def cluster_by_seqno(glyphs: List[Vector], page_rotation: int = 0) -> List[WordGroup]:
    """Chain glyphs, in seqno order, into word groups while each is close to the previous one.

    ``seqno`` is PyMuPDF's paint-order index of a drawing on the page, so glyphs of one word,
    drawn one after another, are consecutive in it.
    """

    sorted_list = sorted(glyphs, key=lambda g: g.seqno)
    list_of_clusters: List[List[Vector]] = []
    cluster: List[Vector] = []
    for g in sorted_list:
        if len(cluster) == 0:
            cluster.append(g)
            continue
        if is_close_to_glyph(g, cluster[-1]):
            cluster.append(g)
        else:
            list_of_clusters.append(cluster)
            cluster = [g]
    if cluster:
        list_of_clusters.append(cluster)
    return [build_word_group(cluster, page_rotation) for cluster in list_of_clusters]


def is_close_to_glyph(g1: Vector, g2: Vector) -> bool:
    ax0, ay0, ax1, ay1 = g1.rect
    bx0, by0, bx1, by1 = g2.rect
    aw, ah, bw, bh = ax1 - ax0, ay1 - ay0, bx1 - bx0, by1 - by0
    gap_x = max(0.0, max(ax0, bx0) - min(ax1, bx1))
    gap_y = max(0.0, max(ay0, by0) - min(ay1, by1))
    dx = abs((ax0 + ax1) * 0.5 - (bx0 + bx1) * 0.5)
    dy = abs((ay0 + ay1) * 0.5 - (by0 + by1) * 0.5)
    tol_h = max(ah, bh)
    tol_w = max(aw, bw)

    if dx >= dy:
        if gap_x < (aw + bw) * GLYPH_GAP_FRAC and gap_y < tol_h * GLYPH_GAP_FRAC:
            return True
    elif gap_y < (ah + bh) * GLYPH_GAP_FRAC and gap_x < tol_w * GLYPH_GAP_FRAC:
        return True

    if dx > 0.0 and dy > 0.0:
        axis_ratio = max(dx, dy) / min(dx, dy)
        if axis_ratio < DIAGONAL_MAX_AXIS_RATIO:
            dist = np.sqrt((ax0 - bx0) ** 2 + (ay0 - by0) ** 2)
            return dist < min(aw, bw) + min(ah, bh)

    return False


def default_orientation(page_rotation: int) -> Orientation:
    if page_rotation in (90, 270):
        return Orientation.VERTICAL
    return Orientation.HORIZONTAL


def orientation_of_word_group(glyphs: List[Vector], page_orientation: int) -> Orientation:
    # If there are too few glyphs to calculate a direction vector, fall back to the page default
    if len(glyphs) < 2:
        return default_orientation(page_orientation)

    # 1. Sort glyphs horizontally or vertically based on layout positioning to ensure directional flow
    sorted_glyphs = sorted(glyphs, key=lambda g: g.rect[0])

    # 2. Extract center points of the first and last glyph elements
    fx0, fy0, fx1, fy1 = sorted_glyphs[0].rect
    lx0, ly0, lx1, ly1 = sorted_glyphs[-1].rect

    cx_first = (fx0 + fx1) / 2
    cy_first = (fy0 + fy1) / 2

    cx_last = (lx0 + lx1) / 2
    cy_last = (ly0 + ly1) / 2

    # 3. Calculate the structural displacement vector delta
    dx = abs(cx_last - cx_first)
    dy = abs(cy_last - cy_first)

    # 4. Measure spatial movement flow rather than the outer bounding box size. The flow is
    #    horizontal when the run is wider than tall in page space; on a quarter-turned page
    #    (90 / 270) it reads the other way round.
    flows_horizontally = dx >= dy
    if page_orientation in (0, 180):
        return Orientation.HORIZONTAL if flows_horizontally else Orientation.VERTICAL
    if page_orientation in (90, 270):
        return Orientation.VERTICAL if flows_horizontally else Orientation.HORIZONTAL
    return default_orientation(page_orientation)
