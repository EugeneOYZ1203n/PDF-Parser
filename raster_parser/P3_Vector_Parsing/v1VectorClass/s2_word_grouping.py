"""Step 2 -- word grouping: fill-glyph candidates in content-stream order (`seqno`), chained into
words while each glyph is close to the previous one; each word group gets an orientation.

Copied from `ocr/wordgrouping.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np
import pymupdf as pymupdf

from raster_parser.commons.models import Vector


class Orientation(Enum):
    """Deduced text-line orientation for a word group (see ``orientation_of_word_group``)."""

    HORIZONTAL = 1
    VERTICAL = 2
    OTHER = 3


@dataclass
class Glyph:
    element_id: int
    x0: float
    y0: float
    x1: float
    y1: float
    vector: Vector

    @property
    def w(self) -> float: return self.x1 - self.x0
    @property
    def h(self) -> float: return self.y1 - self.y0


@dataclass
class WordGroup:
    text_label: str
    words: List[Glyph]
    x0: float
    x1: float
    y0: float
    y1: float
    bbox: pymupdf.Rect
    orientation: Orientation


def build_word_group(glyphs: List[Glyph], page_rotation: int) -> WordGroup:
    x0 = min(g.x0 for g in glyphs)
    y0 = min(g.y0 for g in glyphs)
    x1 = max(g.x1 for g in glyphs)
    y1 = max(g.y1 for g in glyphs)
    return WordGroup(
        text_label="",
        words=glyphs,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        bbox=pymupdf.Rect(x0, y0, x1, y1),
        orientation=orientation_of_word_group(glyphs, page_rotation)
    )


def get_vectors(word_group: WordGroup) -> List[Vector]:
    return [g.vector for g in word_group.words]


def convert_vectors_to_glyphs(vectors: List[Vector]) -> List[Glyph]:
    """Convert vectors to glyphs."""
    glyphs = []
    for vector in vectors:
        x0, y0, x1, y1 = vector.rect
        glyphs.append(Glyph(element_id=vector.seqno, x0=x0, y0=y0, x1=x1, y1=y1, vector=vector))
    return glyphs


def cluster_by_seqno(glyphs: List[Glyph], page_rotation: int = 0) -> List[WordGroup]:
    """Cluster glyphs by seqno."""

    sorted_list = sorted(glyphs, key=lambda g: g.vector.seqno)
    list_of_clusters: List[List[Glyph]] = []
    cluster: List[Glyph] = []
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


def is_close_to_glyph(g1: Glyph, g2: Glyph) -> bool:
    gap_x = max(0.0, max(g1.x0, g2.x0) - min(g1.x1, g2.x1))
    gap_y = max(0.0, max(g1.y0, g2.y0) - min(g1.y1, g2.y1))
    dx = abs((g1.x0 + g1.x1) * 0.5 - (g2.x0 + g2.x1) * 0.5)
    dy = abs((g1.y0 + g1.y1) * 0.5 - (g2.y0 + g2.y1) * 0.5)
    tol_h = max(g1.h, g2.h)
    tol_w = max(g1.w, g2.w)

    if dx >= dy:
        if gap_x < (g1.w + g2.w) * 0.5 and gap_y < tol_h * 0.5:
            return True
    elif gap_y < (g1.h + g2.h) * 0.5 and gap_x < tol_w * 0.5:
        return True

    if dx > 0.0 and dy > 0.0:
        axis_ratio = max(dx, dy) / min(dx, dy)
        if axis_ratio < 1.35:
            dist = np.sqrt((g1.x0 - g2.x0) ** 2 + (g1.y0 - g2.y0) ** 2)
            return dist < min(g1.w, g2.w) + min(g1.h, g2.h)

    return False


def default_orientation(page_rotation: int) -> Orientation:
    if page_rotation in (90, 270):
        return Orientation.VERTICAL
    return Orientation.HORIZONTAL


def orientation_of_word_group(glyphs: List[Glyph], page_orientation: int) -> Orientation:
    # If there are too few glyphs to calculate a direction vector, fall back to the page default
    if len(glyphs) < 2:
        return default_orientation(page_orientation)

    # 1. Sort glyphs horizontally or vertically based on layout positioning to ensure directional flow
    # Assuming glyphs have an x0 attribute
    sorted_glyphs = sorted(glyphs, key=lambda g: g.x0)

    # 2. Extract center points of the first and last glyph elements
    first = sorted_glyphs[0]
    last = sorted_glyphs[-1]

    cx_first = (first.x0 + first.x1) / 2
    cy_first = (first.y0 + first.y1) / 2

    cx_last = (last.x0 + last.x1) / 2
    cy_last = (last.y0 + last.y1) / 2

    # 3. Calculate the structural displacement vector delta
    dx = abs(cx_last - cx_first)
    dy = abs(cy_last - cy_first)

    # 4. Measure spatial movement flow rather than the outer bounding box size
    if dx >= dy and page_orientation == 0 or page_orientation == 180:
        return Orientation.HORIZONTAL
    elif dx < dy and page_orientation == 0 or page_orientation == 180:
        return Orientation.VERTICAL
    elif dx >= dy and page_orientation == 90 or page_orientation == 270:
        return Orientation.VERTICAL
    elif dx < dy and page_orientation == 90 or page_orientation == 270:
        return Orientation.HORIZONTAL
