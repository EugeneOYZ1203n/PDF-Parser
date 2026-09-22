"""Ported from archive/raster_parser/ocr/wordgrouping.py -- seqno-adjacency
glyph clustering into word groups, with an orientation guess from each
group's own horizontal spread relative to page rotation. Retargeted at
commons.models.Vector (no pymupdf.Rect dependency -- plain tuple bboxes).

Own duplicated copy -- per the "sibling P3 backends share zero code" rule,
this is not imported from `P3_Vector_Parsing/LegacyRecreation/wordgrouping.py`
even though the two are identical; each backend duplicates its own copy of
whatever infra it needs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from rastervec.commons.models import Vector

Orientation = Literal["horizontal", "vertical"]


@dataclass
class Glyph:
    x0: float
    y0: float
    x1: float
    y1: float
    vector: Vector

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0


@dataclass
class WordGroup:
    words: list[Glyph]
    bbox: tuple[float, float, float, float]
    orientation: Orientation


def convert_vectors_to_glyphs(vectors: list[Vector]) -> list[Glyph]:
    glyphs = []
    for vector in vectors:
        x0, y0, x1, y1 = vector.rect
        glyphs.append(Glyph(x0=x0, y0=y0, x1=x1, y1=y1, vector=vector))
    return glyphs


def get_vectors(word_group: WordGroup) -> list[Vector]:
    return [g.vector for g in word_group.words]


def _default_orientation(page_rotation: int) -> Orientation:
    return "vertical" if page_rotation in (90, 270) else "horizontal"


def _orientation_of_word_group(glyphs: list[Glyph], page_rotation: int) -> Orientation:
    if len(glyphs) < 2:
        return _default_orientation(page_rotation)
    sorted_glyphs = sorted(glyphs, key=lambda g: g.x0)
    first, last = sorted_glyphs[0], sorted_glyphs[-1]
    cx_first, cy_first = (first.x0 + first.x1) / 2, (first.y0 + first.y1) / 2
    cx_last, cy_last = (last.x0 + last.x1) / 2, (last.y0 + last.y1) / 2
    dx, dy = abs(cx_last - cx_first), abs(cy_last - cy_first)
    page_upright = page_rotation in (0, 180)
    long_axis_is_dx = dx >= dy
    if page_upright:
        return "horizontal" if long_axis_is_dx else "vertical"
    return "vertical" if long_axis_is_dx else "horizontal"


def _build_word_group(glyphs: list[Glyph], page_rotation: int) -> WordGroup:
    x0 = min(g.x0 for g in glyphs)
    y0 = min(g.y0 for g in glyphs)
    x1 = max(g.x1 for g in glyphs)
    y1 = max(g.y1 for g in glyphs)
    return WordGroup(
        words=glyphs, bbox=(x0, y0, x1, y1),
        orientation=_orientation_of_word_group(glyphs, page_rotation),
    )


def _is_close_to_glyph(g1: Glyph, g2: Glyph) -> bool:
    gap_x = max(0.0, max(g1.x0, g2.x0) - min(g1.x1, g2.x1))
    gap_y = max(0.0, max(g1.y0, g2.y0) - min(g1.y1, g2.y1))
    dx = abs((g1.x0 + g1.x1) * 0.5 - (g2.x0 + g2.x1) * 0.5)
    dy = abs((g1.y0 + g1.y1) * 0.5 - (g2.y0 + g2.y1) * 0.5)
    tol_h, tol_w = max(g1.h, g2.h), max(g1.w, g2.w)

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


def cluster_by_seqno(glyphs: list[Glyph], page_rotation: int = 0) -> list[WordGroup]:
    """Sorts glyphs by content-stream draw order (seqno), chain-merging
    consecutive glyphs into one word group whenever adjacent -- the same
    seqno-adjacency heuristic archive/raster_parser's Type-2 path used."""
    sorted_glyphs = sorted(glyphs, key=lambda g: g.vector.seqno)
    clusters: list[list[Glyph]] = []
    cluster: list[Glyph] = []
    for g in sorted_glyphs:
        if not cluster:
            cluster.append(g)
        elif _is_close_to_glyph(g, cluster[-1]):
            cluster.append(g)
        else:
            clusters.append(cluster)
            cluster = [g]
    if cluster:
        clusters.append(cluster)
    return [_build_word_group(c, page_rotation) for c in clusters]
