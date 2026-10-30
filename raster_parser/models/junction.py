"""Data types of the Junction raster_vectorize backend: the tracer's parameters, graph and result,
solid-fill splits, layer components and the adapter's per-page state."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from raster_parser.models.text import Text
from raster_parser.models.vector import Vector

Point = tuple[float, float]
# `SkeletonEdge.a` / `.b` when that end has no node
FREE_END = -1


@dataclass
class Polyline:
    points: list[Point]  # ordered (x, y), Douglas-Peucker fitted
    width: float = 1.0


@dataclass
class Outline:
    """The boundary of one solid filled region: its outer ring plus one ring
    per hole, each closed (first point repeated last)."""

    rings: list[list[Point]]


@dataclass
class Graph:
    nodes: list[Point]  # (x, y) junction nodes (degree >= 3)
    chains: list[list[Point]]  # ordered (x, y) chains, ends on their nodes
    # per chain: (start node index, end node index), -1 = free end
    ends: list[tuple[int, int]] = field(default_factory=list)
    node_widths: list[float] = field(default_factory=list)  # local stroke width per node


@dataclass
class PipelineResult:
    params: "object"
    ink: np.ndarray
    skeleton: np.ndarray
    graph: Graph
    polylines: list[Polyline]
    timings: dict[str, float] = field(default_factory=dict)
    fill: "np.ndarray | None" = None  # solid-region mask (None: no fills)
    outlines: list[Outline] = field(default_factory=list)


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


@dataclass
class FillSplit:
    fill: np.ndarray  # bool mask of the solid regions
    outlines: list[Outline]
    snap_px: float  # free stroke ends this close to the fill snap onto it


@dataclass
class SkeletonEdge:
    pts: list[Point]  # interior points (node positions are added on demand)
    a: int = FREE_END  # node at the start, or FREE_END
    b: int = FREE_END  # node at the end, or FREE_END
    closed: bool = False  # node-less loop: pts[0] == pts[-1]

    def reversed(self) -> "SkeletonEdge":
        return SkeletonEdge(self.pts[::-1], self.b, self.a, self.closed)


@dataclass
class SkeletonGraph:
    pos: dict[int, Point] = field(default_factory=dict)
    width: dict[int, float] = field(default_factory=dict)
    edges: dict[int, SkeletonEdge] = field(default_factory=dict)
    inc: dict[int, list[int]] = field(
        default_factory=dict
    )  # node -> edge ids (twice for a self-loop)
    _next_edge: int = 0

    def add_edge(self, e: SkeletonEdge) -> int:
        eid = self._next_edge
        self._next_edge += 1
        self.edges[eid] = e
        for n in (e.a, e.b):
            if n != FREE_END:
                self.inc[n].append(eid)
        return eid

    def remove_edge(self, eid: int) -> SkeletonEdge:
        e = self.edges.pop(eid)
        for n in (e.a, e.b):
            if n != FREE_END:
                self.inc[n].remove(eid)
        return e

    def points(self, e: SkeletonEdge) -> list[Point]:
        out = list(e.pts)
        if e.a != FREE_END:
            out.insert(0, self.pos[e.a])
        if e.b != FREE_END:
            out.append(self.pos[e.b])
        return out

    def degree(self, n: int) -> int:
        return len(self.inc[n])

    def remove_node(self, n: int) -> None:
        self.pos.pop(n, None)
        self.width.pop(n, None)
        self.inc.pop(n, None)


@dataclass
class Component:
    x0: int
    y0: int
    gray: np.ndarray  # uint8 crop, 255 outside this component's ink
    n_pixels: int


@dataclass
class JunctionPageState:
    vectors: list[Vector] = field(default_factory=list)
    # `vectors` again, for debug drawing only (same seqnos)
    display_vectors: list[Vector] = field(default_factory=list)
    texts: list[Text] = field(default_factory=list)
    seqno: int = 0
    boxes: dict = field(default_factory=lambda: {"component": []})
    # page-space tracing geometry for the point / line debug layers
    geom: dict = field(
        default_factory=lambda: {
            "chains": [],
            "junctions": [],
            "endpoints": [],
            "polylines": [],
            "polyline_ends": [],
            "outlines": [],
        }
    )

    def next_seqno(self) -> int:
        self.seqno += 1
        return self.seqno - 1
