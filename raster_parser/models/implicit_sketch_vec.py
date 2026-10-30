"""Data types of the ImplicitSketchVec raster_vectorize backend: the page-wide harvest of tile
outputs, the dual-contouring graph, one colour layer's result, training crops, and the adapter's
per-page state."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from raster_parser.models.text import Text
from raster_parser.models.vector import Vector


def node_order_key(n) -> tuple:
    """A total order over the node kinds: cells (a, b), surgery vertices
    ("new", i), downsampled blocks ("blk", (a, b))."""
    if isinstance(n[0], (int, np.integer)):
        return (0, int(n[0]), int(n[1]))
    if n[0] == "new":
        return (1, int(n[1]), 0)
    return (2, int(n[1][0]), int(n[1][1]))


@dataclass
class SparseHarvest:
    right: list = field(default_factory=list)  # (N, 2) global cells, refined flags
    bottom: list = field(default_factory=list)
    raw_right: list = field(default_factory=list)  # before flag refinement (debug)
    raw_bottom: list = field(default_factory=list)
    vertex: dict = field(default_factory=dict)  # global cell -> px
    keypoints: dict = field(default_factory=dict)  # keypoint type -> [] (the harvester fills it)
    usm: set = field(default_factory=set)


@dataclass
class LayerResult:
    strokes: list[np.ndarray]  # (K, 4, 2) cubic pieces, layer px
    polylines: list[np.ndarray]  # grouped lines before smoothing
    tiles: list[tuple[int, int, int]]  # (x0, y0, size) of every tile run
    keypoints: dict  # type -> (K, 2) px
    usm_cells: set  # global 1/SR px cells
    regions: list  # cell sets refined by topology surgery
    raw_segments: list  # DC edges straight from the network flags
    graph_segments: list  # DC edges after all graph post-processing
    heat: "np.ndarray | None" = None  # centerline UDF heat map (H, W) uint8, debug only


@dataclass
class DCGraph:
    pos: dict = field(default_factory=dict)
    adj: dict = field(default_factory=dict)
    _new: int = 0

    def add_node(self, node, p) -> None:
        self.pos[node] = np.asarray(p, float)
        self.adj.setdefault(node, set())

    def new_node(self, p):
        node = ("new", self._new)
        self._new += 1
        self.add_node(node, p)
        return node

    def add_edge(self, u, v) -> None:
        if u == v:
            return
        self.adj.setdefault(u, set()).add(v)
        self.adj.setdefault(v, set()).add(u)

    def remove_node(self, u) -> None:
        for v in self.adj.pop(u, ()):
            self.adj[v].discard(u)
        self.pos.pop(u, None)

    def degree(self, u) -> int:
        return len(self.adj.get(u, ()))

    def edges(self):
        for u, vs in self.adj.items():
            for v in vs:
                if node_order_key(u) < node_order_key(v):
                    yield u, v

    def segments(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(self.pos[u], self.pos[v]) for u, v in self.edges()]


@dataclass
class Crop:
    gray: np.ndarray  # (S, S) uint8
    strokes: list[np.ndarray]  # crop px, clipped to the crop + GT_MARGIN_PX


@dataclass
class ImplicitSketchVecPageState:
    vectors: list[Vector] = field(default_factory=list)
    texts: list[Text] = field(default_factory=list)
    seqno: int = 0
    boxes: dict = field(
        default_factory=lambda: {
            "vec_tile": [],
            "region": [],
        }
    )
    # page-space geometry for the debug layers
    geom: dict = field(
        default_factory=lambda: {
            "end": [],
            "sharp": [],
            "junc": [],
            "grouped": [],
            "ends": [],
        }
    )

    def next_seqno(self) -> int:
        self.seqno += 1
        return self.seqno - 1


@dataclass
class DiffPolyline:
    points: np.ndarray
    width: float
