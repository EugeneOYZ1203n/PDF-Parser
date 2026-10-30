"""Data types of the InvolutionStroke raster_vectorize backend: one tile's and one colour layer's
results, the stroke pieces joined across tile seams, and the adapter's per-page state."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from raster_parser.models.text import Text
from raster_parser.models.vector import Vector


@dataclass
class TileResult:
    polylines: list[np.ndarray] = field(default_factory=list)  # graph polylines (tile px)
    nodes: list[tuple[float, float]] = field(default_factory=list)  # polyline end nodes (tile px)
    strokes: list[np.ndarray] = field(default_factory=list)  # matched strokes (tile px)
    centerline: "np.ndarray | None" = None


@dataclass
class StrokePiece:
    pts: np.ndarray  # layer px
    tile: int
    core: tuple[float, float, float, float]


@dataclass
class LayerResult:
    strokes: list[np.ndarray]  # (K, 4, 2) cubic pieces, layer px
    polylines: list[np.ndarray]  # joined strokes before smoothing, layer px
    tiles: list[tuple[int, int, int, int]]  # (x0, y0, w, h) of every tile run
    failed_tiles: list[tuple[int, int, int, int]]
    graph_polylines: list[np.ndarray]  # upstream graph polylines, layer px (debug)
    graph_nodes: list[tuple[float, float]]  # their end nodes, layer px (debug)
    matched: list[np.ndarray]  # matched strokes before clipping, layer px (debug)
    joins: list[tuple[float, float]]  # seam join points, layer px
    n_graph_polylines: int = 0  # upstream graph polylines over all tiles
    centerline: "np.ndarray | None" = None  # stitched centerline image (H, W) uint8, debug only


@dataclass
class InvolutionPageState:
    vectors: list[Vector] = field(default_factory=list)
    texts: list[Text] = field(default_factory=list)
    seqno: int = 0
    boxes: dict = field(
        default_factory=lambda: {
            "vec_tile": [],
            "vec_tile_failed": [],
        }
    )
    # page-space geometry for the debug layers
    geom: dict = field(
        default_factory=lambda: {
            "graph": [],
            "nodes": [],
            "matched": [],
            "joins": [],
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
