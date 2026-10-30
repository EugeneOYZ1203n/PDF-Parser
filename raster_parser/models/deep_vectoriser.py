"""Data types of the DeepVectoriser raster_vectorize backend: per-tile predictions and strokes,
one colour layer's result, training crops, the dataset viewer's rows, and the adapter's per-page state."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from raster_parser.models.raster_text import LayerData
from raster_parser.models.text import Text
from raster_parser.models.vector import Vector


@dataclass
class TileStrokes:
    """One tile's predicted strokes, already in the layer's pixel frame."""

    tile_id: int
    core: tuple[float, float, float, float]
    strokes: list[np.ndarray]  # (K, 4, 2) cubic pieces each


@dataclass
class TilePrediction:
    """One tile's output: strokes `(K, 4, 2)` in the tile's own pixel frame."""

    strokes: list[np.ndarray] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    n_confident: int = 0
    n_queries: int = 1


@dataclass
class LayerResult:
    strokes: list[np.ndarray]  # merged, layer pixel frame
    tiles: list[tuple[int, int, int]]  # (x0, y0, size) of every tile run
    resplit: list[tuple[int, int, int]]  # saturated tiles that were split
    raw: list[np.ndarray]  # every tile's strokes before merging


@dataclass
class Crop:
    gray: np.ndarray  # (S, S) uint8
    strokes: list[np.ndarray] = field(default_factory=list)  # sorted, oriented, local px
    widths: list[float] = field(default_factory=list)
    # `sample_crop` only (not kept by augment): the (x0, y0, side) window it
    # cut from the layer, and how many strokes it held before truncation
    window: "tuple[int, int, int] | None" = None
    n_found: int = 0


@dataclass
class LayerRow:
    key: str
    page_key: str
    layer: int
    split: str  # "train" | "val" | "unindexed" (saved, not in index.json) | "unsaved"
    n_strokes: int
    color: tuple[int, int, int]
    saved: bool
    pdf: str
    page: int


@dataclass
class DatasetInfo:
    data_dir: Path
    index: dict
    pages: dict[str, dict]  # page key -> its manifest
    rows: list[LayerRow]  # sorted by (page key, layer)


@dataclass
class LoadedLayer:
    data: LayerData
    color: tuple[int, int, int]
    origin: tuple[int, int]  # the crop's (x0, y0) on the page (canonical px)
    page_shape: tuple[int, int]  # (h, w) of the full page mask


@dataclass
class DeepVectoriserPageState:
    vectors: list[Vector] = field(default_factory=list)
    texts: list[Text] = field(default_factory=list)
    seqno: int = 0
    boxes: dict = field(
        default_factory=lambda: {
            "vec_tile": [],
            "vec_resplit": [],
        }
    )
    # page-space stroke geometry for the debug layers
    geom: dict = field(default_factory=lambda: {"raw": [], "raw_ends": [], "merged_ends": []})

    def next_seqno(self) -> int:
        self.seqno += 1
        return self.seqno - 1


@dataclass
class DiffPolyline:
    points: np.ndarray
    width: float
