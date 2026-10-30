"""Data types of the DeepTechVec raster_vectorize backend: a predicted primitive, one colour layer's
result, and the adapter's per-page state."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from raster_parser.models.text import Text
from raster_parser.models.vector import Vector

# chords a quadratic primitive's length is measured over
_LENGTH_CHORDS = 16


@dataclass
class Prim:
    """One primitive: `pts` is `(2, 2)` for a line segment or `(3, 2)` for a
    quadratic Bezier `(p1, control, p3)`; `width` in px."""

    pts: np.ndarray
    width: float
    conf: float = 1.0
    meta: dict = field(default_factory=dict)

    @property
    def is_line(self) -> bool:
        return len(self.pts) == 2

    @property
    def start(self) -> np.ndarray:
        return self.pts[0]

    @property
    def end(self) -> np.ndarray:
        return self.pts[-1]

    def length(self) -> float:
        """Arc length; a quadratic is measured over 16 chords."""
        if self.is_line:
            pts = np.asarray(self.pts, float)
        else:
            t = np.linspace(0.0, 1.0, _LENGTH_CHORDS + 1)[:, None]
            mt = 1.0 - t
            pts = mt * mt * self.pts[0] + 2.0 * mt * t * self.pts[1] + t * t * self.pts[2]
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


@dataclass
class LayerResult:
    prims: list[Prim]  # final, layer pixel frame
    patches: list[tuple[int, int, int]]  # (x0, y0, size) of every patch run
    raw: list[Prim]  # network output, layer frame, before refinement
    refined: list[Prim]  # after refinement + core clipping, before merging


@dataclass
class DeepTechVecPageState:
    vectors: list[Vector] = field(default_factory=list)
    texts: list[Text] = field(default_factory=list)
    seqno: int = 0
    boxes: dict = field(
        default_factory=lambda: {
            "patch": [],
        }
    )
    # page-space primitive geometry for the debug layers
    geom: dict = field(
        default_factory=lambda: {"raw": [], "raw_ends": [], "refined": [], "merged_ends": []}
    )

    def next_seqno(self) -> int:
        self.seqno += 1
        return self.seqno - 1


@dataclass
class DiffPolyline:
    points: np.ndarray
    width: float
