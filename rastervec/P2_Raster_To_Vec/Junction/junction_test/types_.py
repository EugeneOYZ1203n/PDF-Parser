"""Dataclasses for the tracer (kept separate to avoid import cycles)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Point = tuple[float, float]


@dataclass
class Polyline:
    points: list[Point]                # ordered (x, y), one per skeleton pixel
    width: float = 1.0


@dataclass
class Graph:
    nodes: list[Point]                 # (x, y)
    chains: list[list[Point]]          # ordered (x, y) pixel chains between nodes


@dataclass
class PipelineResult:
    params: "object"
    ink: np.ndarray
    skeleton: np.ndarray
    graph: Graph
    polylines: list[Polyline]
    timings: dict[str, float] = field(default_factory=dict)
