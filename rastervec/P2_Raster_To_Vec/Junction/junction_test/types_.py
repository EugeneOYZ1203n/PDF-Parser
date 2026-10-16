"""Dataclasses for the tracer (kept separate to avoid import cycles)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Point = tuple[float, float]


@dataclass
class Polyline:
    points: list[Point]                # ordered (x, y), Douglas-Peucker fitted
    width: float = 1.0


@dataclass
class Outline:
    """The boundary of one solid filled region: its outer ring plus one ring
    per hole, each closed (first point repeated last)."""
    rings: list[list[Point]]


@dataclass
class Graph:
    nodes: list[Point]                 # (x, y) junction nodes (degree >= 3)
    chains: list[list[Point]]          # ordered (x, y) chains, ends on their nodes
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
    fill: "np.ndarray | None" = None   # solid-region mask (None: no fills)
    outlines: list[Outline] = field(default_factory=list)
