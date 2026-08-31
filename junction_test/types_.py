"""Shared dataclasses for the spike (kept separate to avoid import cycles)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Point = tuple[float, float]


@dataclass
class Segment:
    p0: Point
    p1: Point
    width: float = 1.0
    thick: bool = False
    dashed: bool = False


@dataclass
class Arc:
    center: Point
    radius: float
    a0: float                 # start angle, degrees
    a1: float                 # end angle, degrees
    polyline: list[Point]
    closed: bool = False      # full circle
    width: float = 1.0


@dataclass
class Junction:
    xy: Point
    directions: list[float]   # headings of incident polyline first-segments, degrees 0..360


@dataclass
class Graph:
    nodes: list[Point]                 # (x, y)
    chains: list[list[Point]]          # ordered (x, y) pixel chains between nodes


@dataclass
class GroundTruth:
    segments: list[Segment]
    arcs: list[Arc]
    junctions: list[Junction]
    size: tuple[int, int]              # (h, w)


@dataclass
class PipelineResult:
    params: "object"
    gray: np.ndarray
    ink: np.ndarray
    text_mask: np.ndarray
    graphics_mask: np.ndarray
    thick_mask: np.ndarray
    thin_mask: np.ndarray
    skeleton: np.ndarray
    dist_map: np.ndarray
    graph: Graph
    polylines: list[list[Point]]
    segments: list[Segment]
    arcs: list[Arc]
    junctions: list[Junction]
    remainder: np.ndarray
    ocr_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
