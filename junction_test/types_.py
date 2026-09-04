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
    # --- richer AEC attributes (SYNTHETIC_DATA.md schema); defaults keep old callers working
    color: tuple[int, int, int] = (40, 40, 40)     # RGB ink colour
    dash_style: str = "solid"                       # solid|dashed|hidden|center|phantom
    dash_array: tuple[float, ...] = ()              # on/off run lengths, px
    role: str = ""                                  # wall|dimension_line|contour|... (see md)
    layer: str = ""                                 # colour/annotation layer name


@dataclass
class Arc:
    center: Point
    radius: float
    a0: float                 # start angle, degrees
    a1: float                 # end angle, degrees
    polyline: list[Point]
    closed: bool = False      # full circle
    width: float = 1.0
    color: tuple[int, int, int] = (40, 40, 40)
    dash_style: str = "solid"
    role: str = ""
    layer: str = ""


@dataclass
class Bezier:
    """Cubic Bezier primitive (topo contours / curved roads / freeform walls)."""
    p0: Point
    c0: Point
    c1: Point
    p1: Point
    polyline: list[Point]
    width: float = 1.0
    color: tuple[int, int, int] = (40, 40, 40)
    dash_style: str = "solid"
    role: str = ""
    layer: str = ""


@dataclass
class Junction:
    xy: Point
    directions: list[float]   # headings of incident polyline first-segments, degrees 0..360
    jtype: str = ""           # L|T|X|Y|star|endpoint|coincident_unrelated
    arm_angles: list[float] = field(default_factory=list)
    members: list[int] = field(default_factory=list)          # indices into GroundTruth.segments
    is_true_connection: bool = True                            # False for coincident_unrelated


@dataclass
class Graph:
    nodes: list[Point]                 # (x, y)
    chains: list[list[Point]]          # ordered (x, y) pixel chains between nodes


@dataclass
class StaircaseRegion:
    polygon: list[Point]        # convex hull of all member tread endpoints
    treads: list[Segment]       # the individual tread segments that make up the run
    axis: tuple[Point, Point]   # walking-direction axis endpoints
    spacing: float              # mean gap between consecutive treads, px
    n_treads: int                # == len(treads); paper's crude 5..30 filter operates on this


@dataclass
class SymbolInstance:
    family: str                             # "door" | "window" (extensible)
    features: list[Segment | Arc]           # the underlying primitives consumed by the match
    anchor: Point                           # representative point for matching (hinge / gap midpoint)
    bbox: tuple[float, float, float, float] # (xmin, ymin, xmax, ymax) over all features
    error: float                            # accumulated constraint error at the NNFinal node


@dataclass
class GroundTruth:
    segments: list[Segment]
    arcs: list[Arc]
    junctions: list[Junction]
    size: tuple[int, int]              # (h, w)
    staircases: list[StaircaseRegion] = field(default_factory=list)
    symbols: list[SymbolInstance] = field(default_factory=list)
    beziers: list[Bezier] = field(default_factory=list)
    meta: dict = field(default_factory=dict)          # archetype, dpi, difficulty knobs


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
    beziers: list[Bezier] = field(default_factory=list)
    ocr_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    staircases: list[StaircaseRegion] = field(default_factory=list)
    symbols: list[SymbolInstance] = field(default_factory=list)
