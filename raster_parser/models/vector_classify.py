"""Intermediate results of the vector_classify backends.

- v1_vector_class: word groups (`WordGroup`) and their `Orientation`.
- v2_vector_class: per-step classification results (`StepResult` holding named
  `CategoryResult`s, collected in a `ClassificationResult`) and the line / quad geometry the
  steps share (`LineFit`, `QuadGeom`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Literal

import numpy as np

from raster_parser.models.vector import Vector


class Orientation(Enum):
    """Deduced text-line orientation for a word group (see ``orientation_of_word_group``)."""

    HORIZONTAL = 1
    VERTICAL = 2
    OTHER = 3


@dataclass
class WordGroup:
    vectors: List[Vector]  # the group's glyphs, in seqno order
    bbox: tuple[float, float, float, float]
    orientation: Orientation


CategoryRole = Literal["kept", "dropped", "info"]


@dataclass
class CategoryResult:
    """One named category within a step's result -- a list of entries plus
    its role. Each entry is `list[Vector]` (a group) or `list[list[Vector]]`
    (a cluster, tiered by member group)."""

    groups: list
    role: CategoryRole


@dataclass
class StepResult:
    """One step's full result: a display label plus every named category
    it produced (`"kept"` always present)."""

    label: str
    categories: dict[str, CategoryResult]


@dataclass
class ClassificationResult:
    """`text_clusters` is tiered (`list[list[list[Vector]]]` -- per cluster,
    its member groups). `drawing_vectors` is flat -- every dropped Vector.
    `global_angles` is the page's deduped global potential angles (deg,
    folded to [0, 180), sorted -- see the module docstring)."""

    text_clusters: list[list[list[Vector]]]
    drawing_vectors: list[Vector]
    clustering: dict
    global_angles: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class LineFit:
    """A straight Vector's line: `angle` folded to [0, 180) degrees,
    `midpoint` of its extent, `length` = extent along the line."""

    angle: float
    midpoint: tuple[float, float]
    length: float


@dataclass(frozen=True)
class QuadGeom:
    """Everything the ownership tests derive from one convex quad, built
    once (`quad_geom`) and reused for every Vector tested against it --
    `parse._text_vectors_by_quad` tests every Vector of a cluster against
    every quad of that cluster."""

    pts: tuple  # the quad's points as plain floats
    edges: tuple  # (ax, ay, ex, ey) per edge, for bbox_inside_quad
    poly: np.ndarray  # (n, 2)
    normals: np.ndarray  # (n, 2) inward edge normals
    signed_area2: float  # twice the signed area (0 -> degenerate)
    envelope: tuple[float, float, float, float]
    area: float
