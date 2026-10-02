"""Step-result bookkeeping for the CollinearVectorClass classification
chain (see `classify_vectors.py`). Each step's result is a `StepResult`
holding one or more named `CategoryResult`s -- exactly one per step has
`role="kept"` and feeds the next step; a `role="dropped"` category is folded
into the final drawing output (the collinear drawing groups, length
outliers and heavily crossed Vectors this backend removes)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
