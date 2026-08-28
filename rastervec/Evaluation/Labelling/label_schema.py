"""Sidecar JSON label format for ground-truth vector-cluster text, used by
both `manual_label.py` (human-entered) and `auto_label.py` (derived from
native text, independent of any pipeline run) -- and consumed by
`Evaluation/Evaluate/evaluate.py` to score a pipeline run against these
labels.

One `LabelEntry` per labelled region. `cluster_signature`'s meaning
differs by `source`: for `source="manual"`, it's `cluster_signature()`
below -- a deterministic string built from a real clustered-run's own
member count and rounded bbox, stable across repeated runs of the *same*
pipeline over the *same* PDF (VectorPath instances have no identity across
runs, so this lets a label re-match a freshly re-clustered run's clusters).
For `source="auto"`, there is no clustered run backing the label at all
(see auto_label.py's own docstring) -- `cluster_signature` there is a
`f"line:{page_index}:{block_no}:{line_no}"` native-text line-region id
instead, not a VectorPath-cluster signature.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from rastervec.helpers.geometry import union_bbox

if TYPE_CHECKING:
    from rastervec.models import VectorPath

LabelSource = Literal["manual", "auto"]


class LabelEntry(BaseModel):
    page_index: int
    cluster_bbox: tuple[float, float, float, float]
    cluster_signature: str
    text: str
    source: LabelSource
    # Ground-truth rotation (degrees) this cluster's text should read at --
    # 0 for every auto-labelled entry (Conversion never rotates text), a
    # manual labeller can set this explicitly for a rotated cluster. Used
    # by Evaluation/Evaluate/evaluate.py's rotation-accuracy metric.
    expected_rotation: int = 0


class LabelSet(BaseModel):
    """Every labelled cluster for one PDF."""

    pdf_path: str
    entries: list[LabelEntry] = Field(default_factory=list)


def cluster_signature(cluster: "list[VectorPath]") -> str:
    """A deterministic, order-independent identifier for a cluster -- its
    own member count plus its rounded union bbox. Two clusters with the
    same members (regardless of Python object identity, which doesn't
    survive a fresh pipeline run) produce the same signature."""
    x0, y0, x1, y1 = union_bbox([p.bbox for p in cluster])
    return f"{len(cluster)}:{x0:.1f}:{y0:.1f}:{x1:.1f}:{y1:.1f}"


def save_labels(labels: LabelSet, path: str) -> None:
    Path(path).write_text(labels.model_dump_json(indent=2), encoding="utf-8")


def load_labels(path: str) -> LabelSet:
    return LabelSet.model_validate_json(Path(path).read_text(encoding="utf-8"))
