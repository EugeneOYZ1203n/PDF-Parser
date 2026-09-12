"""Sidecar JSON label format for ground-truth vector-cluster text, used by
`scripts/label/vector_label.py` (human-entered), `scripts/label/
native_label.py` (derived from native text, independent of any pipeline
run), and `scripts/label/raster_label.py` (human-entered over a rasterized
page, no backing vectors) -- and consumed by `Evaluation/Evaluate/
evaluate.py` to score a pipeline run against these labels.

One `LabelEntry` per labelled region, identified by its stable `label_id`
(not `cluster_signature`, which can go stale the moment a `vector_label`
edit changes the entry's own bbox/vector set). `cluster_signature` is
informational/debug-only now: for `source="vector"`/`source="raster"`, it's
`cluster_signature()` below -- a deterministic string built from a real
clustered-run's own member count and rounded bbox (`"raster:..."` for a
raster entry, which has no backing vectors at all). For `source="native"`,
there is no clustered run backing the label at all (see native_label.py's
own docstring) -- `cluster_signature` there is a
`f"line:{page_index}:{block_no}:{line_no}"` native-text line-region id
instead, not a VectorPath-cluster signature.

`GeometryAnnotation` is a separate, non-text kind of ground truth --
hand-drawn lines/curves for a future raster-image line-tracing pipeline
stage, produced only by `raster_label.py`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from rastervec.helpers.geometry import item_points, union_bbox

if TYPE_CHECKING:
    from rastervec.models import Vector

LabelSource = Literal["native", "vector", "raster"]


class LabelEntry(BaseModel):
    page_index: int
    cluster_bbox: tuple[float, float, float, float]
    cluster_signature: str
    # Stable identity, independent of the entry's current vector set/bbox/
    # text -- lookup key for "edit this label in place". `vector_label`/
    # `raster_label`: uuid4().hex at creation. `native_label`: reuses its
    # existing deterministic `f"line:{page_index}:{block_no}:{line_no}"`
    # (already stable/meaningful, no random id needed there).
    label_id: str
    text: str
    source: LabelSource
    # Ground-truth rotation (degrees) this cluster's text should read at --
    # 0 for every native-labelled entry (Conversion never rotates text), a
    # human labeller can set this explicitly for a rotated cluster. Used
    # by Evaluation/Evaluate/evaluate.py's rotation-accuracy metric.
    expected_rotation: int = 0
    # `path_signature()` of every `Vector` that composed this cluster, sorted,
    # deduped. Lets an external script re-run `extract_vectors` on the same
    # PDF and match each label back to its exact drawing paths. Empty for
    # `source="raster"` (no backing vectors).
    vector_signatures: list[str] = Field(default_factory=list)


class GeometryAnnotation(BaseModel):
    """A hand-drawn non-text ground-truth shape (`raster_label.py`'s line/
    curve tools), for a future raster-image line-tracing pipeline stage.
    Same convention as `Vector.items` entries (`helpers/geometry.py::
    item_points`): `"l"` is a straight line (2 points), `"c"` is a cubic
    bezier (4 points: start, 2 control points, end). Page space."""

    page_index: int
    kind: Literal["l", "c"]
    points: list[tuple[float, float]]


class LabelSet(BaseModel):
    """Every labelled cluster (+ non-text geometry annotation) for one PDF."""

    pdf_path: str
    entries: list[LabelEntry] = Field(default_factory=list)
    geometry_entries: list[GeometryAnnotation] = Field(default_factory=list)


def cluster_signature(cluster: "list[VectorPath]") -> str:
    """A deterministic, order-independent identifier for a cluster -- its
    own member count plus its rounded union bbox. Two clusters with the
    same members (regardless of Python object identity, which doesn't
    survive a fresh pipeline run) produce the same signature."""
    x0, y0, x1, y1 = union_bbox([p.bbox for p in cluster])
    return f"{len(cluster)}:{x0:.1f}:{y0:.1f}:{x1:.1f}:{y1:.1f}"


def path_signature(v: "Vector") -> str:
    """Stable, collision-resistant id for one drawing path (`Vector`),
    matchable across fresh `extract_vectors` runs over the same PDF (same
    PyMuPDF version). Uses absolute page-space geometry plus paint attrs --
    deliberately NOT translation-invariant, unlike
    `Vector_Classification.item_filters.vector_signature`."""
    items = []
    for item in v.items:
        pts = tuple((round(x, 1), round(y, 1)) for x, y in item_points(item))
        items.append((item[0], pts))
    payload = repr((
        v.page_index, v.seqno, v.type, tuple(items),
        tuple(round(c, 1) for c in v.rect),
        v.color, v.fill, v.width, v.dashes,
        bool(v.closePath), bool(v.even_odd), v.layer,
    ))
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def vector_signatures_for(vectors: "list[Vector]") -> list[str]:
    """Sorted, deduped `path_signature()` of every given `Vector` -- shared
    by `native_label.py` (vectors assigned to a native-text line) and
    `vector_label.py` (vectors composing a human-labelled entry)."""
    return sorted({path_signature(v) for v in vectors})


def split_labelset_by_source(labels: LabelSet) -> dict[str, LabelSet]:
    """Split a mixed-source `LabelSet` into `{"auto": ..., "manual": ...}`,
    each key always present (entries may be empty), `pdf_path` preserved --
    so a caller can score auto-derived and human-entered ground truth as
    separate benchmark runs with separate accuracy statistics. Keeps the
    benchmark's own long-standing "auto"/"manual" GT-class vocabulary
    (`Evaluation/Evaluate/metrics.py` etc.) rather than the finer three-way
    `LabelSource` used at labelling time: `source="native"` -> "auto"
    (native-text-derived, no human involved); `source in ("vector",
    "raster")` -> "manual" (human-entered, vector-backed or not)."""
    return {
        "auto": LabelSet(
            pdf_path=labels.pdf_path,
            entries=[e for e in labels.entries if e.source == "native"],
        ),
        "manual": LabelSet(
            pdf_path=labels.pdf_path,
            entries=[e for e in labels.entries if e.source in ("vector", "raster")],
        ),
    }


def save_labels(labels: LabelSet, path: str) -> None:
    Path(path).write_text(labels.model_dump_json(indent=2), encoding="utf-8")


def load_labels(path: str) -> LabelSet:
    return LabelSet.model_validate_json(Path(path).read_text(encoding="utf-8"))
