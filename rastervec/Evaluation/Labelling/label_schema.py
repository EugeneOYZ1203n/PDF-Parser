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

`GeometryAnnotation` is a separate, non-text kind of ground truth -- line/
curve geometry for a future raster-image line-tracing pipeline stage.
`source="auto"` entries are machine-derived (`raster_label.py`'s
`geometry_annotations_for_vector`, decomposing the original PDF's real
`Vector`s -- carries their real paint attrs); `source="manual"` (the
default) is a human hand-drawn line/curve over an embedded raster image
with no vector backing at all, via the interactive `raster_label` tool.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from rastervec.commons.helpers.geometry import item_points, union_bbox

if TYPE_CHECKING:
    from rastervec.commons.models import Vector

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
    """A non-text ground-truth line/curve shape, for a future raster-image
    line-tracing pipeline stage. Same convention as `Vector.items` entries
    (`helpers/geometry.py::item_points`): `"l"` is a straight line (2
    points), `"c"` is a cubic bezier (4 points: start, 2 control points,
    end). Page space. `color`/`fill`/`width`/`dashes`/`opacity` mirror the
    matching `Vector` fields of the same name -- populated for
    `source="auto"` entries (`geometry_annotations_for_vector`), left `None`
    for a hand-drawn `source="manual"` one unless the labeller sets them."""

    page_index: int
    kind: Literal["l", "c"]
    points: list[tuple[float, float]]
    color: tuple[float, ...] | None = None
    fill: tuple[float, ...] | None = None
    width: float | None = None
    dashes: str | None = None
    opacity: float | None = None
    source: Literal["auto", "manual"] = "manual"


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


def geometry_annotations_for_vector(v: "Vector") -> list[GeometryAnnotation]:
    """Decomposes one `Vector`'s raw `items` into `GeometryAnnotation`
    line/curve entries, `source="auto"`, carrying the `Vector`'s own real
    paint attrs: `"l"` -> one line as-is; `"c"` -> one curve as-is; `"re"`/
    `"qu"` -> their 4 edges as 4 separate lines (a rect/quad has no
    dedicated GeometryAnnotation kind of its own). Used by
    `raster_label.raster_geometry_for_page` -- every `Vector` on the
    original, unconverted page becomes ground truth for a future
    raster-image line-tracing stage, CAD-vector text glyphs included."""
    out: list[GeometryAnnotation] = []
    common = dict(
        page_index=v.page_index, color=v.color, fill=v.fill, width=v.width,
        dashes=v.dashes, opacity=v.stroke_opacity, source="auto",
    )
    for item in v.items:
        kind = item[0]
        pts = item_points(item)
        if kind == "l":
            out.append(GeometryAnnotation(kind="l", points=pts, **common))
        elif kind == "c":
            out.append(GeometryAnnotation(kind="c", points=pts, **common))
        elif kind in ("re", "qu"):
            # "re" gives 2 diagonal corners; expand to all 4. "qu" already
            # gives all 4, in polygon order (ul, ur, lr, ll) per
            # helpers.fitz_geometry.plain_item.
            if kind == "re":
                (x0, y0), (x1, y1) = pts
                corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            else:
                corners = pts
            for p1, p2 in zip(corners, corners[1:] + corners[:1]):
                out.append(GeometryAnnotation(kind="l", points=[p1, p2], **common))
    return out


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


_MASTER_LABEL_FILES = ("native_labels.json", "vector_labels.json", "raster_labels.json")


def load_labels_from_master_folder(folder: "str | Path") -> LabelSet:
    """Merges a `scripts/label/master_label.py` output folder's up-to-3
    label files (`native_labels.json`/`vector_labels.json`/`raster_labels.
    json`, whichever exist -- source="native"/"vector"/"raster"
    respectively) into one `LabelSet`. `pdf_path` is the folder's own
    `original.pdf` (a verbatim copy master_label.py made of the source PDF)
    when present, else the first loaded file's own `pdf_path`. Raises
    `FileNotFoundError` if none of the three files exist -- not a
    master_label folder."""
    folder = Path(folder)
    entries: list[LabelEntry] = []
    geometry_entries: list[GeometryAnnotation] = []
    pdf_path: str | None = None
    found_any = False
    for name in _MASTER_LABEL_FILES:
        path = folder / name
        if not path.is_file():
            continue
        found_any = True
        labels = load_labels(str(path))
        entries.extend(labels.entries)
        geometry_entries.extend(labels.geometry_entries)
        if pdf_path is None:
            pdf_path = labels.pdf_path
    if not found_any:
        raise FileNotFoundError(
            f"{folder} has none of {_MASTER_LABEL_FILES} -- not a master_label.py folder"
        )
    original_pdf = folder / "original.pdf"
    if original_pdf.is_file():
        pdf_path = str(original_pdf.resolve())
    return LabelSet(pdf_path=pdf_path or "", entries=entries, geometry_entries=geometry_entries)
