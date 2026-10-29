"""Label files: hand-labelled **original vector text** -- text drawn as vector paths in the source
PDF, typed in by a human.

That is the only thing a label file holds. Every other kind of ground truth is derived when the
benchmark runs: native text from the PDF itself (`native_label.native_label_pdf`, entries with
`source="native"` that are never saved), and raster ground truth from these labels and the page's
vectors.

One `LabelEntry` per labelled word or line. It mirrors a `Text` (`from_text` / `to_text`):
`bbox` is the text's axis-aligned box in unrotated page space, `expected_rotation` its reading
angle (`Text.angle()`, degrees, 0..359). It is tied to the `Vector`s that draw it by
`vector_signatures` (`path_signature` of each), so a fresh extraction of the same PDF can match a
label back to its exact paths. `label_id` is the entry's stable identity (edits replace it in
place).

Version 1 files (with `cluster_bbox`, `cluster_signature`, `baseline_id`, raster / CAD-font
entries, `geometry_entries` and `baselines`) still load: the box is read from `cluster_bbox`,
the unused fields are ignored and raster / CAD-font entries are dropped with a warning.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from raster_parser.commons.helpers.geometry import compute_origin, item_points
from raster_parser.commons.logging_setup import get_logger

if TYPE_CHECKING:
    from raster_parser.models.text import Text
    from raster_parser.models.vector import Vector

_LOG = get_logger("label_schema")

SCHEMA_VERSION = 2

# "vector": hand-labelled original vector text (what label files hold). "native": derived from
# the PDF's native text at benchmark time (`native_label`), never saved.
LabelSource = Literal["native", "vector"]


class LabelEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    page_index: int
    label_id: str
    text: str
    bbox: tuple[float, float, float, float] = Field(
        validation_alias=AliasChoices("bbox", "cluster_bbox")
    )
    expected_rotation: int = 0
    vector_signatures: list[str] = Field(default_factory=list)
    source: LabelSource = "vector"

    @classmethod
    def from_text(
        cls,
        text: "Text",
        vectors: "list[Vector]" = (),
        *,
        label_id: str | None = None,
        source: LabelSource = "vector",
    ) -> "LabelEntry":
        """A label for `text`, drawn by `vectors`."""
        return cls(
            page_index=text.page_index,
            label_id=label_id or uuid.uuid4().hex,
            text=text.text,
            bbox=tuple(float(c) for c in text.bbox),
            expected_rotation=int(round(text.angle())) % 360,
            vector_signatures=vector_signatures_for(list(vectors)),
            source=source,
        )

    def to_text(self) -> "Text":
        """The `Text` this label describes (no font, colour or confidence)."""
        from raster_parser.models.text import Text

        rad = math.radians(self.expected_rotation)
        direction = (round(math.cos(rad), 12) + 0.0, round(math.sin(rad), 12) + 0.0)
        return Text(
            text=self.text,
            bbox=self.bbox,
            direction=direction,
            origin=compute_origin(self.bbox, direction),
            font="",
            font_size=0.0,
            color=None,
            flags=0,
            ascender=None,
            descender=None,
            wmode=0,
            block_no=0,
            line_no=0,
            word_no=0,
            page_index=self.page_index,
            seqno=0,
            source="label",
            orientation_source="dto",
        )


class LabelSet(BaseModel):
    """Every label of one PDF."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = SCHEMA_VERSION
    pdf_path: str
    entries: list[LabelEntry] = Field(default_factory=list)


def path_signature(v: "Vector") -> str:
    """Stable, collision-resistant id for one drawing path (`Vector`), matchable across fresh
    extractions of the same PDF (same PyMuPDF version). Absolute page-space geometry plus paint
    attributes -- deliberately not translation-invariant."""
    items = []
    for item in v.items:
        pts = tuple((round(x, 1), round(y, 1)) for x, y in item_points(item))
        items.append((item[0], pts))
    payload = repr(
        (
            v.page_index,
            v.seqno,
            v.type,
            tuple(items),
            tuple(round(c, 1) for c in v.rect),
            v.color,
            v.fill,
            v.width,
            v.dashes,
            bool(v.closePath),
            bool(v.even_odd),
            v.layer,
        )
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def vector_signatures_for(vectors: "list[Vector]") -> list[str]:
    """Sorted, deduped `path_signature()` of every given `Vector`."""
    return sorted({path_signature(v) for v in vectors})


def save_labels(labels: LabelSet, path: str) -> None:
    Path(path).write_text(labels.model_dump_json(indent=2), encoding="utf-8")


def load_labels(path: str) -> LabelSet:
    """A label file of either schema version (see the module docstring)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = raw.get("entries", [])
    kept = [e for e in entries if e.get("source", "vector") in ("native", "vector")]
    if len(kept) != len(entries):
        _LOG.warning(
            "%s: dropped %d raster / CAD-font label(s); label files hold vector text only",
            path,
            len(entries) - len(kept),
        )
    raw = {**raw, "entries": kept, "schema_version": SCHEMA_VERSION}
    return LabelSet.model_validate(raw)
