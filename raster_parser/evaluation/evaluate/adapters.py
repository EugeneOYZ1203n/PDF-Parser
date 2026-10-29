"""Bridge from labels and pipeline output to the pure `metrics.py` inputs.

`metrics.py` is deliberately pipeline-agnostic (plain `GtRegion` / `Prediction` lists). This
module turns a `LabelSet` and a run's `Text`s into those inputs, and buckets a mixed `LabelSet`
into the text types scored (`metrics.TEXT_TYPES`):

- `native_to_vector` = `LabelEntry.source == "native"` (derived from the PDF's native text)
- `original_vector` = `LabelEntry.source == "vector"` (the hand-labelled vector text)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from raster_parser.evaluation.evaluate.metrics import TEXT_TYPES, GtRegion, Prediction
from raster_parser.evaluation.labelling.label_schema import LabelEntry, LabelSet

if TYPE_CHECKING:
    from raster_parser.models.text import Text

_TYPE_BY_SOURCE = {"native": "native_to_vector", "vector": "original_vector"}


def text_type_of(entry: LabelEntry) -> "str | None":
    """The scored text type of a label entry, or `None` if it belongs to none."""
    return _TYPE_BY_SOURCE.get(entry.source)


def entries_by_text_type(labels: LabelSet) -> "dict[str, list[LabelEntry]]":
    buckets: "dict[str, list[LabelEntry]]" = {t: [] for t in TEXT_TYPES}
    for e in labels.entries:
        text_type = text_type_of(e)
        if text_type is not None:
            buckets[text_type].append(e)
    return buckets


def gt_regions_by_text_type(labels: LabelSet) -> "dict[str, list[GtRegion]]":
    return {
        text_type: [
            GtRegion(
                page_index=e.page_index,
                bbox=tuple(e.bbox),  # type: ignore[arg-type]
                text=e.text,
                expected_rotation=e.expected_rotation,
                text_type=text_type,
            )
            for e in entries
        ]
        for text_type, entries in entries_by_text_type(labels).items()
    }


def gt_regions_from_labelset(labels: LabelSet) -> "list[GtRegion]":
    """Flat `GtRegion` list (every scored type together, `text_type` still set on each)."""
    buckets = gt_regions_by_text_type(labels)
    return [g for t in TEXT_TYPES for g in buckets[t]]


def predictions_from_texts(texts: "list[Text]") -> "list[Prediction]":
    """One `Prediction` per OCR `Text` -- blank readings kept (`ocr_blank=True`) so the candidate
    metrics still see that the region reached OCR. `rotation` is `Text.angle()` snapped to the
    nearest quarter turn, matching `GtRegion.expected_rotation`'s convention."""
    preds: "list[Prediction]" = []
    for t in texts:
        rotation = int(round(t.angle() / 90.0) * 90) % 360
        preds.append(
            Prediction(
                text=t.text,
                bbox=tuple(t.bbox),  # type: ignore[arg-type]
                rotation=rotation,
                reached_ocr=True,
                ocr_blank=not t.text.strip(),
                source_cluster_id=id(t),
            )
        )
    return preds
