"""Bridge from a real pipeline run to the pure `metrics.py`/`vector_metrics.py`
inputs.

`metrics.py`/`vector_metrics.py` are deliberately pipeline-agnostic (plain
`GtRegion`/`Prediction`/`GeometryEntry` lists). This module turns a
`LabelSet` and a `PipelineResult` into those inputs, and buckets a mixed
`LabelSet` into the 4 text types / 3 vector types the rework scores
separately (see the plan this rework was built from):

- `native_to_vector` = `LabelEntry.source == "native"`
- `original_vector` = `LabelEntry.source == "vector"`
- `vector_to_raster` = `LabelEntry.source == "raster"`, `label_id` prefixed
  `"vecsync:"` (raster_label.py's own prefix for a vector-label text copy)
- `original_raster` = `LabelEntry.source == "raster"`, no such prefix
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rastervec.Evaluation.Evaluate.metrics import TEXT_TYPES, Bbox, GtRegion, Prediction
from rastervec.Evaluation.Evaluate.vector_metrics import (
    GeometryEntry,
    geometry_entries_from_annotations,
    geometry_entries_from_vector,
)
from rastervec.Evaluation.Labelling.label_schema import LabelEntry, LabelSet, path_signature
from rastervec.models import Text

if TYPE_CHECKING:
    from rastervec.models import Vector
    from rastervec.pipelines.result import PipelineResult

_VECSYNC_PREFIX = "vecsync:"


def _text_type_of(entry: LabelEntry) -> str:
    if entry.source == "native":
        return "native_to_vector"
    if entry.source == "vector":
        return "original_vector"
    if entry.label_id.startswith(_VECSYNC_PREFIX):
        return "vector_to_raster"
    return "original_raster"


def entries_by_text_type(labels: LabelSet) -> "dict[str, list[LabelEntry]]":
    buckets: "dict[str, list[LabelEntry]]" = {t: [] for t in TEXT_TYPES}
    for e in labels.entries:
        buckets[_text_type_of(e)].append(e)
    return buckets


def gt_regions_by_text_type(labels: LabelSet) -> "dict[str, list[GtRegion]]":
    buckets: "dict[str, list[GtRegion]]" = {t: [] for t in TEXT_TYPES}
    for e in labels.entries:
        text_type = _text_type_of(e)
        buckets[text_type].append(
            GtRegion(
                page_index=e.page_index,
                bbox=tuple(e.cluster_bbox),  # type: ignore[arg-type]
                text=e.text,
                expected_rotation=e.expected_rotation,
                text_type=text_type,
            )
        )
    return buckets


def gt_vector_signatures_by_text_type(labels: LabelSet) -> "dict[str, set[str]]":
    """Union of `vector_signatures` per text type -- always empty for
    `vector_to_raster`/`original_raster` (raster entries never populate
    that field). For `native_to_vector` to be nonempty, the caller must
    first run `enrich_native_vector_signatures` on `labels`."""
    entries = entries_by_text_type(labels)
    return {t: {sig for e in es for sig in e.vector_signatures} for t, es in entries.items()}


def enrich_native_vector_signatures(labels: LabelSet, pdf_path: str, page_index: int) -> None:
    """Mutates `labels` in place: populates `vector_signatures` on every
    `source=="native"` entry of `page_index`, via a temporary
    `convert_page_text_only` render + `native_label.attach_vector_
    signatures`. Needed for category 6's `native_to_vector` funnel row,
    which otherwise has no GT-vector population (native labels leave
    `vector_signatures` empty by default)."""
    import tempfile
    from pathlib import Path

    from rastervec.Evaluation.conversion import convert_page_text_only
    from rastervec.Evaluation.Labelling.native_label import attach_vector_signatures

    if not any(e.source == "native" and e.page_index == page_index for e in labels.entries):
        return
    data = convert_page_text_only(pdf_path, page_index)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "vectors.pdf"
        path.write_bytes(data)
        attach_vector_signatures(labels, page_index, str(path))


def gt_geometry_by_vector_type(labels: LabelSet) -> "dict[str, list]":
    """`{"vector_to_raster": [source=='auto' GeometryAnnotation, ...],
    "original_raster": [source=='manual' GeometryAnnotation, ...]}`."""
    return {
        "vector_to_raster": [g for g in labels.geometry_entries if g.source == "auto"],
        "original_raster": [g for g in labels.geometry_entries if g.source == "manual"],
    }


def gt_original_vectors_from_labelset(
    labels: LabelSet, fresh_vectors: "list[Vector]",
) -> "list[Vector]":
    """Reconstructs the `original_vector` GT `Vector`-level ground truth:
    matches every `path_signature` in every `source=="vector"` entry's
    `vector_signatures` against `path_signature(v)` for `v` in
    `fresh_vectors` -- a fresh `extract_vectors()` call on the SAME
    original, unconverted PDF page, NOT a pipeline run's own
    `PipelineResult.vectors` (already filtered/reclassified into
    text-vs-drawing, so it no longer contains every original path)."""
    sigs: "set[str]" = set()
    for e in labels.entries:
        if e.source == "vector":
            sigs.update(e.vector_signatures)
    if not sigs:
        return []
    return [v for v in fresh_vectors if path_signature(v) in sigs]


def vector_predictions_from_pipeline_result(res: "PipelineResult") -> "list[Vector]":
    """`res.vectors` (final drawing content) + `res.reassigned_text`
    (verbose-only -- every Vector actually assigned to an OCR detection) --
    together the full predicted-vector population for `original_vector`
    scoring. `[]` cleanly on a non-verbose run (no crash)."""
    vectors = list(res.vectors or [])
    vectors.extend(getattr(res, "reassigned_text", None) or [])
    return vectors


def reclass_passed_signatures(res: "PipelineResult") -> "set[str]":
    """`path_signature`s of `PipelineResult.reclassify_result.passed`
    (verbose-only) -- the vectors that survived to the step immediately
    before PaddleOCR detection, for category 6's funnel table. Empty set
    on a non-verbose run."""
    reclass = getattr(res, "reclassify_result", None)
    passed = getattr(reclass, "passed", None) if reclass is not None else None
    if not passed:
        return set()
    return {path_signature(v) for v in passed}


def predictions_from_texts(texts: "list[Text]") -> "list[Prediction]":
    """One `Prediction` per OCR `Text` (`source="ocr"`) -- blank readings
    kept (`ocr_blank=True`) so the classification/candidate metrics can
    still see that the segment reached OCR. `rotation` is `Text.angle()`
    snapped to the nearest quarter turn, matching `GtRegion.expected_
    rotation`'s own convention."""
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


def text_candidate_boxes(texts: "list[Text]") -> "list[Bbox]":
    """Bbox per OCR `Text` (`source="ocr"`) -- every segment occurrence
    that reached OCR, blank or not."""
    return [tuple(t.bbox) for t in texts]  # type: ignore[misc]


@dataclass
class TextEvalInputs:
    gt_by_type: "dict[str, list[GtRegion]]"
    entries_by_type: "dict[str, list[LabelEntry]]"
    predictions: "list[Prediction]"
    gt_vector_signatures_by_type: "dict[str, set[str]]"
    survived_signatures: "set[str]"


def build_text_eval_inputs(labels: LabelSet, res: "PipelineResult") -> TextEvalInputs:
    """The pipeline-derived half of the text-metric inputs (everything
    except the ground truth's own bucketing, which `evaluate_text_metrics`
    itself does not need pre-bucketed by prediction). Requires `res` from a
    `verbose=True` run for `reclassify_result` to be populated."""
    ocr_texts = [t for t in res.texts if t.source == "ocr"]
    return TextEvalInputs(
        gt_by_type=gt_regions_by_text_type(labels),
        entries_by_type=entries_by_text_type(labels),
        predictions=predictions_from_texts(ocr_texts),
        gt_vector_signatures_by_type=gt_vector_signatures_by_text_type(labels),
        survived_signatures=reclass_passed_signatures(res),
    )


@dataclass
class VectorEvalInputs:
    gt_by_type: "dict[str, list[GeometryEntry]]"
    preds_by_type: "dict[str, list[GeometryEntry]]"
    label_counts: "dict[str, int]"


def build_vector_eval_inputs(
    labels: LabelSet, res: "PipelineResult", fresh_vectors: "list[Vector]",
) -> VectorEvalInputs:
    """`fresh_vectors` must come from `extract_vectors` on the SAME
    original, unconverted PDF page `labels` was built against (see
    `gt_original_vectors_from_labelset`'s docstring)."""
    gt_original_vectors = gt_original_vectors_from_labelset(labels, fresh_vectors)
    gt_geometry = gt_geometry_by_vector_type(labels)
    pred_original_vectors = vector_predictions_from_pipeline_result(res)

    gt_by_type: "dict[str, list[GeometryEntry]]" = {
        "original_vector": [e for v in gt_original_vectors for e in geometry_entries_from_vector(v)],
        "vector_to_raster": geometry_entries_from_annotations(gt_geometry["vector_to_raster"]),
        "original_raster": geometry_entries_from_annotations(gt_geometry["original_raster"]),
    }
    preds_by_type: "dict[str, list[GeometryEntry]]" = {
        "original_vector": [e for v in pred_original_vectors for e in geometry_entries_from_vector(v)],
        "vector_to_raster": [],  # no raster-tracing pipeline stage exists
        "original_raster": [],
    }
    label_counts = {
        "original_vector": len({path_signature(v) for v in gt_original_vectors}),
        "vector_to_raster": len(gt_geometry["vector_to_raster"]),
        "original_raster": len(gt_geometry["original_raster"]),
    }
    return VectorEvalInputs(gt_by_type=gt_by_type, preds_by_type=preds_by_type, label_counts=label_counts)
