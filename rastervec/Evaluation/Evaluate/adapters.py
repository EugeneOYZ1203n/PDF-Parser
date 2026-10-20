"""Bridge from a real pipeline run to the pure `metrics.py`/`vector_metrics.py`
inputs.

`metrics.py`/`vector_metrics.py` are deliberately pipeline-agnostic (plain
`GtRegion`/`Prediction`/`GeometryEntry` lists). This module turns a
`LabelSet` and a `PipelineResult` into those inputs, and buckets a mixed
`LabelSet` into the 5 text types / 2 vector types the rework scores
separately (see the plan this rework was built from):

- `native_to_vector` = `LabelEntry.source == "native"`
- `original_vector` = `LabelEntry.source == "vector"`
- `native_to_raster` = `LabelEntry.source == "raster"`, `label_id` prefixed
  `"natsync:"` (raster_label.py's own prefix for a native-label text copy --
  the same native text region, scored against a rasterised-PDF pipeline run
  instead of the vectorised one)
- `vector_to_raster` = `LabelEntry.source == "raster"`, `label_id` prefixed
  `"vecsync:"` (raster_label.py's own prefix for a vector-label text copy)
- `original_raster` = `LabelEntry.source == "raster"`, neither prefix

Checked in that order -- `natsync:`/`vecsync:` must be checked before the
plain `source == "raster"` fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rastervec.Evaluation.Evaluate.metrics import TEXT_TYPES, Bbox, GtRegion, Prediction
from rastervec.Evaluation.Evaluate.vector_metrics import (
    GeometryEntry,
    geometry_entries_from_annotations,
    geometry_entries_from_vectors,
)
from rastervec.Evaluation.Labelling.label_schema import LabelEntry, LabelSet, path_signature
from rastervec.commons.models import Text

if TYPE_CHECKING:
    from rastervec.pipelines.result import PipelineResult

_VECSYNC_PREFIX = "vecsync:"
_NATSYNC_PREFIX = "natsync:"


def _text_type_of(entry: LabelEntry) -> str:
    if entry.label_id.startswith(_NATSYNC_PREFIX):
        return "native_to_raster"
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


def gt_regions_from_labelset(labels: LabelSet) -> "list[GtRegion]":
    """Flat `GtRegion` list (every text type together, `text_type` still set
    on each) -- kept for callers that score one shared graph across every
    label source at once (`scripts/generate_pipeline_report.py`'s
    single-combined-run benchmark mode), as opposed to
    `gt_regions_by_text_type`'s 4-way bucketing."""
    buckets = gt_regions_by_text_type(labels)
    return [g for t in TEXT_TYPES for g in buckets[t]]


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
    preds: "list[GeometryEntry]"
    label_counts: "dict[str, int]"
    # Every text label's page-space bbox (all text types) -- the Liu-Dori
    # text / non-text split.
    text_boxes: "list[Bbox]"


def build_vector_eval_inputs(labels: LabelSet, raster_vectors: "list | None" = None) -> VectorEvalInputs:
    """GT = `vector_to_raster`/`original_raster` geometry annotations (each
    entry tagged with its type); predictions = every `Vector` of the
    rasterised-PDF run (`PageDump.raster_vectors` -- P2's output), or none
    when there was no such run. `original_vector` is never scored here --
    it stays a `metrics.TEXT_TYPES` text-provenance type."""
    gt_geometry = gt_geometry_by_vector_type(labels)
    gt_by_type: "dict[str, list[GeometryEntry]]" = {
        t: geometry_entries_from_annotations(gt_geometry[t], vector_type=t)
        for t in ("vector_to_raster", "original_raster")
    }
    label_counts = {t: len(gt_geometry[t]) for t in ("vector_to_raster", "original_raster")}
    return VectorEvalInputs(
        gt_by_type=gt_by_type,
        preds=geometry_entries_from_vectors(raster_vectors or []),
        label_counts=label_counts,
        text_boxes=[tuple(e.cluster_bbox) for e in labels.entries],  # type: ignore[misc]
    )


def raster_dpi_for_page(pdf_path: str, page_index: int) -> "float | None":
    """Resolution of a rasterised PDF page (`master_label.py`'s
    `rasterised.pdf`, one full-page embedded image): the largest embedded
    image's pixels per inch, area-based so a 90-degree placement reads the
    same. `None` when the file/page has no image or can't be opened."""
    import fitz

    try:
        doc = fitz.open(pdf_path)
    except Exception:  # noqa: BLE001
        return None
    try:
        if not 0 <= page_index < doc.page_count:
            return None
        best = None
        for info in doc[page_index].get_image_info():
            x0, y0, x1, y1 = info["bbox"]
            area_pt = abs(x1 - x0) * abs(y1 - y0)
            area_px = float(info.get("width", 0)) * float(info.get("height", 0))
            if area_pt <= 0 or area_px <= 0:
                continue
            if best is None or area_pt > best[0]:
                best = (area_pt, 72.0 * (area_px / area_pt) ** 0.5)
        return best[1] if best else None
    finally:
        doc.close()
