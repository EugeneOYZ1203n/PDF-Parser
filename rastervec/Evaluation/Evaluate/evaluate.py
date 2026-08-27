"""Evaluate: given a PDF's ground-truth label file (`Evaluation/Labelling/
label_schema.py`) and a completed pipeline run's OCR/drawing outputs,
computes accuracy metrics for the Vector_Classification + OCR pipeline.

Not the pipeline's own metric collector -- callers pass their own
`ClusterOcrResult`/`DrawingVector`/`RotationCheck` lists directly (read off
a real `PipelineContext` after a `Pipeline.run_page()` call, or hand-built),
so this module stays testable against small synthetic inputs (see
`tests/rastervec/Evaluation/Evaluate/test_evaluate.py`) without needing a
real PDF/pipeline run.

Matching a label to a predicted OCR result is done purely spatially --
`helpers.geometry.bbox_iou` between the label's `cluster_bbox` and the
prediction's own resolved bbox, above `iou_threshold` -- via greedy
highest-IoU-first one-to-one matching (a label matches at most one
prediction and vice versa).

Metrics computed (`EvaluationResult`):
  - `characters_found_pct` -- fraction of ground-truth characters belonging
    to a matched label (character-count-weighted, not label-count-weighted).
  - `character_accuracy` / `character_error_rate` -- mean `difflib.
    SequenceMatcher` ratio (and its complement) between each matched pair's
    label text and predicted text. `difflib` (stdlib) is used since no
    string-distance dependency exists in requirements.txt.
  - `rotation_accuracy` -- fraction of matched pairs whose predicted
    rotation (rotation_verify's corrected `rotation_used` when it applied a
    fix, else ocr_compare's own) equals the label's `expected_rotation`.
  - `bbox_accuracy` -- mean IoU across matched pairs.
  - `classification_precision`/`classification_recall` -- text-vs-drawing
    classification: true positives are matched pairs, false negatives are
    unmatched labels (ground-truth text the pipeline never recovered as
    text -- likely ended up in `drawing_vectors`), false positives are
    predicted OCR readings matching no label (over-detection).
  - `drawing_vector_count` -- just `len(drawing_vectors)`.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from rastervec.Evaluation.Labelling.label_schema import LabelEntry, LabelSet
from rastervec.helpers.geometry import bbox_iou
from rastervec.models import ClusterOcrResult, DrawingVector
from rastervec.OCR.Rotation_Correction.rotation_correction import RotationCheck

DEFAULT_IOU_THRESHOLD = 0.3


@dataclass
class MatchedPair:
    label: LabelEntry
    predicted_text: str
    predicted_bbox: tuple[float, float, float, float]
    predicted_rotation: int
    iou: float


@dataclass
class EvaluationResult:
    matched: list[MatchedPair] = field(default_factory=list)
    unmatched_labels: list[LabelEntry] = field(default_factory=list)
    unmatched_predictions: int = 0
    characters_found_pct: float = 0.0
    character_accuracy: float = 0.0
    character_error_rate: float = 0.0
    rotation_accuracy: float = 0.0
    bbox_accuracy: float = 0.0
    classification_precision: float = 0.0
    classification_recall: float = 0.0
    drawing_vector_count: int = 0


def _predictions_from_ocr(
    cluster_ocr_results: list[ClusterOcrResult],
    rotation_checks: list[RotationCheck] | None,
) -> list[tuple[str, tuple[float, float, float, float], int]]:
    """One (text, bbox, rotation_used) per non-blank OCR reading -- using
    rotation_checks' corrected rotation_used for a cluster when a fix was
    applied there, else ocr_compare's own reading (mirrors debug_app.py's
    own rotation_verify-vs-ocr_compare precedence)."""
    corrected_by_cluster_id = {
        id(rc.cluster): rc.resolved for rc in (rotation_checks or []) if rc.applied
    }
    predictions = []
    for result in cluster_ocr_results:
        resolved = result.resolved
        if not resolved.text.strip():
            continue
        corrected = corrected_by_cluster_id.get(id(result.cluster))
        rotation_used = (
            corrected.rotation_used if corrected is not None else resolved.rotation_used
        )
        predictions.append((resolved.text, resolved.bbox, rotation_used))
    return predictions


def evaluate_pipeline(
    labels: LabelSet,
    cluster_ocr_results: list[ClusterOcrResult],
    drawing_vectors: list[DrawingVector],
    rotation_checks: list[RotationCheck] | None = None,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> EvaluationResult:
    predictions = _predictions_from_ocr(cluster_ocr_results, rotation_checks)

    candidates: list[tuple[float, LabelEntry, int, str, tuple[float, float, float, float], int]] = []
    for label in labels.entries:
        for pred_index, (text, bbox, rotation_used) in enumerate(predictions):
            iou = bbox_iou(label.cluster_bbox, bbox)
            if iou >= iou_threshold:
                candidates.append((iou, label, pred_index, text, bbox, rotation_used))
    candidates.sort(key=lambda c: c[0], reverse=True)

    matched_label_ids: set[int] = set()
    matched_pred_indices: set[int] = set()
    matched: list[MatchedPair] = []
    for iou, label, pred_index, text, bbox, rotation_used in candidates:
        if id(label) in matched_label_ids or pred_index in matched_pred_indices:
            continue
        matched_label_ids.add(id(label))
        matched_pred_indices.add(pred_index)
        matched.append(MatchedPair(label, text, bbox, rotation_used, iou))

    unmatched_labels = [l for l in labels.entries if id(l) not in matched_label_ids]
    unmatched_predictions = len(predictions) - len(matched_pred_indices)

    total_gt_chars = sum(len(l.text) for l in labels.entries) or 1
    matched_gt_chars = sum(len(m.label.text) for m in matched)
    characters_found_pct = matched_gt_chars / total_gt_chars

    if matched:
        char_ratios = [
            difflib.SequenceMatcher(None, m.label.text, m.predicted_text).ratio()
            for m in matched
        ]
        character_accuracy = sum(char_ratios) / len(char_ratios)
        rotation_accuracy = sum(
            1 for m in matched if m.predicted_rotation == m.label.expected_rotation
        ) / len(matched)
        bbox_accuracy = sum(m.iou for m in matched) / len(matched)
    else:
        character_accuracy = 0.0
        rotation_accuracy = 0.0
        bbox_accuracy = 0.0
    character_error_rate = 1.0 - character_accuracy

    true_positive = len(matched)
    false_negative = len(unmatched_labels)
    false_positive = unmatched_predictions
    classification_precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0 else 0.0
    )
    classification_recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0 else 0.0
    )

    return EvaluationResult(
        matched=matched,
        unmatched_labels=unmatched_labels,
        unmatched_predictions=unmatched_predictions,
        characters_found_pct=characters_found_pct,
        character_accuracy=character_accuracy,
        character_error_rate=character_error_rate,
        rotation_accuracy=rotation_accuracy,
        bbox_accuracy=bbox_accuracy,
        classification_precision=classification_precision,
        classification_recall=classification_recall,
        drawing_vector_count=len(drawing_vectors),
    )
