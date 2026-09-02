"""Evaluate: given a PDF's ground-truth label file (`Evaluation/Labelling/
label_schema.py`) and a completed pipeline run's OCR/drawing outputs,
computes accuracy metrics for the Vector_Classification + OCR pipeline.

Not the pipeline's own metric collector -- callers pass their own
`ClusterOcrResult`/`DrawingVector` lists directly (read off a real
`PipelineContext` after a `Pipeline.run_page()` call, or hand-built), so
this module stays testable against small synthetic inputs (see
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
    rotation (ocr_compare's own `rotation_used`) equals the label's
    `expected_rotation`.
  - `bbox_accuracy` -- mean IoU across matched pairs.
  - `classification_precision`/`classification_recall` -- text-vs-drawing
    classification: true positives are matched pairs, false negatives are
    unmatched labels (ground-truth text the pipeline never recovered as
    text -- likely ended up in `drawing_vectors`), false positives are
    predicted OCR readings matching no label (over-detection).
  - `drawing_vector_count` -- just `len(drawing_vectors)`.
  - `miss_attributions` -- populated only when `clustering` is passed:
    for each unmatched label, which stage lost it (see `_attribute_miss`).

`clustering`/`fast_dropped`/`ocr_failed` (all optional, default `None`,
fully backward-compatible with earlier callers/tests) let
`evaluate_pipeline` build a stage-attributed "loss funnel" instead of just
reporting *that* a label was missed: for each unmatched label,
`_attribute_miss` checks -- in pipeline order -- whether its bbox overlaps
a `role="dropped"` category's groups in `clustering` (attributes to that
classification step, the *earliest* one that matches), else a
`fast_dropped` cluster (`"fast_text_detect"`), else an `ocr_failed`
cluster (`"ocr_blank"`), else `"not_found"` (never appeared in any known
bucket at all -- a Conversion-fidelity gap or extraction issue, not a
classification/OCR decision).

`predicted_groups` (optional, default `None`) additionally populates
`EvaluationResult.textbox_grouping` -- a grouping-*quality* metric
independent of OCR correctness, adapted from archive's
`evaluation/native_vs_ocr/page_textbox_stats.py` correct/split/joint
classification: for each label, count how many of `predicted_groups`' own
bboxes it IoU-matches (>= `iou_threshold`) -- 0 matches is skipped
(unrelated to grouping quality, already covered by `classification_recall`),
>1 matches means that label's content was split across multiple groups
(under-merged), exactly 1 match that some *other* label also uniquely
matches means multiple labels were jointly swallowed into one group
(over-merged), otherwise it's a clean one-to-one match. `split_score`/
`joint_score` are each error's own rate; `f1` is the harmonic mean of their
complements (1.0 when neither kind of error occurs).

`same_word_bag`/`normalize_for_cer` are separate pure helpers (not wired
into `evaluate_pipeline` automatically) callers can use for their own
order-independent text comparisons -- ports of archive's
`_same_word_bag`/`_text_for_compare` from `native_vs_raster_text.py`/
`native_vs_ocr.py`.

`render_evaluation_pdf(result, page_meta)` -- a separate, optional visual
counterpart to one `EvaluationResult`: matched/unmatched-label/
unmatched-prediction bboxes drawn as a real PDF page (green/red/yellow),
via `renderer.render_boxes_pdf`. Not called automatically by
`evaluate_pipeline` (this module stays I/O-free) -- wired in by
`benchmark.py`'s `--eval-pdf-dir`.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rastervec.Evaluation.Labelling.label_schema import LabelEntry, LabelSet, LabelSource
from rastervec.helpers.geometry import bbox_iou, union_bbox
from rastervec.models import ClusterOcrResult, DrawingVector, PageMeta, VectorPath
from rastervec.renderer import render_boxes_pdf

if TYPE_CHECKING:
    from rastervec.pipeline import ClusteringStageResult, GroupKey

DEFAULT_IOU_THRESHOLD = 0.3


@dataclass
class MatchedPair:
    label: LabelEntry
    predicted_text: str
    predicted_bbox: tuple[float, float, float, float]
    predicted_rotation: int
    iou: float


@dataclass
class MissAttribution:
    label: LabelEntry
    reason: str  # "classification:<step label>" | "fast_text_detect" | "ocr_blank" | "not_found"


@dataclass
class TextboxGroupingResult:
    """Grouping-quality metric, independent of OCR correctness -- see
    `classify_textbox_grouping`/this module's docstring."""

    correct: int = 0
    split: int = 0
    joint: int = 0
    split_score: float = 0.0
    joint_score: float = 0.0
    f1: float = 0.0


@dataclass
class EvaluationResult:
    matched: list[MatchedPair] = field(default_factory=list)
    unmatched_labels: list[LabelEntry] = field(default_factory=list)
    unmatched_predictions: int = 0
    # Same unmatched predictions as unmatched_predictions above, but keeping
    # each one's own bbox (not just the count) -- consumed by
    # render_evaluation_pdf below to draw the yellow "over-detection" boxes.
    unmatched_prediction_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    characters_found_pct: float = 0.0
    character_accuracy: float = 0.0
    character_error_rate: float = 0.0
    rotation_accuracy: float = 0.0
    bbox_accuracy: float = 0.0
    classification_precision: float = 0.0
    classification_recall: float = 0.0
    drawing_vector_count: int = 0
    miss_attributions: list[MissAttribution] = field(default_factory=list)
    textbox_grouping: TextboxGroupingResult | None = None


_CONFUSABLE_LETTERS = set("zxcvmskwuop")


def normalize_for_cer(text: str) -> str:
    """Strips whitespace, then uppercases only the OCR-confusable lowercase
    letter set (letters that look nearly identical upper/lower in many
    fonts) -- port of archive's `native_vs_ocr.py::_text_for_compare`, used
    as an optional pre-step before comparing two strings so a case swap on
    an ambiguous letter doesn't count as an error."""
    stripped = "".join(text.split())
    return "".join(c.upper() if c.lower() in _CONFUSABLE_LETTERS else c for c in stripped)


def same_word_bag(a: str, b: str) -> bool:
    """Order-independent match: True if `a` and `b` contain the exact same
    set of whitespace-separated tokens (case-insensitive), regardless of
    order -- e.g. "line set back building 5m" vs "5m building set back
    line" -- port of archive's `native_vs_raster_text.py::_same_word_bag`."""
    tokens_a = tuple(sorted(a.lower().split()))
    tokens_b = tuple(sorted(b.lower().split()))
    return bool(tokens_a) and tokens_a == tokens_b


def split_labelset_by_source(labels: LabelSet) -> dict[LabelSource, LabelSet]:
    """Split a mixed-source `LabelSet` into one `LabelSet` per `source`
    ("auto"/"manual"), each key always present (entries may be empty),
    `pdf_path` preserved -- so a caller can score auto-derived and
    human-entered ground truth as separate `evaluate_pipeline` runs with
    separate accuracy statistics."""
    return {
        source: LabelSet(
            pdf_path=labels.pdf_path,
            entries=[e for e in labels.entries if e.source == source],
        )
        for source in ("auto", "manual")
    }


def classify_textbox_grouping(
    labels: list[LabelEntry],
    predicted_groups: list[list[VectorPath]],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> TextboxGroupingResult:
    """See this module's docstring for the correct/split/joint definitions
    -- port of archive's `evaluation/native_vs_ocr/page_textbox_stats.py`."""
    group_bboxes = [union_bbox([p.bbox for p in group]) for group in predicted_groups if group]

    label_matches: list[list[int]] = [
        [i for i, gb in enumerate(group_bboxes) if bbox_iou(label.cluster_bbox, gb) >= iou_threshold]
        for label in labels
    ]

    group_label_counts: dict[int, int] = {}
    for matches in label_matches:
        if len(matches) == 1:
            group_label_counts[matches[0]] = group_label_counts.get(matches[0], 0) + 1

    correct = split = joint = 0
    for matches in label_matches:
        if not matches:
            continue
        if len(matches) > 1:
            split += 1
        elif group_label_counts.get(matches[0], 0) > 1:
            joint += 1
        else:
            correct += 1

    split_score = split / (split + correct) if (split + correct) else 0.0
    joint_score = joint / (joint + correct) if (joint + correct) else 0.0
    precision_like = 1.0 - split_score
    recall_like = 1.0 - joint_score
    f1 = (
        2 * precision_like * recall_like / (precision_like + recall_like)
        if (precision_like + recall_like) else 0.0
    )

    return TextboxGroupingResult(
        correct=correct, split=split, joint=joint,
        split_score=split_score, joint_score=joint_score, f1=f1,
    )


def _predictions_from_ocr(
    cluster_ocr_results: list[ClusterOcrResult],
) -> list[tuple[str, tuple[float, float, float, float], int]]:
    """One (text, bbox, rotation_used) per non-blank OCR reading, straight
    off ocr_compare's own resolved reading."""
    predictions = []
    for result in cluster_ocr_results:
        resolved = result.resolved
        if not resolved.text.strip():
            continue
        predictions.append((resolved.text, resolved.bbox, resolved.rotation_used))
    return predictions


def _group_matches(
    label: LabelEntry, groups: list[list[VectorPath]], iou_threshold: float,
) -> bool:
    for group in groups:
        if not group:
            continue
        if bbox_iou(label.cluster_bbox, union_bbox([p.bbox for p in group])) >= iou_threshold:
            return True
    return False


def _attribute_miss(
    label: LabelEntry,
    clustering: "dict[GroupKey, ClusteringStageResult] | None",
    fast_dropped: list[list[VectorPath]] | None,
    ocr_failed: list[list[VectorPath]] | None,
    iou_threshold: float,
) -> str:
    """One unmatched label's loss-funnel reason -- see this module's
    docstring. Checked in pipeline order so the *earliest* stage that lost
    this label's region is reported, not a later one that also happens to
    contain a nearby group."""
    for stage_result in (clustering or {}).values():
        for step in stage_result.steps:
            for category in step.categories.values():
                if category.role != "dropped":
                    continue
                if _group_matches(label, category.groups, iou_threshold):
                    return f"classification:{step.label}"

    if _group_matches(label, fast_dropped or [], iou_threshold):
        return "fast_text_detect"

    if _group_matches(label, ocr_failed or [], iou_threshold):
        return "ocr_blank"

    return "not_found"


def evaluate_pipeline(
    labels: LabelSet,
    cluster_ocr_results: list[ClusterOcrResult],
    drawing_vectors: list[DrawingVector],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    clustering: "dict[GroupKey, ClusteringStageResult] | None" = None,
    fast_dropped: list[list[VectorPath]] | None = None,
    ocr_failed: list[list[VectorPath]] | None = None,
    predicted_groups: list[list[VectorPath]] | None = None,
) -> EvaluationResult:
    predictions = _predictions_from_ocr(cluster_ocr_results)

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
    unmatched_prediction_boxes = [
        bbox for i, (_text, bbox, _rotation) in enumerate(predictions) if i not in matched_pred_indices
    ]
    unmatched_predictions = len(unmatched_prediction_boxes)

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

    miss_attributions = (
        [
            MissAttribution(
                label=label,
                reason=_attribute_miss(label, clustering, fast_dropped, ocr_failed, iou_threshold),
            )
            for label in unmatched_labels
        ]
        if clustering is not None
        else []
    )

    textbox_grouping = (
        classify_textbox_grouping(labels.entries, predicted_groups, iou_threshold)
        if predicted_groups is not None
        else None
    )

    return EvaluationResult(
        matched=matched,
        unmatched_labels=unmatched_labels,
        unmatched_predictions=unmatched_predictions,
        unmatched_prediction_boxes=unmatched_prediction_boxes,
        characters_found_pct=characters_found_pct,
        character_accuracy=character_accuracy,
        character_error_rate=character_error_rate,
        rotation_accuracy=rotation_accuracy,
        bbox_accuracy=bbox_accuracy,
        classification_precision=classification_precision,
        classification_recall=classification_recall,
        drawing_vector_count=len(drawing_vectors),
        miss_attributions=miss_attributions,
        textbox_grouping=textbox_grouping,
    )


# render_evaluation_pdf's box colors -- (r, g, b), each channel 0..1, as
# page.draw_rect (via renderer.render_boxes_pdf) expects.
MATCHED_BOX_COLOR = (0.0, 0.6, 0.0)
UNMATCHED_LABEL_BOX_COLOR = (0.85, 0.0, 0.0)
UNMATCHED_PREDICTION_BOX_COLOR = (0.95, 0.75, 0.0)


def render_evaluation_pdf(result: EvaluationResult, page_meta: PageMeta) -> bytes:
    """One PDF page, sized/rotated to `page_meta`, with every bbox one
    `evaluate_pipeline()` result touched drawn as an unfilled colored
    rectangle -- a visual counterpart to classification_precision/_recall
    for a single page:

    - green: each matched pair's label bbox AND its own predicted bbox
      (drawn separately, both green, so any positional drift between the
      two is visible rather than hidden behind a single averaged box).
    - red: each unmatched label -- ground-truth text the pipeline never
      recovered as text.
    - yellow: each unmatched prediction -- an OCR reading matching no
      label (over-detection).
    """
    boxes: list[tuple[tuple[float, float, float, float], tuple[float, float, float]]] = []
    for m in result.matched:
        boxes.append((m.label.cluster_bbox, MATCHED_BOX_COLOR))
        boxes.append((m.predicted_bbox, MATCHED_BOX_COLOR))
    for label in result.unmatched_labels:
        boxes.append((label.cluster_bbox, UNMATCHED_LABEL_BOX_COLOR))
    for bbox in result.unmatched_prediction_boxes:
        boxes.append((bbox, UNMATCHED_PREDICTION_BOX_COLOR))
    return render_boxes_pdf(page_meta, boxes)
