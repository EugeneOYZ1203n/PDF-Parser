from __future__ import annotations

import difflib

import pytest

from rastervec.Evaluation.Evaluate.evaluate import (
    classify_textbox_grouping,
    evaluate_pipeline,
    normalize_for_cer,
    same_word_bag,
)
from rastervec.Evaluation.Labelling.label_schema import LabelEntry, LabelSet
from rastervec.models import ClusterOcrResult, DrawingVector, TextVectorResult, VectorPath
from rastervec.OCR.Rotation_Correction.rotation_correction import RotationCheck
from rastervec.pipeline import ClusteringStageResult
from rastervec.Vector_Classification.classification import CategoryResult, StepResult


def _make_path(*, bbox=(0, 0, 1, 1)) -> VectorPath:
    return VectorPath(
        seq=0, item_index=0, kind="l", fill_rule="s",
        points=[(bbox[0], bbox[1]), (bbox[2], bbox[3])], bbox=bbox,
        stroke_color=(0, 0, 0), fill_color=None, stroke_opacity=None,
        fill_opacity=None, stroke_width=1.0, dashes=None, closed=False,
        layer=None, page_index=0,
    )


def _make_ocr_result(
    *, cluster: list[VectorPath], text: str, bbox: tuple, rotation_used: int = 0,
) -> ClusterOcrResult:
    resolved = TextVectorResult(
        paths=cluster, text=text, confidence=0.9, bbox=bbox, ocr_bbox=bbox,
        rotation_used=rotation_used, page_index=0,
    )
    return ClusterOcrResult(cluster=cluster, resolved=resolved, ocr_seconds=0.1)


def _make_drawing_vector() -> DrawingVector:
    path = _make_path()
    return DrawingVector(
        paths=[path], bbox=path.bbox, stroke_color=(0, 0, 0), fill_color=None,
        stroke_width=1.0, dashed=False, page_index=0,
    )


def test_evaluate_pipeline_perfect_match():
    label_hello = LabelEntry(
        page_index=0, cluster_bbox=(0, 0, 10, 5), cluster_signature="a",
        text="Hello", source="manual", expected_rotation=0,
    )
    label_world = LabelEntry(
        page_index=0, cluster_bbox=(20, 20, 30, 25), cluster_signature="b",
        text="World", source="manual", expected_rotation=90,
    )
    labels = LabelSet(pdf_path="x.pdf", entries=[label_hello, label_world])

    cluster_hello = [_make_path(bbox=(0, 0, 10, 5))]
    cluster_world = [_make_path(bbox=(20, 20, 30, 25))]
    result_hello = _make_ocr_result(cluster=cluster_hello, text="Hello", bbox=(0, 0, 10, 5), rotation_used=0)
    result_world = _make_ocr_result(cluster=cluster_world, text="World", bbox=(20, 20, 30, 25), rotation_used=0)

    rotation_checks = [
        RotationCheck(
            cluster=cluster_world, text="World", bbox=(20, 20, 30, 25),
            before_rotation=0, after_rotation=90, applied=True,
            error_unrotated=0.5, error_rotated=0.05,
            resolved=TextVectorResult(
                paths=cluster_world, text="World", confidence=0.9,
                bbox=(20, 20, 30, 25), ocr_bbox=(20, 20, 30, 25),
                rotation_used=90, page_index=0,
            ),
        )
    ]

    result = evaluate_pipeline(
        labels, [result_hello, result_world], [_make_drawing_vector()],
        rotation_checks=rotation_checks,
    )

    assert len(result.matched) == 2
    assert result.unmatched_labels == []
    assert result.unmatched_predictions == 0
    assert result.characters_found_pct == pytest.approx(1.0)
    assert result.character_accuracy == pytest.approx(1.0)
    assert result.character_error_rate == pytest.approx(0.0)
    assert result.rotation_accuracy == pytest.approx(1.0)
    assert result.bbox_accuracy == pytest.approx(1.0)
    assert result.classification_precision == pytest.approx(1.0)
    assert result.classification_recall == pytest.approx(1.0)
    assert result.drawing_vector_count == 1


def test_evaluate_pipeline_mismatch_missed_label_and_false_positive():
    label_hello = LabelEntry(
        page_index=0, cluster_bbox=(0, 0, 10, 5), cluster_signature="a",
        text="Hello", source="manual",
    )
    label_missed = LabelEntry(
        page_index=0, cluster_bbox=(50, 50, 60, 55), cluster_signature="c",
        text="Missed", source="manual",
    )
    labels = LabelSet(pdf_path="x.pdf", entries=[label_hello, label_missed])

    cluster_hello = [_make_path(bbox=(0, 0, 10, 5))]
    cluster_extra = [_make_path(bbox=(100, 100, 110, 105))]
    # "Worid" instead of "Hello"'s bbox match text "Helo" (typo/dropped char).
    result_hello = _make_ocr_result(cluster=cluster_hello, text="Helo", bbox=(0, 0, 10, 5), rotation_used=0)
    result_extra = _make_ocr_result(cluster=cluster_extra, text="Extra", bbox=(100, 100, 110, 105), rotation_used=0)

    result = evaluate_pipeline(labels, [result_hello, result_extra], [])

    assert len(result.matched) == 1
    assert result.matched[0].label is label_hello
    assert result.unmatched_labels == [label_missed]
    assert result.unmatched_predictions == 1

    expected_ratio = difflib.SequenceMatcher(None, "Hello", "Helo").ratio()
    assert result.character_accuracy == pytest.approx(expected_ratio)
    assert result.character_error_rate == pytest.approx(1.0 - expected_ratio)

    # characters_found_pct is char-count-weighted: "Hello" (5 chars) matched
    # out of "Hello"+"Missed" (5+6=11) total ground-truth chars.
    assert result.characters_found_pct == pytest.approx(5 / 11)

    # precision = TP / (TP + FP) = 1 / (1 + 1); recall = TP / (TP + FN) = 1 / (1 + 1)
    assert result.classification_precision == pytest.approx(0.5)
    assert result.classification_recall == pytest.approx(0.5)
    assert result.drawing_vector_count == 0


def test_evaluate_pipeline_blank_ocr_reading_excluded():
    label = LabelEntry(
        page_index=0, cluster_bbox=(0, 0, 10, 5), cluster_signature="a",
        text="Hello", source="manual",
    )
    labels = LabelSet(pdf_path="x.pdf", entries=[label])
    cluster = [_make_path(bbox=(0, 0, 10, 5))]
    blank_result = _make_ocr_result(cluster=cluster, text="", bbox=(0, 0, 10, 5))

    result = evaluate_pipeline(labels, [blank_result], [])

    assert result.matched == []
    assert result.unmatched_labels == [label]
    assert result.unmatched_predictions == 0


def test_evaluate_pipeline_attributes_miss_to_classification_step():
    label = LabelEntry(
        page_index=0, cluster_bbox=(0, 0, 10, 5), cluster_signature="a",
        text="Hello", source="manual",
    )
    labels = LabelSet(pdf_path="x.pdf", entries=[label])
    dropped_group = [_make_path(bbox=(0, 0, 10, 5))]

    clustering = {
        ("", ()): ClusteringStageResult(
            steps=[
                StepResult(
                    "Large items", {
                        "kept": CategoryResult([], "kept"),
                        "dropped_oversized": CategoryResult([dropped_group], "dropped"),
                    },
                ),
            ]
        )
    }

    result = evaluate_pipeline(labels, [], [], clustering=clustering)

    assert result.unmatched_labels == [label]
    assert len(result.miss_attributions) == 1
    assert result.miss_attributions[0].label is label
    assert result.miss_attributions[0].reason == "classification:Large items"


def test_evaluate_pipeline_attributes_miss_to_fast_and_ocr_stages():
    label_fast = LabelEntry(
        page_index=0, cluster_bbox=(0, 0, 10, 5), cluster_signature="a",
        text="Fast", source="manual",
    )
    label_ocr = LabelEntry(
        page_index=0, cluster_bbox=(20, 20, 30, 25), cluster_signature="b",
        text="Ocr", source="manual",
    )
    labels = LabelSet(pdf_path="x.pdf", entries=[label_fast, label_ocr])

    fast_dropped = [[_make_path(bbox=(0, 0, 10, 5))]]
    ocr_failed = [[_make_path(bbox=(20, 20, 30, 25))]]

    result = evaluate_pipeline(
        labels, [], [], clustering={}, fast_dropped=fast_dropped, ocr_failed=ocr_failed,
    )

    reasons = {m.label.text: m.reason for m in result.miss_attributions}
    assert reasons == {"Fast": "fast_text_detect", "Ocr": "ocr_blank"}


def test_evaluate_pipeline_attributes_miss_to_not_found():
    label = LabelEntry(
        page_index=0, cluster_bbox=(0, 0, 10, 5), cluster_signature="a",
        text="Ghost", source="manual",
    )
    labels = LabelSet(pdf_path="x.pdf", entries=[label])

    result = evaluate_pipeline(labels, [], [], clustering={})

    assert result.miss_attributions[0].reason == "not_found"


def test_evaluate_pipeline_no_miss_attributions_when_clustering_omitted():
    label = LabelEntry(
        page_index=0, cluster_bbox=(0, 0, 10, 5), cluster_signature="a",
        text="Hello", source="manual",
    )
    labels = LabelSet(pdf_path="x.pdf", entries=[label])

    result = evaluate_pipeline(labels, [], [])

    assert result.unmatched_labels == [label]
    assert result.miss_attributions == []


def test_normalize_for_cer_strips_whitespace_and_uppercases_confusables():
    assert normalize_for_cer("5 mm") == "5MM"
    assert normalize_for_cer("Foo Bar") == "FOOBar"


def test_same_word_bag_matches_regardless_of_order():
    assert same_word_bag("line setback building 5m", "5m building setback line")
    assert same_word_bag("Hello World", "world hello")
    assert not same_word_bag("Hello World", "Hello")
    assert not same_word_bag("", "")


def test_classify_textbox_grouping_correct_split_joint():
    label_a = LabelEntry(
        page_index=0, cluster_bbox=(0, 0, 10, 5), cluster_signature="a",
        text="A", source="manual",
    )
    label_b = LabelEntry(
        page_index=0, cluster_bbox=(20, 20, 30, 25), cluster_signature="b",
        text="B", source="manual",
    )
    label_c = LabelEntry(
        page_index=0, cluster_bbox=(30, 20, 40, 25), cluster_signature="c",
        text="C", source="manual",
    )
    labels = [label_a, label_b, label_c]

    # A -> its own group exactly (correct).
    group_correct = [_make_path(bbox=(0, 0, 10, 5))]
    # B and C are each fully inside (and each >=30% IoU with) the same
    # wider group -- both jointly swallowed into one group.
    group_joint = [_make_path(bbox=(20, 20, 40, 25))]

    result = classify_textbox_grouping(labels, [group_correct, group_joint])

    assert result.correct == 1
    assert result.joint == 2
    assert result.split == 0
    assert result.joint_score == pytest.approx(2 / 3)
    assert result.split_score == pytest.approx(0.0)


def test_classify_textbox_grouping_split():
    label = LabelEntry(
        page_index=0, cluster_bbox=(0, 0, 20, 5), cluster_signature="a",
        text="A", source="manual",
    )
    group_left = [_make_path(bbox=(0, 0, 10, 5))]
    group_right = [_make_path(bbox=(10, 0, 20, 5))]

    result = classify_textbox_grouping([label], [group_left, group_right])

    assert result.split == 1
    assert result.correct == 0
    assert result.split_score == pytest.approx(1.0)


def test_classify_textbox_grouping_no_matches_skipped():
    label = LabelEntry(
        page_index=0, cluster_bbox=(100, 100, 110, 105), cluster_signature="a",
        text="A", source="manual",
    )
    group = [_make_path(bbox=(0, 0, 10, 5))]

    result = classify_textbox_grouping([label], [group])

    assert result.correct == 0
    assert result.split == 0
    assert result.joint == 0
    # No label matched any group -- neither error rate has any signal, so
    # f1 stays at its no-errors-observed default of 1.0.
    assert result.f1 == pytest.approx(1.0)


def test_evaluate_pipeline_populates_textbox_grouping_when_predicted_groups_given():
    label = LabelEntry(
        page_index=0, cluster_bbox=(0, 0, 10, 5), cluster_signature="a",
        text="Hello", source="manual",
    )
    labels = LabelSet(pdf_path="x.pdf", entries=[label])
    group = [_make_path(bbox=(0, 0, 10, 5))]

    result = evaluate_pipeline(labels, [], [], predicted_groups=[group])

    assert result.textbox_grouping is not None
    assert result.textbox_grouping.correct == 1


def test_evaluate_pipeline_textbox_grouping_none_when_omitted():
    label = LabelEntry(
        page_index=0, cluster_bbox=(0, 0, 10, 5), cluster_signature="a",
        text="Hello", source="manual",
    )
    labels = LabelSet(pdf_path="x.pdf", entries=[label])

    result = evaluate_pipeline(labels, [], [])

    assert result.textbox_grouping is None
