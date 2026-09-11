"""label_overlays: GT bbox + per-word correctness overlays."""
from __future__ import annotations

from rastervec.Evaluation.Evaluate.label_overlays import (
    WORD_OK_COLOR,
    WORD_PARTIAL_COLOR,
    WORD_WRONG_COLOR,
    gt_bbox_overlay,
    gt_word_overlay,
)
from rastervec.Evaluation.Evaluate.metrics import (
    MATCH_BOX_COLOR,
    MISSED_GT_BOX_COLOR,
    GtRegion,
    Prediction,
    build_overlap_graph,
)


def _graph(gt, preds):
    return build_overlap_graph(gt, preds)


def test_gt_bbox_overlay_match_and_miss():
    gt = [
        GtRegion(0, (0, 0, 10, 10), "hello", 0),
        GtRegion(0, (100, 100, 110, 110), "world", 0),
    ]
    preds = [Prediction("HELLO", (0, 0, 10, 10), 0)]
    boxes = gt_bbox_overlay(_graph(gt, preds))
    assert boxes[0][1] == MATCH_BOX_COLOR
    assert boxes[1][1] == MISSED_GT_BOX_COLOR


def test_gt_word_overlay_colors():
    gt = [GtRegion(0, (0, 0, 300, 20), "alpha beta gamma", 0)]
    # alpha exact, beta off-by-one, gamma absent
    preds = [Prediction("ALPHA BETO", (0, 0, 300, 20), 0)]
    words = gt_word_overlay(_graph(gt, preds))
    assert [w[0] for w in words] == ["ALPHA", "BETA", "GAMMA"]
    assert words[0][3] == WORD_OK_COLOR
    assert words[1][3] == WORD_PARTIAL_COLOR
    assert words[2][3] == WORD_WRONG_COLOR


def test_gt_word_overlay_no_prediction_all_wrong():
    gt = [GtRegion(0, (0, 0, 100, 20), "foo bar", 0)]
    words = gt_word_overlay(_graph(gt, []))
    assert all(w[3] == WORD_WRONG_COLOR for w in words)


def test_gt_word_overlay_slices_span_bbox():
    gt = [GtRegion(0, (10, 0, 110, 20), "aa bbbb", 0)]
    words = gt_word_overlay(_graph(gt, []))
    assert words[0][1][0] == 10
    assert abs(words[-1][1][2] - 110) < 1e-6
    # wider word gets the wider slice
    assert (words[1][1][2] - words[1][1][0]) > (words[0][1][2] - words[0][1][0])
