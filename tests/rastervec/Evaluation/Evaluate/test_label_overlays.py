"""label_overlays: GT bbox + per-word correctness overlays."""
from __future__ import annotations

from rastervec.Evaluation.Evaluate.label_overlays import (
    WORD_OK_COLOR,
    WORD_PARTIAL_COLOR,
    WORD_WRONG_COLOR,
    gt_bbox_overlay,
    gt_word_bboxes,
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


def test_gt_word_bboxes_follow_reading_direction():
    # "AB CD": two equal-length words; the first must sit where reading starts.
    def first(rot):
        return gt_word_bboxes(GtRegion(0, (0, 0, 40, 40), "ab cd", rot))[0]

    assert first(0) == ("AB", (0, 0, 20.0, 40))      # left half
    assert first(180) == ("AB", (20.0, 0, 40, 40))   # right half (right-to-left)
    assert first(90) == ("AB", (0, 0, 40, 20.0))     # top half (downward)
    assert first(270) == ("AB", (0, 20.0, 40, 40))   # bottom half (upward)


def test_extra_predictions_definitions():
    from types import SimpleNamespace as NS

    from rastervec.Evaluation.Evaluate.label_overlays import extra_predictions

    gt = [(0, 0, 10, 10), (50, 50, 60, 60)]
    manual = [(50, 50, 60, 60)]
    texts = [
        NS(bbox=(1, 1, 5, 5), source="ocr"),       # inside GT -> not extra
        NS(bbox=(9, 9, 20, 20), source="ocr"),     # touches GT -> not extra
        NS(bbox=(30, 30, 40, 40), source="ocr"),   # outside -> extra
        NS(bbox=(30, 30, 40, 40), source="native"),  # not OCR -> ignored
    ]
    routed = [
        NS(bbox=(2, 2, 4, 4)),        # covered
        NS(bbox=(8, 8, 18, 18)),      # 4% covered -> extra
        NS(bbox=(2, 5, 8, 5)),        # zero-area line, centre inside -> covered
        NS(bbox=(20, 5, 30, 5)),      # zero-area line outside -> extra
    ]
    drawing = [NS(bbox=(51, 51, 55, 55)), NS(bbox=(1, 1, 2, 2))]
    extra_t, extra_v, missed_v = extra_predictions(texts, routed, drawing, gt, manual)
    assert extra_t == [texts[2]]
    assert extra_v == [routed[1], routed[3]]
    assert missed_v == [drawing[0]]  # only manual regions count as missed
