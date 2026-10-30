"""`raster_benchmark`: raster ground truth from a page's vectors, and the one scoring entry point
(`score_vectors`) the report and benchmark call -- swapping the evaluation method replaces it."""

from __future__ import annotations

import pytest

from raster_parser.evaluation.evaluate.raster_benchmark import (
    VECTOR_TYPE,
    raster_ground_truth,
    score_vectors,
)


def _line(vector, seqno, p0, p1, **kw):
    x0, x1 = sorted((p0[0], p1[0]))
    y0, y1 = sorted((p0[1], p1[1]))
    return vector(
        seqno=seqno, type="s", items=[("l", p0, p1)], bbox=(x0, y0, x1, y1), width=1.0, **kw
    )


def test_ground_truth_is_every_painted_vector_as_line_entries(vector):
    a = _line(vector, 1, (10.0, 10.0), (190.0, 10.0))
    box = vector(
        seqno=2,
        type="s",
        items=[("re", (20.0, 20.0, 40.0, 40.0))],
        bbox=(20, 20, 40, 40),
        width=1.0,
    )
    invisible = _line(vector, 3, (0.0, 90.0), (100.0, 90.0), color=None)
    gt = raster_ground_truth([a, box, invisible])
    assert [e.kind for e in gt] == ["l"] * 5
    assert {e.vector_type for e in gt} == {VECTOR_TYPE}
    assert len({e.group for e in gt}) == 2


def test_a_perfect_prediction_scores_full_detection(vector, page_meta):
    gt = [
        _line(vector, 1, (10.0, 10.0), (190.0, 10.0)),
        _line(vector, 2, (10.0, 50.0), (10.0, 90.0)),
    ]
    res = score_vectors(gt, gt, page_meta(width=200, height=100), dpi=72.0)
    assert res.liu_dori.d_v == pytest.approx(1.0)
    assert res.liu_dori.f_v == pytest.approx(0.0)
    assert res.length.ratio.value == pytest.approx(1.0)
    assert res.label_counts == {VECTOR_TYPE: 2}


def test_no_prediction_detects_nothing(vector, page_meta):
    gt = [_line(vector, 1, (10.0, 10.0), (190.0, 10.0))]
    res = score_vectors([], gt, page_meta(width=200, height=100), dpi=72.0, render=False)
    assert res.liu_dori.d_v == pytest.approx(0.0)


def test_text_boxes_split_text_from_drawing_lines(vector, page_meta):
    gt = [
        _line(vector, 1, (10.0, 10.0), (190.0, 10.0)),
        _line(vector, 2, (20.0, 60.0), (40.0, 60.0)),
    ]
    res = score_vectors(
        gt,
        gt,
        page_meta(width=200, height=100),
        dpi=72.0,
        render=False,
        text_boxes=[(15.0, 55.0, 45.0, 65.0)],
    )
    assert res.liu_dori.by_text["text"].d_v.value == pytest.approx(1.0)
    assert res.liu_dori.by_text["non-text"].d_v.value == pytest.approx(1.0)
