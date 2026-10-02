from __future__ import annotations

import pytest

from rastervec.commons.models import Page
from rastervec.P3_Vector_Parsing.CollinearVectorClass import classify_vectors as cv
from rastervec.P3_Vector_Parsing.CollinearVectorClass.config import (
    COLLINEAR_DRAWING_MIN_COUNT,
    MAX_CROSSING_SEGMENTS,
)


def _line(vector, x0, y0, x1, y1, **kw):
    return vector(bbox=(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
                  items=[("l", (x0, y0), (x1, y1))], **kw)


def _dashes(vector, n, y=0.0, length=2.0, gap=2.0, seq0=0):
    return [
        _line(vector, i * (length + gap), y, i * (length + gap) + length, y, seqno=seq0 + i)
        for i in range(n)
    ]


def test_collinear_drawing_drops_long_regular_dashed_line(vector):
    dashes = _dashes(vector, COLLINEAR_DRAWING_MIN_COUNT + 1)
    other = _line(vector, 0, 50, 3, 53)
    kept, drawing, angles = cv.collinear_drawing(dashes + [other])
    assert kept == [other]
    assert len(drawing) == 1 and len(drawing[0]) == len(dashes)
    assert sorted(round(a) for a in angles) == [0, 45]


def test_collinear_drawing_keeps_group_at_count_threshold(vector):
    dashes = _dashes(vector, COLLINEAR_DRAWING_MIN_COUNT)  # needs MORE than the threshold
    kept, drawing, _ = cv.collinear_drawing(dashes)
    assert drawing == [] and len(kept) == len(dashes)


def test_collinear_drawing_keeps_irregular_lengths(vector):
    lines = [
        _line(vector, x, 0, x + (1 if i % 2 else 20), 0)
        for i, x in enumerate(range(0, 30 * (COLLINEAR_DRAWING_MIN_COUNT + 1), 30))
    ]
    _, drawing, _ = cv.collinear_drawing(lines)
    assert drawing == []  # std of 1/20 alternating lengths is 9.5 pt > 5


def test_length_outliers_pooled_over_parallel_groups(vector):
    strokes = [_line(vector, x, 0, x, 5) for x in range(10)]  # 10 vertical, length 5
    strokes += [_line(vector, 0, y, 5, y) for y in range(10, 20)]  # 10 horizontal, length 5
    long_one = _line(vector, 20, 0, 20, 60)  # vertical, length 60
    lone = _line(vector, 30, 0, 90, 30)  # its own angle -> no parallel group, never judged
    removed = cv.length_outliers(strokes + [long_one, lone])
    assert removed == [long_one]


def test_length_outliers_nothing_when_uniform(vector):
    assert cv.length_outliers([_line(vector, x, 0, x, 5) for x in range(5)]) == []


def test_heavily_crossed_removes_vector_over_threshold(vector):
    hatch = [_line(vector, x + 0.5, -5, x + 0.5, 5) for x in range(MAX_CROSSING_SEGMENTS + 1)]
    bar = _line(vector, 0, 0, MAX_CROSSING_SEGMENTS + 2, 0)
    assert cv.heavily_crossed([bar, *hatch]) == [bar]
    assert cv.heavily_crossed([bar, *hatch[:-1]]) == []  # exactly the threshold survives


def test_classify_vectors_end_to_end(page_meta, vector):
    page = Page(doc_path="synthetic.pdf", meta=page_meta(width=1000.0, height=1000.0), fitz_page=None)
    dashes = _dashes(vector, COLLINEAR_DRAWING_MIN_COUNT + 1, y=500.0, seq0=100)
    glyph = [_line(vector, 10, 10, 10, 15, seqno=1), _line(vector, 12, 10, 12, 15, seqno=2)]
    res = cv.classify_vectors(dashes + glyph, page)

    assert [s.label for s in next(iter(res.clustering.values())).steps] == list(cv.STEP_LABELS)
    assert {id(v) for v in res.drawing_vectors} == {id(v) for v in dashes}
    kept = [v for cluster in res.text_clusters for g in cluster for v in g]
    assert {id(v) for v in kept} == {id(v) for v in glyph}
    assert res.global_angles == pytest.approx([0.0, 90.0])
