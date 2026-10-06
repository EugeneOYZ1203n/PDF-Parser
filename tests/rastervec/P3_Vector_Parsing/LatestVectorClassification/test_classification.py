from __future__ import annotations

import math

import pytest

from rastervec.commons.models import Page
from rastervec.P3_Vector_Parsing.LatestVectorClassification import classify_vectors as cv
from rastervec.P3_Vector_Parsing.LatestVectorClassification.config import (
    COLLINEAR_DRAWING_MIN_COUNT,
    MIN_CROSSINGS,
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
    kept, drawing, _ = cv.collinear_drawing(dashes + [other])
    assert kept == [other]
    assert len(drawing) == 1 and len(drawing[0]) == len(dashes)


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


def _curve(vector, p0, p1, p2, p3):
    xs, ys = [p[0] for p in (p0, p1, p2, p3)], [p[1] for p in (p0, p1, p2, p3)]
    return vector(kind="c", bbox=(min(xs), min(ys), max(xs), max(ys)), items=[("c", p0, p1, p2, p3)])


def test_crossed_grid_drops_bar_crossed_four_times(vector):
    hatch = [_line(vector, x + 0.5, -5, x + 0.5, 5) for x in range(MIN_CROSSINGS)]
    bar = _line(vector, 0, 0, MIN_CROSSINGS + 2, 0)
    dropped, flagged_kept = cv.crossed_grid([bar, *hatch])
    assert dropped == [bar] and flagged_kept == []


def test_crossed_grid_keeps_bar_crossed_three_times(vector):
    hatch = [_line(vector, x + 0.5, -5, x + 0.5, 5) for x in range(MIN_CROSSINGS - 1)]
    bar = _line(vector, 0, 0, MIN_CROSSINGS + 2, 0)
    assert cv.crossed_grid([bar, *hatch]) == ([], [])


def test_crossed_grid_only_considers_line_only_candidates(vector):
    # A filled rect crossed by 4 lines is never a candidate (not "l"-only).
    rect = vector(kind="re", type="f", bbox=(0, -1, 10, 1), items=[("re", (0, -1, 10, 1))])
    hatch = [_line(vector, x + 0.5, -5, x + 0.5, 5) for x in range(1, 5)]
    dropped, _ = cv.crossed_grid([rect, *hatch])
    assert rect not in dropped


def test_crossed_grid_counts_curve_crossings(vector):
    bar = _line(vector, 0, 0, 10, 0)
    # An S-curve crossing the bar three times, plus one plain crosser.
    s_curve = _curve(vector, (1, -5), (12, 15), (-2, -15), (9, 5))
    crosser = _line(vector, 9.5, -5, 9.5, 5)
    dropped, _ = cv.crossed_grid([bar, s_curve, crosser])
    assert dropped == [bar]


def test_crossed_grid_drops_only_the_dominant_grid(vector):
    # Grid A: two long horizontal bars (axis grid); grid B: one shorter 30-deg
    # bar. Each is crossed >= 4 times. A holds > 50% of the flagged length.
    def crossers(x0, x1, y):
        return [_line(vector, x, y - 30, x + 0.01, y + 30) for x in range(x0 + 1, x1, 2)][:5]

    a1 = _line(vector, 0, 0, 20, 0)
    a2 = _line(vector, 0, 100, 20, 100)
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    b = _line(vector, 200, 0, 200 + 10 * c, 10 * s)
    b_cross = [_line(vector, 200 + t * c - 4 * s, t * s + 4 * c, 200 + t * c + 4 * s, t * s - 4 * c)
               for t in (1, 3, 5, 7, 9)]
    vectors = [a1, a2, b, *crossers(0, 20, 0), *crossers(0, 20, 100), *b_cross]
    dropped, flagged_kept = cv.crossed_grid(vectors)
    assert {id(v) for v in dropped} == {id(a1), id(a2)}
    assert flagged_kept == [b]


def test_crossed_grid_even_split_drops_nothing(vector):
    a = _line(vector, 0, 0, 10, 0)
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    b = _line(vector, 200, 0, 200 + 10 * c, 10 * s)
    a_cross = [_line(vector, x + 0.5, -5, x + 0.5, 5) for x in range(1, 9, 2)]
    b_cross = [_line(vector, 200 + t * c - 4 * s, t * s + 4 * c, 200 + t * c + 4 * s, t * s - 4 * c)
               for t in (1, 3, 5, 7)]
    dropped, flagged_kept = cv.crossed_grid([a, b, *a_cross, *b_cross])
    assert dropped == [] and {id(v) for v in flagged_kept} == {id(a), id(b)}


def test_crossed_grid_keeps_mixed_direction_candidate(vector):
    # One polyline: a horizontal leg and a 30-deg leg -- not one grid.
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    poly = vector(bbox=(0, 0, 10 + 10 * c, 10 * s),
                  items=[("l", (0, 0), (10, 0)), ("l", (10, 0), (10 + 10 * c, 10 * s))])
    hatch = [_line(vector, x + 0.5, -5, x + 0.5, 5) for x in range(1, 9, 2)]
    dropped, flagged_kept = cv.crossed_grid([poly, *hatch])
    assert dropped == [] and flagged_kept == [poly]


def test_classify_vectors_end_to_end(page_meta, vector):
    page = Page(doc_path="synthetic.pdf", meta=page_meta(width=1000.0, height=1000.0), fitz_page=None)
    dashes = _dashes(vector, COLLINEAR_DRAWING_MIN_COUNT + 1, y=500.0, seq0=100)
    glyph = [_line(vector, 10, 10, 10, 15, seqno=1), _line(vector, 12, 10, 12, 15, seqno=2)]
    res = cv.classify_vectors(dashes + glyph, page)

    assert [s.label for s in next(iter(res.clustering.values())).steps] == list(cv.STEP_LABELS)
    assert {id(v) for v in res.drawing_vectors} == {id(v) for v in dashes}
    kept = [v for cluster in res.text_clusters for g in cluster for v in g]
    assert {id(v) for v in kept} == {id(v) for v in glyph}
    assert res.global_angles == pytest.approx([0.0])  # the dashed line's collinear group


def test_global_angles_from_collinear_groups_of_two_or_more(vector, page_meta):
    c, s = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
    pair = [_line(vector, t * c, t * s, (t + 5) * c, (t + 5) * s, seqno=i) for i, t in enumerate((0.0, 20.0))]
    lone = _line(vector, 100, 100, 100 + 5 * math.cos(math.radians(60)), 100 + 5 * math.sin(math.radians(60)), seqno=9)
    page = Page(doc_path="synthetic.pdf", meta=page_meta(width=200.0, height=200.0), fitz_page=None)
    res = cv.classify_vectors(pair + [lone], page)
    assert res.global_angles == pytest.approx([30.0], abs=0.01)
