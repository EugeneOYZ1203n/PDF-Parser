from __future__ import annotations

import pytest

from rastervec.P3_Vector_Parsing.VectorClassification import line_geometry as lg


def _line(vector, p0, p1, seqno=0):
    return vector(kind="l", items=[("l", p0, p1)], bbox=(
        min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1]),
    ), seqno=seqno)


def test_straight_line_fit_and_non_straight_rejected(vector):
    fit = lg.straight_line(_line(vector, (0.0, 0.0), (10.0, 0.0)), 0.25)
    assert fit is not None and fit.angle == pytest.approx(0.0) and fit.length == pytest.approx(10.0)
    bent = vector(kind="l", items=[("l", (0.0, 0.0), (10.0, 0.0)), ("l", (10.0, 0.0), (10.0, 10.0))])
    assert lg.straight_line(bent, 0.25) is None
    curve = vector(kind="c", bbox=(0.0, 0.0, 10.0, 10.0))
    assert lg.straight_line(curve, 0.25) is None


def test_collinear_splits_parallel_lines_by_offset(vector):
    a = _line(vector, (0.0, 0.0), (10.0, 0.0), 1)
    b = _line(vector, (20.0, 0.0), (30.0, 0.0), 2)  # same infinite line as a
    c = _line(vector, (0.0, 5.0), (10.0, 5.0), 3)   # parallel, 5pt off
    straight, _ = lg.split_straight([a, b, c], 0.25)
    parallel = lg.group_parallel(straight, 2.0)
    collinear = lg.group_collinear(straight, 2.0, 1.0)
    assert [sorted(v.seqno for v, _ in g) for g in parallel] == [[1, 2, 3]]
    assert sorted(sorted(v.seqno for v, _ in g) for g in collinear) == [[1, 2], [3]]


def test_parallel_grouping_wraps_around_180(vector):
    a = _line(vector, (0.0, 0.0), (100.0, -0.3), 1)   # ~179.8 deg folded
    b = _line(vector, (0.0, 10.0), (100.0, 10.2), 2)  # ~0.1 deg
    straight, _ = lg.split_straight([a, b], 0.25)
    assert len(lg.group_parallel(straight, 2.0)) == 1


def test_golden_hues_are_distinct():
    hues = lg.golden_hues(5)
    assert len(set(round(h, 6) for h in hues)) == 5 and all(0.0 <= h < 1.0 for h in hues)
