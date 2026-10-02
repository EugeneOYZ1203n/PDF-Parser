from __future__ import annotations

import pytest

from rastervec.P3_Vector_Parsing.CollinearVectorClass import line_geometry as lg


def _line(vector, x0, y0, x1, y1, **kw):
    return vector(bbox=(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
                  items=[("l", (x0, y0), (x1, y1))], **kw)


def test_straight_line_fits_polyline_and_rejects_bend(vector):
    v = vector(items=[("l", (0.0, 0.0), (5.0, 5.0)), ("l", (5.0, 5.0), (10.0, 10.0))])
    fit = lg.straight_line(v, 0.25)
    assert fit is not None
    assert fit.angle == pytest.approx(45.0)
    assert fit.length == pytest.approx(10.0 * 2 ** 0.5)
    bent = vector(items=[("l", (0.0, 0.0), (5.0, 0.0)), ("l", (5.0, 0.0), (5.0, 5.0))])
    assert lg.straight_line(bent, 0.25) is None
    assert lg.straight_line(vector(kind="re", bbox=(0, 0, 5, 5)), 0.25) is None


def test_group_collinear_splits_parallel_offsets(vector):
    straight, _ = lg.split_straight(
        [_line(vector, x, 0, x + 2, 0) for x in range(0, 20, 4)]
        + [_line(vector, x, 5, x + 2, 5) for x in range(0, 20, 4)],
        0.25,
    )
    groups = lg.group_collinear(straight, 1.0, 0.5)
    assert sorted(len(g) for g in groups) == [5, 5]


def test_group_parallel_wraps_around_180(vector):
    straight, _ = lg.split_straight([
        _line(vector, 0, 0, 10, 0.05),     # ~0.3 deg
        _line(vector, 0, 0.05, 10, 0),     # ~179.7 deg
    ], 0.25)
    assert len(lg.group_parallel(straight, 1.0)) == 1


def test_dedupe_angles_merges_close_and_wrapping_angles():
    out = lg.dedupe_angles([0.2, 179.9, 90.0, 90.5, 45.0], 1.0)
    assert len(out) == 3
    assert any(lg.axial_distance(a, 0.05) < 1e-6 for a in out)  # 0.2 and 179.9 merged
    assert any(abs(a - 90.25) < 1e-6 for a in out)
    assert any(abs(a - 45.0) < 1e-6 for a in out)


def test_crossing_segment_counts_counts_distinct_foreign_segments(vector):
    hatch = [_line(vector, x, -5, x, 5) for x in range(1, 6)]
    bar = _line(vector, 0, 0, 10, 0)
    touching = _line(vector, 10, 0, 10, 5)  # endpoint on the bar's end -- not a crossing
    counts = lg.crossing_segment_counts([bar, touching, *hatch], eps=0.01, curve_steps=8)
    assert counts[0] == 5
    assert counts[1] == 0
    assert counts[2:] == [1] * 5
