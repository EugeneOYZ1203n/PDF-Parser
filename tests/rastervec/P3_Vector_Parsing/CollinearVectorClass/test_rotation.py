from __future__ import annotations

import pytest

from rastervec.P3_Vector_Parsing.CollinearVectorClass import rotation as rot
from rastervec.P3_Vector_Parsing.CollinearVectorClass.rotation import ParallelGroup as PG


def _no_hough():
    raise AssertionError("Hough must not run on this branch")


def test_parallel_groups_ignores_singletons(vector):
    lines = [vector(bbox=(x, 0, x, 5)) for x in (0, 2, 4)] + [vector(bbox=(0, 10, 7, 17))]
    groups = rot.parallel_groups(lines)
    assert len(groups) == 1
    assert groups[0].angle == pytest.approx(90.0)
    assert groups[0].count == 3 and groups[0].total_length == pytest.approx(15.0)


@pytest.mark.parametrize("groups, expected, rule", [
    ([], 0.0, "cluster no parallel group"),
    ([PG(30.0, 5.0, 2)], 30.0, "cluster 1 parallel group"),
    ([PG(30.0, 5.0, 2), PG(100.0, 50.0, 2)], 100.0, "cluster longest parallel group"),
])
def test_pre_detect_direction(groups, expected, rule):
    d = rot.pre_detect_direction(groups)
    assert (d.angle, d.rule) == (expected, rule)


def test_final_direction_single_component_never_runs_hough():
    assert rot.final_direction(1, [], _no_hough, [45.0]).angle == 0.0
    assert rot.final_direction(1, [PG(20.0, 1.0, 2)], _no_hough, []).angle == 20.0
    d = rot.final_direction(1, [PG(20.0, 1.0, 2), PG(80.0, 9.0, 2)], _no_hough, [])
    assert d.angle == 80.0 and d.rule == "1cc longest parallel group"


def test_final_direction_multi_component_one_group_skips_hough():
    d = rot.final_direction(3, [PG(20.0, 1.0, 2)], _no_hough, [])
    assert d.angle == 20.0 and d.hough_angle is None


def test_final_direction_multi_component_no_group_snaps_hough_to_global():
    d = rot.final_direction(2, [], lambda: 178.0, [0.0, 45.0, 90.0])
    assert d.angle == 0.0 and d.hough_angle == 178.0  # axial: 178 is 2 deg from 0
    assert d.rule == "multi-cc hough snapped to global angle"


def test_final_direction_multi_component_groups_snap_hough_to_group():
    groups = [PG(10.0, 100.0, 2), PG(85.0, 1.0, 2)]
    d = rot.final_direction(2, groups, lambda: 70.0, [0.0])
    assert d.angle == 85.0  # nearest group wins, not the longest
    assert d.rule == "multi-cc hough snapped to parallel group"


def test_final_direction_fallbacks_without_hough_or_globals():
    assert rot.final_direction(2, [], lambda: None, [10.0]).angle == 0.0
    groups = [PG(10.0, 100.0, 2), PG(85.0, 1.0, 2)]
    assert rot.final_direction(2, groups, lambda: None, []).angle == 10.0
    assert rot.final_direction(2, [], lambda: 33.0, []).angle == 33.0
