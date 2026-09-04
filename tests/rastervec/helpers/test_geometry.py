from __future__ import annotations

import pytest

from rastervec.helpers import geometry


def test_round_color_none():
    assert geometry.round_color(None) is None
    assert geometry.round_color(()) is None


def test_round_color_rounds():
    assert geometry.round_color((0.123456, 1.0, 0.0)) == (0.123, 1.0, 0.0)


def test_rect_gap_overlapping_is_zero():
    assert geometry.rect_gap((0, 0, 10, 10), (5, 5, 15, 15)) == 0.0


def test_rect_gap_touching_is_zero():
    assert geometry.rect_gap((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_rect_gap_horizontal_separation():
    assert geometry.rect_gap((0, 0, 10, 10), (13, 0, 20, 10)) == pytest.approx(3.0)


def test_rect_gap_diagonal_separation():
    # boxes offset by (3, 4) with no overlap on either axis -> gap = 5 (3-4-5 triangle)
    assert geometry.rect_gap((0, 0, 10, 10), (13, 14, 20, 20)) == pytest.approx(5.0)


def test_bbox_iou_identical_boxes_is_one():
    box = (0, 0, 10, 10)
    assert geometry.bbox_iou(box, box) == pytest.approx(1.0)


def test_bbox_iou_disjoint_boxes_is_zero():
    assert geometry.bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_bbox_iou_partial_overlap():
    # (0,0,10,10) and (5,5,15,15): intersection 5x5=25, union 100+100-25=175
    assert geometry.bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)


@pytest.mark.parametrize(
    "dashes, expected",
    [
        (None, False),
        ("", False),
        ("[] 0", False),
        ("[3 2] 0", True),
    ],
)
def test_is_dashed(dashes, expected):
    assert geometry.is_dashed(dashes) is expected


def test_bbox_area_basic():
    assert geometry.bbox_area((0, 0, 10, 4)) == pytest.approx(40.0)


def test_bbox_area_degenerate_is_zero():
    assert geometry.bbox_area((5, 5, 5, 20)) == 0.0
    assert geometry.bbox_area((10, 0, 0, 10)) == 0.0  # inverted extent


def test_bbox_intersection_area_partial():
    assert geometry.bbox_intersection_area((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25.0)


def test_bbox_intersection_area_disjoint_is_zero():
    assert geometry.bbox_intersection_area((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_bbox_intersection_area_touching_is_zero():
    assert geometry.bbox_intersection_area((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_bbox_coverage_fraction():
    # small box fully inside a big one -> big is 100% covered? no: coverage(a,b)
    # is fraction of a covered by b. a=(0,0,10,10) area 100, b=(0,0,5,10) -> 50
    assert geometry.bbox_coverage((0, 0, 10, 10), (0, 0, 5, 10)) == pytest.approx(0.5)


def test_bbox_coverage_fully_contained_is_one():
    assert geometry.bbox_coverage((2, 2, 4, 4), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_bbox_coverage_degenerate_a_is_zero():
    assert geometry.bbox_coverage((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


def test_bbox_iou_regression_via_intersection_area():
    # bbox_iou now delegates to bbox_intersection_area; values must be unchanged
    assert geometry.bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)
    assert geometry.bbox_iou((0, 0, 4, 4), (2, 0, 6, 4)) == pytest.approx(8 / 24)


def test_dims_and_max_dimension():
    assert geometry.dims((1, 2, 4, 10)) == (3, 8)
    assert geometry.max_dimension((1, 2, 4, 10)) == 8
    assert geometry.max_dimension((5, 5, 3, 3)) == 0.0  # inverted -> clamped


def test_bboxes_intersect():
    assert geometry.bboxes_intersect((0, 0, 10, 10), (5, 5, 15, 15))
    assert geometry.bboxes_intersect((0, 0, 10, 10), (10, 0, 20, 10))  # touching
    assert not geometry.bboxes_intersect((0, 0, 10, 10), (11, 0, 20, 10))


def test_bbox_contains():
    assert geometry.bbox_contains((0, 0, 10, 10), 5, 5)
    assert geometry.bbox_contains((0, 0, 10, 10), 0, 0)  # on edge
    assert not geometry.bbox_contains((0, 0, 10, 10), 11, 5)


def test_bbox_fully_contains():
    assert geometry.bbox_fully_contains((0, 0, 10, 10), (2, 2, 4, 4))
    assert geometry.bbox_fully_contains((2, 2, 4, 4), (0, 0, 10, 10))
    assert geometry.bbox_fully_contains((0, 0, 10, 10), (0, 0, 10, 10))  # equal
    assert not geometry.bbox_fully_contains((0, 0, 10, 10), (5, 5, 15, 15))  # partial
    assert not geometry.bbox_fully_contains((0, 0, 10, 10), (20, 20, 30, 30))  # disjoint


def test_make_oriented_quad_horizontal_is_the_bbox():
    ul, ur, lr, ll = geometry.make_oriented_quad((0, 0, 10, 4), 1.0, 0.0)
    assert ul == pytest.approx((0, 0))
    assert ur == pytest.approx((10, 0))
    assert lr == pytest.approx((10, 4))
    assert ll == pytest.approx((0, 4))


def test_make_oriented_quad_vertical_direction():
    # text direction pointing "down" (0, 1): the along-axis edge (ul->ur)
    # should run vertically and span the bbox's height (10), not its width.
    from math import hypot

    ul, ur, lr, ll = geometry.make_oriented_quad((0, 0, 4, 10), 0.0, 1.0)
    assert hypot(ur[0] - ul[0], ur[1] - ul[1]) == pytest.approx(10.0)
    assert hypot(ll[0] - ul[0], ll[1] - ul[1]) == pytest.approx(4.0)

