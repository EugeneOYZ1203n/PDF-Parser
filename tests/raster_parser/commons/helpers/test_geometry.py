from __future__ import annotations

import pytest

from raster_parser.commons.helpers import geometry


def test_round_color_none():
    assert geometry.round_color(None) is None
    assert geometry.round_color(()) is None


def test_round_color_rounds():
    assert geometry.round_color((0.123456, 1.0, 0.0)) == (0.123, 1.0, 0.0)


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


# --------------------------------------------------------------------------
# compute_origin
# --------------------------------------------------------------------------
def test_compute_origin_horizontal_is_bottom_left():
    origin = geometry.compute_origin((0.0, 0.0, 10.0, 4.0), (1.0, 0.0))
    assert origin == pytest.approx((0.0, 2.0))


def test_compute_origin_matches_leading_edge_for_rotated_direction():
    # direction pointing straight down (0, 1): the "leading" (along-min)
    # edge is the bbox's top, centered on the normal (x) axis.
    origin = geometry.compute_origin((0.0, 0.0, 10.0, 4.0), (0.0, 1.0))
    assert origin == pytest.approx((5.0, 0.0))


def test_compute_origin_uses_baseline_point_normal_offset_when_given():
    # With a baseline sample, the perpendicular offset comes from that
    # point (here y=3.5, near the bbox bottom), not the bbox centre (y=2).
    origin = geometry.compute_origin(
        (0.0, 0.0, 10.0, 4.0),
        (1.0, 0.0),
        baseline_point=(2.0, 3.5),
    )
    assert origin == pytest.approx((0.0, 3.5))


def test_compute_origin_baseline_point_projects_along_direction_normal():
    # direction (0, 1): the normal axis is x, so only the baseline point's
    # x component sets the perpendicular offset; along-axis still leading.
    origin = geometry.compute_origin(
        (0.0, 0.0, 10.0, 4.0),
        (0.0, 1.0),
        baseline_point=(1.0, 999.0),
    )
    assert origin == pytest.approx((1.0, 0.0))


# --------------------------------------------------------------------------
# Vector.items geometry -- item_points / item_bbox
# --------------------------------------------------------------------------
def test_item_points_line():
    assert geometry.item_points(("l", (0.0, 1.0), (2.0, 3.0))) == [(0.0, 1.0), (2.0, 3.0)]


def test_item_points_rect():
    assert geometry.item_points(("re", (0.0, 0.0, 10.0, 5.0))) == [(0.0, 0.0), (10.0, 5.0)]


def test_item_points_quad():
    corners = ((0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0))
    assert geometry.item_points(("qu", corners)) == list(corners)


def test_item_points_curve():
    pts = geometry.item_points(("c", (0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)))
    assert pts == [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]


def test_item_points_unknown_kind_is_empty():
    assert geometry.item_points(("x", 1, 2)) == []


def test_item_bbox_line():
    assert geometry.item_bbox(("l", (0.0, 5.0), (10.0, 0.0))) == (0.0, 0.0, 10.0, 5.0)


def test_item_bbox_empty_for_unknown_kind():
    assert geometry.item_bbox(("x",)) == (0.0, 0.0, 0.0, 0.0)


def test_bbox_iou_identical_boxes_is_one():
    box = (0, 0, 10, 10)
    assert geometry.bbox_iou(box, box) == pytest.approx(1.0)


def test_bbox_iou_disjoint_boxes_is_zero():
    assert geometry.bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_bbox_iou_partial_overlap():
    # (0,0,10,10) and (5,5,15,15): intersection 5x5=25, union 100+100-25=175
    assert geometry.bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)


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


def test_bbox_iou_regression_via_intersection_area():
    # bbox_iou now delegates to bbox_intersection_area; values must be unchanged
    assert geometry.bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)) == pytest.approx(25 / 175)
    assert geometry.bbox_iou((0, 0, 4, 4), (2, 0, 6, 4)) == pytest.approx(8 / 24)
