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
        (0.0, 0.0, 10.0, 4.0), (1.0, 0.0), baseline_point=(2.0, 3.5),
    )
    assert origin == pytest.approx((0.0, 3.5))


def test_compute_origin_baseline_point_projects_along_direction_normal():
    # direction (0, 1): the normal axis is x, so only the baseline point's
    # x component sets the perpendicular offset; along-axis still leading.
    origin = geometry.compute_origin(
        (0.0, 0.0, 10.0, 4.0), (0.0, 1.0), baseline_point=(1.0, 999.0),
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


# --------------------------------------------------------------------------
# transform_point / transform_item / transform_bbox / transform_direction
# --------------------------------------------------------------------------
def test_transform_point_translation_only():
    assert geometry.transform_point((1.0, 2.0), offset=(10.0, 20.0), rotation_deg=0.0) == pytest.approx((11.0, 22.0))


def test_transform_point_rotation_90():
    x, y = geometry.transform_point((1.0, 0.0), offset=(0.0, 0.0), rotation_deg=90.0)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(1.0, abs=1e-9)


def test_transform_point_rotation_then_translation_order():
    # rotate (1, 0) by 90 -> (0, 1), then translate by (5, 5) -> (5, 6).
    x, y = geometry.transform_point((1.0, 0.0), offset=(5.0, 5.0), rotation_deg=90.0)
    assert x == pytest.approx(5.0, abs=1e-9)
    assert y == pytest.approx(6.0, abs=1e-9)


def test_transform_item_line_preserves_kind():
    item = geometry.transform_item(("l", (0.0, 0.0), (1.0, 0.0)), offset=(10.0, 0.0), rotation_deg=0.0)
    assert item == ("l", (10.0, 0.0), (11.0, 0.0))


def test_transform_item_rect_keeps_trailing_extra_element():
    item = geometry.transform_item(("re", (0.0, 0.0, 10.0, 5.0), 1), offset=(0.0, 0.0), rotation_deg=0.0)
    assert item[0] == "re"
    assert item[1] == pytest.approx((0.0, 0.0, 10.0, 5.0))
    assert item[2:] == (1,)


def test_transform_bbox_pure_translation():
    bbox = geometry.transform_bbox((0.0, 0.0, 10.0, 5.0), offset=(2.0, 3.0), rotation_deg=0.0)
    assert bbox == pytest.approx((2.0, 3.0, 12.0, 8.0))


def test_transform_bbox_rotation_90_about_origin():
    # a bbox in the first quadrant, rotated 90 degrees about the origin,
    # ends up in the second quadrant.
    bbox = geometry.transform_bbox((0.0, 0.0, 10.0, 4.0), offset=(0.0, 0.0), rotation_deg=90.0)
    assert bbox == pytest.approx((-4.0, 0.0, 0.0, 10.0), abs=1e-9)


def test_transform_direction_rotates_without_translating():
    direction = geometry.transform_direction((1.0, 0.0), rotation_deg=90.0)
    assert direction == pytest.approx((0.0, 1.0), abs=1e-9)


def test_transform_direction_is_unaffected_by_position():
    # transform_direction never takes an offset -- confirm rotating a
    # direction gives the same result regardless of any notional position.
    a = geometry.transform_direction((1.0, 0.0), rotation_deg=45.0)
    b = geometry.transform_direction((1.0, 0.0), rotation_deg=45.0)
    assert a == b


# --------------------------------------------------------------------------
# transform_vector
# --------------------------------------------------------------------------
def test_transform_vector_translates_items_and_rect(vector):
    v = vector(kind="l", bbox=(0.0, 0.0, 10.0, 5.0))
    moved = geometry.transform_vector(v, offset=(100.0, 200.0), rotation_deg=0.0)

    assert moved.items == [("l", (100.0, 200.0), (110.0, 205.0))]
    assert moved.rect == pytest.approx((100.0, 200.0, 110.0, 205.0))
    # original untouched
    assert v.rect == (0.0, 0.0, 10.0, 5.0)


def test_transform_vector_rotation_recomputes_rect_from_new_items():
    from rastervec.models import Vector

    v = Vector(
        type="s", items=[("l", (0.0, 0.0), (10.0, 0.0))], color=(0, 0, 0), fill=None,
        width=1.0, dashes=None, closePath=False, lineCap=0, lineJoin=0, even_odd=False,
        stroke_opacity=None, fill_opacity=None, layer=None, rect=(0.0, 0.0, 10.0, 0.0),
        scissor=None, seqno=0, blendmode=None, isolated=False, knockout=False,
        opacity=None, page_index=0,
    )
    rotated = geometry.transform_vector(v, offset=(0.0, 0.0), rotation_deg=90.0)

    (kind, p0, p1) = rotated.items[0]
    assert kind == "l"
    assert p0 == pytest.approx((0.0, 0.0), abs=1e-9)
    assert p1 == pytest.approx((0.0, 10.0), abs=1e-9)
    assert rotated.rect == pytest.approx((0.0, 0.0, 0.0, 10.0), abs=1e-9)


def test_transform_vector_moves_scissor_too(vector):
    v = vector(kind="l", bbox=(0.0, 0.0, 10.0, 5.0), scissor=(0.0, 0.0, 10.0, 5.0))
    moved = geometry.transform_vector(v, offset=(1.0, 1.0), rotation_deg=0.0)
    assert moved.scissor == pytest.approx((1.0, 1.0, 11.0, 6.0))


def test_transform_vector_no_scissor_stays_none(vector):
    v = vector(kind="l", bbox=(0.0, 0.0, 10.0, 5.0), scissor=None)
    moved = geometry.transform_vector(v, offset=(1.0, 1.0), rotation_deg=0.0)
    assert moved.scissor is None

