from __future__ import annotations

from rastervec.commons.helpers.geometry import clip_line_to_bbox


def test_clip_line_to_bbox_horizontal_through_center():
    clipped = clip_line_to_bbox((5.0, 5.0), (1.0, 0.0), (0.0, 0.0, 10.0, 10.0))
    assert clipped is not None
    p0, p1 = clipped
    assert {round(p0[0]), round(p1[0])} == {0, 10}
    assert round(p0[1]) == 5 and round(p1[1]) == 5


def test_clip_line_to_bbox_diagonal():
    clipped = clip_line_to_bbox((0.0, 0.0), (1.0, 1.0), (0.0, 0.0, 10.0, 10.0))
    assert clipped is not None
    p0, p1 = clipped
    pts = {p0, p1}
    assert (0.0, 0.0) in pts
    assert (10.0, 10.0) in pts


def test_clip_line_to_bbox_vertical_line():
    clipped = clip_line_to_bbox((5.0, 100.0), (0.0, 1.0), (0.0, 0.0, 10.0, 10.0))
    assert clipped is not None
    p0, p1 = clipped
    assert round(p0[0]) == 5 and round(p1[0]) == 5
    assert {round(p0[1]), round(p1[1])} == {0, 10}


def test_clip_line_to_bbox_misses_bbox_entirely():
    # A horizontal line well above the bbox never crosses it.
    clipped = clip_line_to_bbox((5.0, 50.0), (1.0, 0.0), (0.0, 0.0, 10.0, 10.0))
    assert clipped is None


def test_clip_line_to_bbox_parallel_and_outside():
    # A vertical line entirely to the right of the bbox.
    clipped = clip_line_to_bbox((50.0, 5.0), (0.0, 1.0), (0.0, 0.0, 10.0, 10.0))
    assert clipped is None
