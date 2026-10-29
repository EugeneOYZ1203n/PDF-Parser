"""Placing an image's vectors on the page (`extract_vectors_from_png._transform_vectors_to_matrix`).

Image size 72 x 72 px at 72 dpi, so a vector point (x, y) is the unit-square point (x/72, y/72)
before the placement matrix.
"""

from __future__ import annotations

import math

import pymupdf
import pytest

from raster_parser.extract_vectors_from_png import _transform_vectors_to_matrix

SIZE, DPI = (72, 72), 72


def _place(vector, matrix):
    (out,) = _transform_vectors_to_matrix([vector], matrix, SIZE, DPI)
    return out


def _map(pt, matrix):
    p = pymupdf.Point(pt[0] / 72, pt[1] / 72) * matrix
    return (p.x, p.y)


@pytest.fixture
def shapes(vector):
    return vector(
        type="fs",
        fill=(0.0, 0.0, 0.0),
        items=[
            ("l", (0.0, 0.0), (36.0, 0.0)),
            ("c", (0.0, 0.0), (10.0, 20.0), (30.0, 20.0), (36.0, 36.0)),
            ("re", (18.0, 18.0, 54.0, 36.0), 1),
            ("qu", ((0.0, 36.0), (36.0, 36.0), (36.0, 72.0), (0.0, 72.0))),
        ],
        bbox=(0.0, 0.0, 54.0, 72.0),
        scissor=(0.0, 0.0, 72.0, 72.0),
    )


@pytest.mark.parametrize(
    "matrix",
    [
        pymupdf.Matrix(200, 0, 0, 100, 50, 60),  # scale + translate
        pymupdf.Matrix(-200, 0, 0, 100, 300, 60),  # flip x
        pymupdf.Matrix(0, 100, -200, 0, 300, 60),  # quarter turn
    ],
)
def test_axis_preserving_placement_keeps_rectangles(shapes, matrix) -> None:
    out = _place(shapes, matrix)
    re_item = out.items[2]
    assert re_item[0] == "re" and re_item[2] == 1  # orientation flag kept
    pts = [_map(c, matrix) for c in ((18, 18), (54, 18), (54, 36), (18, 36))]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    assert re_item[1] == pytest.approx((min(xs), min(ys), max(xs), max(ys)))


def test_rotated_placement_turns_rectangles_into_quads(shapes) -> None:
    t = math.radians(30)
    matrix = pymupdf.Matrix(
        200 * math.cos(t), 200 * math.sin(t), -100 * math.sin(t), 100 * math.cos(t), 300, 300
    )
    out = _place(shapes, matrix)
    op, corners = out.items[2]
    assert op == "qu"
    expected = [_map(c, matrix) for c in ((18, 18), (54, 18), (54, 36), (18, 36))]  # ul ur lr ll
    for got, want in zip(corners, expected):
        assert got == pytest.approx(want)


def test_points_are_mapped_exactly(shapes) -> None:
    t = math.radians(30)
    matrix = pymupdf.Matrix(
        200 * math.cos(t), 200 * math.sin(t), -100 * math.sin(t), 100 * math.cos(t), 300, 300
    )
    out = _place(shapes, matrix)
    for kind, before, after in zip(
        "lcq",
        (shapes.items[0][1:], shapes.items[1][1:], shapes.items[3][1]),
        (out.items[0][1:], out.items[1][1:], out.items[3][1]),
    ):
        for p, q in zip(before, after):
            assert q == pytest.approx(_map(p, matrix)), kind


def test_rect_and_scissor_become_bounds_of_mapped_corners(shapes) -> None:
    matrix = pymupdf.Matrix(0, 100, -200, 0, 300, 60)
    out = _place(shapes, matrix)
    for before, after in ((shapes.rect, out.rect), (shapes.scissor, out.scissor)):
        x0, y0, x1, y1 = before
        pts = [_map(c, matrix) for c in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        assert after == pytest.approx((min(xs), min(ys), max(xs), max(ys)))


def test_input_vectors_are_not_modified(shapes) -> None:
    items = list(shapes.items)
    _place(shapes, pymupdf.Matrix(2, 0, 0, 2, 5, 5))
    assert shapes.items == items
