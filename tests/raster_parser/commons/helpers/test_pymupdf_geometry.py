from __future__ import annotations

import pymupdf
import pytest

from raster_parser.commons.helpers import pymupdf_geometry


def test_point_angle_horizontal():
    assert pymupdf_geometry.point_angle(pymupdf.Point(0, 0), pymupdf.Point(10, 0)) == pytest.approx(
        0.0
    )


def test_point_angle_vertical():
    assert pymupdf_geometry.point_angle(pymupdf.Point(0, 0), pymupdf.Point(0, 10)) == pytest.approx(
        90.0
    )


def test_point_angle_diagonal():
    assert pymupdf_geometry.point_angle(
        pymupdf.Point(0, 0), pymupdf.Point(10, 10)
    ) == pytest.approx(45.0)


def test_point_angle_zero_length():
    assert pymupdf_geometry.point_angle(pymupdf.Point(5, 5), pymupdf.Point(5, 5)) == 0.0


def test_line_length():
    assert pymupdf_geometry.line_length(pymupdf.Point(0, 0), pymupdf.Point(3, 4)) == pytest.approx(
        5.0
    )


def test_quad_angle_horizontal():
    quad = pymupdf.Quad(
        pymupdf.Point(0, 0),
        pymupdf.Point(10, 0),
        pymupdf.Point(0, 5),
        pymupdf.Point(10, 5),
    )
    assert pymupdf_geometry.quad_angle(quad) == pytest.approx(0.0)


def test_matrix_rotation_identity():
    assert pymupdf_geometry.matrix_rotation(pymupdf.Matrix(1, 1)) == pytest.approx(0.0)


def test_matrix_rotation_90():
    # rotation matrix for 90 degrees: a=0, b=1, c=-1, d=0
    m = pymupdf.Matrix(0, 1, -1, 0, 0, 0)
    assert pymupdf_geometry.matrix_rotation(m) == pytest.approx(90.0)


def test_matrix_scale_known():
    m = pymupdf.Matrix(2, 0, 0, 3, 0, 0)
    sx, sy = pymupdf_geometry.matrix_scale(m)
    assert sx == pytest.approx(2.0)
    assert sy == pytest.approx(3.0)


def test_format_matrix_rounds():
    m = pymupdf.Matrix(1.0000001, 0, 0, 1.0000001, 0, 0)
    a, b, c, d, e, f = pymupdf_geometry.format_matrix(m)
    assert a == pytest.approx(1.0)
