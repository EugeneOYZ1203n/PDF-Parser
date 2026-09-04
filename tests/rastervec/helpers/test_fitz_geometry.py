from __future__ import annotations

import pymupdf as fitz
import pytest

from rastervec.helpers import fitz_geometry


def test_point_angle_horizontal():
    assert fitz_geometry.point_angle(fitz.Point(0, 0), fitz.Point(10, 0)) == pytest.approx(0.0)


def test_point_angle_vertical():
    assert fitz_geometry.point_angle(fitz.Point(0, 0), fitz.Point(0, 10)) == pytest.approx(90.0)


def test_point_angle_diagonal():
    assert fitz_geometry.point_angle(fitz.Point(0, 0), fitz.Point(10, 10)) == pytest.approx(45.0)


def test_point_angle_zero_length():
    assert fitz_geometry.point_angle(fitz.Point(5, 5), fitz.Point(5, 5)) == 0.0


def test_line_length():
    assert fitz_geometry.line_length(fitz.Point(0, 0), fitz.Point(3, 4)) == pytest.approx(5.0)


def test_quad_angle_horizontal():
    quad = fitz.Quad(
        fitz.Point(0, 0),
        fitz.Point(10, 0),
        fitz.Point(0, 5),
        fitz.Point(10, 5),
    )
    assert fitz_geometry.quad_angle(quad) == pytest.approx(0.0)


def test_matrix_rotation_identity():
    assert fitz_geometry.matrix_rotation(fitz.Matrix(1, 1)) == pytest.approx(0.0)


def test_matrix_rotation_90():
    # rotation matrix for 90 degrees: a=0, b=1, c=-1, d=0
    m = fitz.Matrix(0, 1, -1, 0, 0, 0)
    assert fitz_geometry.matrix_rotation(m) == pytest.approx(90.0)


def test_matrix_scale_known():
    m = fitz.Matrix(2, 0, 0, 3, 0, 0)
    sx, sy = fitz_geometry.matrix_scale(m)
    assert sx == pytest.approx(2.0)
    assert sy == pytest.approx(3.0)


def test_format_matrix_rounds():
    m = fitz.Matrix(1.0000001, 0, 0, 1.0000001, 0, 0)
    a, b, c, d, e, f = fitz_geometry.format_matrix(m)
    assert a == pytest.approx(1.0)
