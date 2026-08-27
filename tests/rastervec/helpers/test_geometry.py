from __future__ import annotations

import math

import pymupdf as fitz
import pytest

from rastervec.helpers import geometry


def test_point_angle_horizontal():
    assert geometry.point_angle(fitz.Point(0, 0), fitz.Point(10, 0)) == pytest.approx(0.0)


def test_point_angle_vertical():
    assert geometry.point_angle(fitz.Point(0, 0), fitz.Point(0, 10)) == pytest.approx(90.0)


def test_point_angle_diagonal():
    assert geometry.point_angle(fitz.Point(0, 0), fitz.Point(10, 10)) == pytest.approx(45.0)


def test_point_angle_zero_length():
    assert geometry.point_angle(fitz.Point(5, 5), fitz.Point(5, 5)) == 0.0


def test_line_length():
    assert geometry.line_length(fitz.Point(0, 0), fitz.Point(3, 4)) == pytest.approx(5.0)


def test_quad_angle_horizontal():
    quad = fitz.Quad(
        fitz.Point(0, 0),
        fitz.Point(10, 0),
        fitz.Point(0, 5),
        fitz.Point(10, 5),
    )
    assert geometry.quad_angle(quad) == pytest.approx(0.0)


def test_round_color_none():
    assert geometry.round_color(None) is None
    assert geometry.round_color(()) is None


def test_round_color_rounds():
    assert geometry.round_color((0.123456, 1.0, 0.0)) == (0.123, 1.0, 0.0)


def test_matrix_rotation_identity():
    assert geometry.matrix_rotation(fitz.Matrix(1, 1)) == pytest.approx(0.0)


def test_matrix_rotation_90():
    # rotation matrix for 90 degrees: a=0, b=1, c=-1, d=0
    m = fitz.Matrix(0, 1, -1, 0, 0, 0)
    assert geometry.matrix_rotation(m) == pytest.approx(90.0)


def test_matrix_scale_known():
    m = fitz.Matrix(2, 0, 0, 3, 0, 0)
    sx, sy = geometry.matrix_scale(m)
    assert sx == pytest.approx(2.0)
    assert sy == pytest.approx(3.0)


def test_format_matrix_rounds():
    m = fitz.Matrix(1.0000001, 0, 0, 1.0000001, 0, 0)
    a, b, c, d, e, f = geometry.format_matrix(m)
    assert a == pytest.approx(1.0)


def test_rect_gap_overlapping_is_zero():
    assert geometry.rect_gap((0, 0, 10, 10), (5, 5, 15, 15)) == 0.0


def test_rect_gap_touching_is_zero():
    assert geometry.rect_gap((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_rect_gap_horizontal_separation():
    assert geometry.rect_gap((0, 0, 10, 10), (13, 0, 20, 10)) == pytest.approx(3.0)


def test_rect_gap_diagonal_separation():
    # boxes offset by (3, 4) with no overlap on either axis -> gap = 5 (3-4-5 triangle)
    assert geometry.rect_gap((0, 0, 10, 10), (13, 14, 20, 20)) == pytest.approx(5.0)
