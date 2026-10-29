"""The inspector reuses the commons geometry / colour helpers instead of keeping its own copies."""

from __future__ import annotations

import colorsys

import pymupdf
import pytest

from raster_parser.commons.helpers import pymupdf_geometry
from raster_parser.commons.helpers.geometry import make_oriented_quad, rgb01_to_hex, round_color
from raster_parser.commons.renderer._shapes import path_color_hex
from scripts.inspector import pdf_model_core
from scripts.inspector.layer_colors import rgb_to_hex
from scripts.inspector.pdf_model_text import _make_oriented_quad


@pytest.mark.parametrize(
    "name",
    [
        "point_angle",
        "line_length",
        "quad_angle",
        "quad_metadata",
        "rect_metadata",
        "format_matrix",
        "matrix_rotation",
        "matrix_scale",
    ],
)
def test_inspector_metadata_helpers_are_the_commons_ones(name):
    assert getattr(pdf_model_core, f"_{name}") is getattr(pymupdf_geometry, name)


def test_inspector_round_color_is_the_commons_one():
    assert pdf_model_core._round_color is round_color


@pytest.mark.parametrize(
    "bbox, dx, dy",
    [
        ((10.0, 20.0, 50.0, 30.0), 1.0, 0.0),
        ((10.0, 20.0, 50.0, 30.0), 0.0, -1.0),
        ((3.5, 7.25, 41.0, 66.0), 0.6, 0.8),
        ((0.0, 0.0, 1.0, 1.0), 0.0, 0.0),
    ],
)
def test_oriented_quad_matches_commons(bbox, dx, dy):
    quad = _make_oriented_quad(pymupdf.Rect(bbox), dx, dy)
    ul, ur, lr, ll = make_oriented_quad(bbox, dx, dy)
    assert isinstance(quad, pymupdf.Quad)
    assert [tuple(quad.ul), tuple(quad.ur), tuple(quad.lr), tuple(quad.ll)] == [ul, ur, lr, ll]


@pytest.mark.parametrize(
    "color",
    [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (0.2, 0.4, 0.6),
        (1.2, -0.1, 0.5),
        colorsys.hsv_to_rgb(0.75, 1.0, 1.0),
        (0.002, 0.998, 0.5019),
    ],
)
def test_hex_colour_helpers_agree(color, vector):
    assert rgb_to_hex(color) == rgb01_to_hex(color)
    assert path_color_hex(vector(color=color)) == rgb01_to_hex(color)


def test_rgb_to_hex_keeps_black_for_missing_colour():
    assert rgb_to_hex(None) == "#000000"
    assert rgb_to_hex(()) == "#000000"
