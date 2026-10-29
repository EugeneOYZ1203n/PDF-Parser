"""commons/renderer/png.py + ocr_prep.py: isolated black-and-white cluster renders for OCR."""

from __future__ import annotations

import numpy as np
import pytest

from raster_parser.commons.renderer.ocr_prep import dpi_for_cluster, render_cluster_with_dynamic_dpi
from raster_parser.commons.renderer.png import render_vector_cluster


def _red_box(vector):
    return vector(
        bbox=(100.0, 200.0, 136.0, 218.0),
        items=[("re", (100.0, 200.0, 136.0, 218.0))],
        color=(1.0, 0.0, 0.0),
        width=2.0,
    )


def test_render_is_the_cluster_bbox_at_dpi(vector):
    img = render_vector_cluster([_red_box(vector)], dpi=144)
    assert img.size == (72, 36)  # 36 x 18 pt at 2 px / pt, no padding


def test_padding_expands_every_side(vector):
    img = render_vector_cluster([_red_box(vector)], dpi=72, padding=5.0)
    assert img.size == (46, 28)


def test_render_is_black_on_white(vector):
    arr = np.asarray(render_vector_cluster([_red_box(vector)], dpi=144, padding=2.0))
    colours = {tuple(px) for px in arr.reshape(-1, 3)}
    assert (255, 255, 255) in colours and (0, 0, 0) in colours
    assert all(r == g == b for r, g, b in colours)  # the red stroke is drawn black


def test_flat_cluster_still_renders(vector):
    rule = vector(bbox=(0.0, 10.0, 50.0, 10.0), items=[("l", (0.0, 10.0), (50.0, 10.0))])
    assert min(render_vector_cluster([rule], dpi=72).size) >= 1


def test_render_needs_a_vector():
    with pytest.raises(ValueError):
        render_vector_cluster([], dpi=72)


def test_dpi_is_raised_for_a_small_cluster_and_capped(vector):
    small = vector(bbox=(0.0, 0.0, 4.0, 2.0))
    assert dpi_for_cluster([small], 72, 0.0, min_render_side_px=100, max_render_dpi=10_000) == 3600
    assert dpi_for_cluster([small], 72, 0.0, min_render_side_px=100, max_render_dpi=1200) == 1200


def test_dpi_is_never_lowered(vector):
    big = vector(bbox=(0.0, 0.0, 400.0, 300.0))
    assert dpi_for_cluster([big], 300, 0.0, min_render_side_px=100, max_render_dpi=1200) == 300


def test_dynamic_render_reports_the_dpi_it_used(vector):
    small = vector(bbox=(0.0, 0.0, 4.0, 2.0), items=[("re", (0.0, 0.0, 4.0, 2.0))], width=0.5)
    img, dpi = render_cluster_with_dynamic_dpi([small], 72, 100, 1200)
    assert dpi == 1200 and min(img.size) == pytest.approx(2.0 * dpi / 72, abs=1)
