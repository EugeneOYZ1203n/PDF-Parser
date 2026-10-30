"""Tests for vector_metrics.py (Liu-Dori wiring / pixel / length /
direction / image) and vector_render.py. The Liu-Dori protocol itself is
tested in test_liu_dori.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from raster_parser.evaluation.evaluate.vector_metrics import (
    GeometryEntry,
    VectorMetricConfig,
    aggregate_vector_metrics,
    direction_histogram,
    evaluate_vector_metrics,
    flatten_entries,
    geometry_entries_from_vector,
    image_similarity,
    length_stats,
)

CFG = VectorMetricConfig()


def _line(p0, p1, *, group=-1, vector_type="", width=None):
    return GeometryEntry(
        kind="l", points=(p0, p1), group=group, vector_type=vector_type, width=width
    )


def _staircase(x_end: float, slope: float, group: int = 0):
    """An 8-connected 1 px pixel walk along y = slope * x, one "l" per step."""
    pts = [(0.0, 0.0)]
    x, y = 0, 0
    while x < x_end:
        x += 1
        ny = round(slope * x)
        pts.append((float(x), float(ny)))
        y = ny
    return [_line(a, b, group=group) for a, b in zip(pts, pts[1:])]


def _eval(gt, preds, **kw):
    gt_by_type = {"vector_to_raster": gt, "original_raster": []}
    return evaluate_vector_metrics(gt_by_type, preds, {"vector_to_raster": len(gt)}, cfg=CFG, **kw)


# ---------------------------------------------------------------- flatten
def test_flatten_joins_touching_entries_of_one_group():
    entries = [
        _line((0, 0), (5, 0), group=0),
        _line((5, 0), (10, 0), group=0),
        _line((10, 0), (10, 5), group=1),
    ]
    polys = flatten_entries(entries, CFG)
    assert len(polys) == 2
    assert polys[0].points.shape == (3, 2)


def test_flatten_curve_into_ten_pieces():
    c = GeometryEntry(kind="c", points=((0, 0), (0, 10), (10, 10), (10, 0)))
    (poly,) = flatten_entries([c], CFG)
    assert len(poly.points) == 11


def test_geometry_entries_from_vector_rect_gives_four_lines(vector):
    v = vector(kind="re", bbox=(0.0, 0.0, 4.0, 2.0))
    entries = geometry_entries_from_vector(v, group=3)
    assert [e.kind for e in entries] == ["l"] * 4
    assert all(e.group == 3 for e in entries)


# ---------------------------------------------------------------- Liu-Dori wiring
def test_staircase_pred_items_match_a_diagonal_gt_line():
    # Raw items, no RDP: each 1-px step is its own predicted bar.
    gt = [_line((0, 0), (40, 7), width=1.0)]
    pred = _staircase(40, 7 / 40)
    res = _eval(gt, pred)
    ld = res.liu_dori
    assert ld.pred.n_lines == len(pred)
    assert ld.gt.histogram["6+"] == 1
    assert 0.0 < ld.d_v < 1.0


def test_union_of_gt_types_and_split_by_type():
    gt_by_type = {
        "vector_to_raster": [_line((0, 0), (10, 0), vector_type="vector_to_raster", width=1.0)],
        "original_raster": [_line((0, 50), (10, 50), vector_type="original_raster", width=1.0)],
    }
    res = evaluate_vector_metrics(gt_by_type, [_line((0, 0), (10, 0), width=1.0)], {}, cfg=CFG)
    ld = res.liu_dori
    assert ld.by_type["vector_to_raster"].d_v.value == pytest.approx(1.0)
    assert ld.by_type["original_raster"].d_v.value == 0.0
    assert ld.d_v == pytest.approx(0.5)
    assert ld.f_v == pytest.approx(0.0)


def test_dpi_sets_the_width_floor():
    # A 0.3 pt offset matches at 72 dpi (floor 2 pt) but not at 600 dpi
    # (floor 0.24 pt).
    gt, pred = [_line((0, 0), (50, 0), width=0.0)], [_line((0, 0.3), (50, 0.3), width=0.0)]
    assert _eval(gt, pred, dpi=72.0).liu_dori.n_overlaps == 1
    assert _eval(gt, pred, dpi=600.0).liu_dori.n_overlaps == 0


# ---------------------------------------------------------------- length / direction
def test_length_ratio():
    gt = flatten_entries([_line((0, 0), (10, 0))], CFG)
    pred = flatten_entries([_line((0, 0), (5, 0))], CFG)
    ls = length_stats(gt, pred)
    assert ls.gt_length == 10 and ls.pred_length == 5 and ls.ratio.value == 0.5


def test_curve_length_close_to_quarter_circle():
    k = 0.5522847498 * 10
    c = GeometryEntry(kind="c", points=((10, 0), (10, k), (k, 10), (0, 10)))
    ls = length_stats(flatten_entries([c], CFG), [])
    assert ls.gt_length == pytest.approx(math.pi * 5, rel=5e-3)


def test_direction_identical_is_one_and_orthogonal_is_zero():
    h = _eval([_line((0, 0), (10, 0))], [_line((0, 0), (10, 0))]).direction
    assert h.intersection == pytest.approx(1.0)
    v = _eval([_line((0, 0), (10, 0))], [_line((0, 0), (0, 10))]).direction
    assert v.intersection == pytest.approx(0.0)


def test_direction_is_undirected():
    a = direction_histogram(flatten_entries([_line((0, 0), (10, 0))], CFG), 36)
    b = direction_histogram(flatten_entries([_line((10, 0), (0, 0))], CFG), 36)
    assert np.allclose(a, b)


def test_staircase_direction_is_penalised_on_raw_geometry():
    res = _eval([_line((0, 0), (40, 7))], _staircase(40, 7 / 40))
    assert res.direction.intersection < 0.5


def test_empty_prediction_direction_nan_and_length_zero():
    res = _eval([_line((0, 0), (10, 0))], [])
    assert math.isnan(res.direction.intersection)
    assert res.length.ratio.value == 0.0


# ---------------------------------------------------------------- image
def test_image_similarity_identical_and_blank():
    a = np.full((64, 64), 255, np.uint8)
    a[30:34, 5:60] = 0
    same = image_similarity(a, a.copy(), CFG)
    assert same.ssim == pytest.approx(1.0) and same.psnr == CFG.psnr_cap_db
    blank = image_similarity(a, np.full_like(a, 255), CFG)
    assert blank.ssim < 1.0 and blank.psnr < CFG.psnr_cap_db


def test_render_pair_and_evaluate(page_meta):
    from raster_parser.evaluation.evaluate.vector_render import render_pair

    gt = {"vector_to_raster": [_line((10, 10), (190, 10), width=1.0)], "original_raster": []}
    pm = page_meta(width=200, height=100)
    gt_img, pred_img = render_pair(gt, [_line((10, 10), (190, 10), width=1.0)], pm, CFG)
    assert gt_img.shape == pred_img.shape
    assert abs(gt_img.shape[0] - 100 * 150 / 72) <= 1 and abs(gt_img.shape[1] - 200 * 150 / 72) <= 1
    assert gt_img.min() < 128  # ink was drawn
    res = evaluate_vector_metrics(
        gt, [_line((10, 10), (190, 10), width=1.0)], {}, images=(gt_img, pred_img), cfg=CFG
    )
    assert res.image.ssim == pytest.approx(1.0)
    assert math.isnan(res.pixel.pri)  # no ink masks given

    _gi, blank = render_pair(gt, [], pm, CFG)
    assert blank.min() == 255


def test_render_ink_pair_and_pixel_stats(page_meta):
    from raster_parser.evaluation.evaluate.vector_render import render_ink_pair

    gt = {"vector_to_raster": [_line((10, 10), (190, 10), width=0.0)], "original_raster": []}
    pm = page_meta(width=200, height=100)
    gt_ink, pred_ink = render_ink_pair(gt, [_line((10, 10), (100, 10), width=0.0)], pm, 300.0, CFG)
    assert gt_ink.shape == pred_ink.shape
    assert gt_ink.dtype == bool and gt_ink.any()  # a hairline still renders at the floor width
    res = evaluate_vector_metrics(
        gt, [_line((10, 10), (100, 10), width=0.0)], {}, ink=(gt_ink, pred_ink), cfg=CFG
    )
    assert res.pixel.d_p.value == pytest.approx(0.5, abs=0.05)
    assert res.pixel.f_p.value == pytest.approx(0.0, abs=0.01)


def test_no_images_gives_nan():
    res = _eval([_line((0, 0), (10, 0))], [])
    assert math.isnan(res.image.ssim) and math.isnan(res.image.psnr)


# ---------------------------------------------------------------- aggregate
def test_aggregate_sums_and_pools():
    a = np.full((32, 32), 255, np.uint8)
    b = a.copy()
    b[0, 0] = 0
    r1 = _eval([_line((0, 0), (10, 0))], [_line((0, 0), (10, 0))], images=(a, a.copy()))
    r2 = _eval([_line((0, 0), (0, 10))], [], images=(a, b))
    agg = aggregate_vector_metrics([r1, r2])
    assert agg.liu_dori.d_v == pytest.approx(0.5)
    assert agg.liu_dori.gt.histogram["1"] == 1 and agg.liu_dori.gt.histogram["0"] == 1
    assert agg.length.gt_length == 20 and agg.length.pred_length == 10
    assert agg.image.n_pixels == 2 * 32 * 32
    assert agg.image.mse == pytest.approx((r1.image.mse + r2.image.mse) / 2)
    assert agg.image.ssim == pytest.approx((r1.image.ssim + r2.image.ssim) / 2)
    # summed histograms: GT half H half V, pred all H
    assert agg.direction.intersection == pytest.approx(0.5)


def test_aggregate_empty_is_none():
    assert aggregate_vector_metrics([]) is None
