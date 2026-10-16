"""Tests for vector_metrics.py (count / length / direction / image) and
vector_render.py."""
from __future__ import annotations

import math

import numpy as np
import pytest

from rastervec.Evaluation.Evaluate.vector_metrics import (
    GeometryEntry,
    VectorMetricConfig,
    aggregate_vector_metrics,
    build_strokes,
    direction_histogram,
    evaluate_vector_metrics,
    flatten_entries,
    geometry_entries_from_vector,
    image_similarity,
    length_stats,
    pair_strokes,
)

CFG = VectorMetricConfig()


def _line(p0, p1, *, group=-1, vector_type="", width=None):
    return GeometryEntry(kind="l", points=(p0, p1), group=group, vector_type=vector_type, width=width)


def _strokes(entries):
    return build_strokes(flatten_entries(entries, CFG), CFG)


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
    entries = [_line((0, 0), (5, 0), group=0), _line((5, 0), (10, 0), group=0), _line((10, 0), (10, 5), group=1)]
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


# ---------------------------------------------------------------- strokes / count
def test_staircase_chains_into_one_stroke_and_pairs():
    gt = [_line((0, 0), (40, 7))]
    pred = _staircase(40, 7 / 40)
    res = _eval(gt, pred)
    assert res.count.n_pred_strokes == 1
    assert res.count.n_paired == 1
    assert res.count.precision.value == 1.0 and res.count.recall.value == 1.0


def test_collinear_pieces_from_separate_vectors_merge():
    gt = [_line((0, 0), (30, 0))]
    pred = [_line((0, 0), (10, 0), group=0), _line((11, 0), (20, 0), group=1), _line((20.5, 0), (30, 0), group=2)]
    res = _eval(gt, pred)
    assert res.count.n_pred_strokes == 1
    assert res.count.precision.value == 1.0 and res.count.recall.value == 1.0


def test_perpendicular_pieces_sharing_an_end_do_not_merge():
    strokes = _strokes([_line((0, 0), (10, 0), group=0), _line((10, 0), (10, 10), group=1)])
    assert len(strokes) == 2


def test_polyline_corner_splits_into_two_strokes():
    strokes = _strokes([_line((0, 0), (10, 0), group=0), _line((10, 0), (10, 10), group=0)])
    assert len(strokes) == 2


def test_crossing_rejoins_straight_through():
    # A "+" traced as 4 arms ending at the centre -> 2 straight strokes.
    arms = [
        _line((0, 0), (10, 0), group=0), _line((10, 0), (20, 0), group=1),
        _line((10, -10), (10, 0), group=2), _line((10, 0), (10, 10), group=3),
    ]
    strokes = _strokes(arms)
    assert len(strokes) == 2


def test_reversed_orientation_still_pairs():
    gt = _strokes([_line((0, 0), (10, 0))])
    pred = _strokes([_line((10, 0), (0, 0))])
    assert len(pair_strokes(gt, pred, CFG).paired) == 1


def test_endpoints_close_but_direction_off_does_not_pair():
    # Both ends within tolerance, but the pred is a short diagonal hop.
    gt = _strokes([_line((0, 0), (3, 0))])
    pred = _strokes([_line((0, 1.5), (3, -1.5))])
    assert pair_strokes(gt, pred, CFG).paired == []


def test_outside_tolerance_unpaired():
    gt = _strokes([_line((0, 0), (10, 0))])
    pred = _strokes([_line((0, 5), (10, 5))])
    p = pair_strokes(gt, pred, CFG)
    assert p.paired == [] and p.unpaired_gt == [0] and p.unpaired_pred == [0]


def test_recall_by_type():
    gt_by_type = {
        "vector_to_raster": [_line((0, 0), (10, 0), vector_type="vector_to_raster")],
        "original_raster": [_line((0, 50), (10, 50), vector_type="original_raster")],
    }
    res = evaluate_vector_metrics(gt_by_type, [_line((0, 0), (10, 0))], {}, cfg=CFG)
    assert res.count.recall_by_type["vector_to_raster"].value == 1.0
    assert res.count.recall_by_type["original_raster"].value == 0.0
    assert res.count.recall.value == 0.5


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
    from rastervec.Evaluation.Evaluate.vector_render import render_pair

    gt = {"vector_to_raster": [_line((10, 10), (190, 10), width=1.0)], "original_raster": []}
    pm = page_meta(width=200, height=100)
    gt_img, pred_img = render_pair(gt, [_line((10, 10), (190, 10), width=1.0)], pm, CFG)
    assert gt_img.shape == pred_img.shape
    assert abs(gt_img.shape[0] - 100 * 150 / 72) <= 1 and abs(gt_img.shape[1] - 200 * 150 / 72) <= 1
    assert gt_img.min() < 128  # ink was drawn
    res = evaluate_vector_metrics(gt, [_line((10, 10), (190, 10), width=1.0)], {}, images=(gt_img, pred_img), cfg=CFG)
    assert res.image.ssim == pytest.approx(1.0)

    _gi, blank = render_pair(gt, [], pm, CFG)
    assert blank.min() == 255


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
    assert agg.count.n_paired == 1 and agg.count.n_missed == 1
    assert agg.count.recall.value == 0.5
    assert agg.length.gt_length == 20 and agg.length.pred_length == 10
    assert agg.image.n_pixels == 2 * 32 * 32
    assert agg.image.mse == pytest.approx((r1.image.mse + r2.image.mse) / 2)
    assert agg.image.ssim == pytest.approx((r1.image.ssim + r2.image.ssim) / 2)
    # summed histograms: GT half H half V, pred all H
    assert agg.direction.intersection == pytest.approx(0.5)


def test_aggregate_empty_is_none():
    assert aggregate_vector_metrics([]) is None
