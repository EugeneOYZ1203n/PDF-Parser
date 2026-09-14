"""Tests for vector_metrics.py -- categories 9-12."""
from __future__ import annotations

from rastervec.Evaluation.Evaluate.vector_metrics import (
    GeometryEntry,
    endpoint_accuracy_stats,
    pair_vectors,
    property_accuracy_table,
    vector_count_stats,
)


def _line(p0, p1, **kw):
    return GeometryEntry(kind="l", points=(p0, p1), **kw)


def test_pair_vectors_exact_match():
    gt = [_line((0, 0), (10, 0))]
    pred = [_line((0, 0), (10, 0))]
    pairing = pair_vectors(gt, pred, tolerance=1.0)
    assert pairing.paired == [(0, 0)]
    assert not pairing.unpaired_gt
    assert not pairing.unpaired_pred


def test_pair_vectors_reversed_endpoints_still_match():
    gt = [_line((0, 0), (10, 0))]
    pred = [_line((10, 0), (0, 0))]
    pairing = pair_vectors(gt, pred, tolerance=0.5)
    assert pairing.paired == [(0, 0)]
    assert pairing.paired_reversed == [True]


def test_pair_vectors_outside_tolerance_unpaired():
    gt = [_line((0, 0), (10, 0))]
    pred = [_line((0, 5), (10, 5))]
    pairing = pair_vectors(gt, pred, tolerance=1.0)
    assert not pairing.paired
    assert pairing.unpaired_gt == [0]
    assert pairing.unpaired_pred == [0]


def test_pair_vectors_cross_kind_never_matches():
    gt = [_line((0, 0), (10, 0))]
    pred = [GeometryEntry(kind="c", points=((0, 0), (1, 1), (2, 2), (10, 0)))]
    pairing = pair_vectors(gt, pred, tolerance=100.0)
    assert not pairing.paired


def test_vector_count_stats():
    gt = [_line((0, 0), (10, 0)), _line((0, 0), (0, 10))]
    pred = [_line((0, 0), (10, 0)), _line((100, 100), (200, 200))]
    pairing = pair_vectors(gt, pred, vector_type="original_vector", tolerance=0.1)
    stats = vector_count_stats(pairing)
    assert stats.n_paired == 1
    assert stats.n_missed == 1
    assert stats.n_spurious == 1
    assert stats.precision.value == 0.5
    assert stats.recall.value == 0.5


def test_endpoint_accuracy_stats_zero_for_exact_match():
    gt = [_line((0, 0), (10, 0))]
    pred = [_line((0, 0), (10, 0))]
    pairing = pair_vectors(gt, pred, tolerance=1.0)
    stats = endpoint_accuracy_stats(pairing, gt, pred)
    assert stats.rmse == 0.0
    assert stats.n_paired == 1


def test_property_accuracy_table_none_vs_none_excluded_but_counted():
    gt = [_line((0, 0), (10, 0), color=None)]
    pred = [_line((0, 0), (10, 0), color=None)]
    pairing = pair_vectors(gt, pred, vector_type="vector_to_raster", tolerance=1.0)
    rows = property_accuracy_table(pairing, gt, pred, "vector_to_raster")
    color_row = next(r for r in rows if r.property_name == "color")
    assert color_row.n_applicable == 0
    assert color_row.none_vs_none_count == 1


def test_property_accuracy_table_full_vector_only_rows_blank_for_raster_types():
    gt = [_line((0, 0), (10, 0))]
    pred = [_line((0, 0), (10, 0))]
    pairing = pair_vectors(gt, pred, vector_type="vector_to_raster", tolerance=1.0)
    rows = property_accuracy_table(pairing, gt, pred, "vector_to_raster")
    close_path_row = next(r for r in rows if r.property_name == "close_path")
    assert close_path_row.applicable is False


def test_property_accuracy_table_discrete_exact_match_rate():
    gt = [_line((0, 0), (10, 0), dashes="[] 0"), _line((0, 0), (0, 10), dashes="[2] 0")]
    pred = [_line((0, 0), (10, 0), dashes="[] 0"), _line((0, 0), (0, 10), dashes="[] 0")]
    pairing = pair_vectors(gt, pred, vector_type="original_vector", tolerance=1.0)
    rows = property_accuracy_table(pairing, gt, pred, "original_vector")
    dashes_row = next(r for r in rows if r.property_name == "dashes")
    assert dashes_row.metric_value == 0.5
