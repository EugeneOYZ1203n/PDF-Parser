import numpy as np

from rastervec.P3_Vector_Parsing.FastIntoPaddle.similarity import (
    item_type_signature,
    normalize_vector,
    point_cloud_mse,
    vector_similarity_group,
)


def test_item_type_signature_matches_same_shaped_items(vector):
    a = vector(items=[("l", (0.0, 0.0), (1.0, 1.0)), ("l", (1.0, 1.0), (2.0, 0.0))])
    b = vector(items=[("l", (5.0, 5.0), (6.0, 6.0)), ("l", (6.0, 6.0), (7.0, 5.0))])
    assert item_type_signature(a) == item_type_signature(b)


def test_item_type_signature_differs_for_different_item_counts(vector):
    a = vector(items=[("l", (0.0, 0.0), (1.0, 1.0))])
    b = vector(items=[("l", (0.0, 0.0), (1.0, 1.0)), ("l", (1.0, 1.0), (2.0, 0.0))])
    assert item_type_signature(a) != item_type_signature(b)


def test_normalize_vector_is_translation_invariant(vector):
    a = vector(items=[("l", (0.0, 0.0), (10.0, 0.0)), ("l", (10.0, 0.0), (10.0, 5.0))])
    b = vector(items=[("l", (100.0, 100.0), (110.0, 100.0)), ("l", (110.0, 100.0), (110.0, 105.0))])
    na, nb = normalize_vector(a), normalize_vector(b)
    assert na.shape == nb.shape
    assert point_cloud_mse(na, nb) == pytest_approx_zero()


def test_normalize_vector_is_scale_invariant(vector):
    a = vector(items=[("l", (0.0, 0.0), (10.0, 0.0)), ("l", (10.0, 0.0), (10.0, 5.0))])
    b = vector(items=[("l", (0.0, 0.0), (100.0, 0.0)), ("l", (100.0, 0.0), (100.0, 50.0))])
    na, nb = normalize_vector(a), normalize_vector(b)
    assert point_cloud_mse(na, nb) == pytest_approx_zero()


def test_normalize_vector_degenerate_zero_length_line(vector):
    """A zero-length line ("l" with identical endpoints) has a singular
    covariance -- normalize_vector must not raise or return NaNs."""
    v = vector(items=[("l", (3.0, 3.0), (3.0, 3.0))])
    norm = normalize_vector(v)
    assert norm.shape[1] == 2
    assert np.all(np.isfinite(norm))


def test_point_cloud_mse_zero_for_identical_clouds():
    a = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert point_cloud_mse(a, a) == 0.0


def test_point_cloud_mse_positive_for_different_clouds():
    a = np.array([[0.0, 0.0], [1.0, 1.0]])
    b = np.array([[0.0, 0.0], [2.0, 2.0]])
    assert point_cloud_mse(a, b) > 0.0


def test_point_cloud_mse_empty_is_infinite():
    empty = np.zeros((0, 2))
    assert point_cloud_mse(empty, empty) == float("inf")


def test_vector_similarity_group_groups_same_shape_vectors(vector):
    # Two same-shaped L's (translated/scaled copies of each other) and one
    # unrelated shape (different item-type signature).
    l1 = vector(items=[("l", (0.0, 0.0), (10.0, 0.0)), ("l", (10.0, 0.0), (10.0, 10.0))])
    l2 = vector(items=[("l", (50.0, 50.0), (150.0, 50.0)), ("l", (150.0, 50.0), (150.0, 150.0))])
    curve = vector(items=[("c", (0.0, 0.0), (1.0, 2.0), (3.0, 2.0), (4.0, 0.0))])

    groups = vector_similarity_group([l1, l2, curve], mse_threshold=0.01)

    assert len(groups) == 2
    sizes = sorted(len(g.members) for g in groups)
    assert sizes == [1, 2]


def test_vector_similarity_group_separates_different_shapes(vector):
    l_shape = vector(items=[("l", (0.0, 0.0), (10.0, 0.0)), ("l", (10.0, 0.0), (10.0, 10.0))])
    t_shape = vector(items=[("l", (0.0, 0.0), (10.0, 0.0)), ("l", (5.0, 0.0), (5.0, 10.0))])

    groups = vector_similarity_group([l_shape, t_shape], mse_threshold=0.001)

    assert len(groups) == 2


def pytest_approx_zero():
    import pytest

    return pytest.approx(0.0, abs=1e-6)
