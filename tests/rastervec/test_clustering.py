from __future__ import annotations

from rastervec.helpers.clustering import Clustering


def _item(x0, y0, x1, y1, seq=0):
    return {"bbox": (x0, y0, x1, y1), "seq": seq}


def _bbox(item):
    return item["bbox"]


def _seq(item):
    return item["seq"]


def test_cluster_spatial_merges_close_items():
    clustering = Clustering()
    items = [_item(0, 0, 1, 1), _item(1.5, 0, 2.5, 1)]  # gap = 0.5

    clusters = clustering.cluster_spatial(items, _bbox, threshold=1.0)

    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_cluster_spatial_keeps_far_items_separate():
    clustering = Clustering()
    items = [_item(0, 0, 1, 1), _item(100, 100, 101, 101)]

    clusters = clustering.cluster_spatial(items, _bbox, threshold=1.0)

    assert len(clusters) == 2


def test_cluster_spatial_transitive_chain():
    clustering = Clustering()
    # a-b close, b-c close, a-c far -- single-linkage should still merge all 3
    items = [
        _item(0, 0, 1, 1),
        _item(1.5, 0, 2.5, 1),
        _item(3.0, 0, 4.0, 1),
    ]

    clusters = clustering.cluster_spatial(items, _bbox, threshold=1.0)

    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_cluster_spatial_empty_input():
    assert Clustering().cluster_spatial([], _bbox, threshold=1.0) == []


def test_cluster_by_dimension_splits_different_sizes():
    clustering = Clustering()
    group = [
        _item(0, 0, 1, 1),
        _item(2, 0, 3, 1),  # same size as first
        _item(4, 0, 24, 1),  # much wider
    ]

    clusters = clustering.cluster_by_dimension([group], _bbox, tolerance=0.1)

    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]


def test_cluster_by_seq_splits_on_gap():
    clustering = Clustering()
    group = [
        _item(0, 0, 1, 1, seq=0),
        _item(0, 0, 1, 1, seq=1),
        _item(0, 0, 1, 1, seq=10),
    ]

    clusters = clustering.cluster_by_seq([group], _seq, max_gap=3)

    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]


def test_cluster_by_seq_single_item_group_passthrough():
    clustering = Clustering()
    group = [_item(0, 0, 1, 1, seq=5)]

    clusters = clustering.cluster_by_seq([group], _seq, max_gap=3)

    assert clusters == [group]
