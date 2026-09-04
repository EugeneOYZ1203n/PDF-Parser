from __future__ import annotations

from rastervec.helpers.clustering import cluster_spatial


def _item(x0, y0, x1, y1, seq=0):
    return {"bbox": (x0, y0, x1, y1), "seq": seq}


def _bbox(item):
    return item["bbox"]


def test_cluster_spatial_merges_close_items():
    items = [_item(0, 0, 1, 1), _item(1.5, 0, 2.5, 1)]  # gap = 0.5

    clusters = cluster_spatial(items, _bbox, threshold=1.0)

    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_cluster_spatial_keeps_far_items_separate():
    items = [_item(0, 0, 1, 1), _item(100, 100, 101, 101)]

    clusters = cluster_spatial(items, _bbox, threshold=1.0)

    assert len(clusters) == 2


def test_cluster_spatial_transitive_chain():
    # a-b close, b-c close, a-c far -- single-linkage should still merge all 3
    items = [
        _item(0, 0, 1, 1),
        _item(1.5, 0, 2.5, 1),
        _item(3.0, 0, 4.0, 1),
    ]

    clusters = cluster_spatial(items, _bbox, threshold=1.0)

    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_cluster_spatial_empty_input():
    assert cluster_spatial([], _bbox, threshold=1.0) == []


def test_cluster_spatial_extra_close_gate():
    # two overlapping items that extra_close rejects -> stay separate
    a = _item(0, 0, 10, 10, seq=1)
    b = _item(5, 5, 15, 15, seq=2)

    merged = cluster_spatial(
        [a, b], _bbox, threshold=5.0,
        extra_close=lambda x, y: x["seq"] == y["seq"],
    )
    assert sorted(len(c) for c in merged) == [1, 1]

    unmerged = cluster_spatial([a, b], _bbox, threshold=5.0)
    assert len(unmerged) == 1
