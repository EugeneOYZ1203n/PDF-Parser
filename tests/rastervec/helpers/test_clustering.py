from __future__ import annotations

from rastervec.commons.helpers.clustering import cluster_spatial, group_by_overlap


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


def test_group_by_overlap_merges_touching_boxes():
    a, b = _item(0.0, 0.0, 5.0, 5.0), _item(3.0, 0.0, 8.0, 5.0)  # overlap
    clusters = group_by_overlap([a, b], _bbox)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_group_by_overlap_keeps_non_overlapping_boxes_separate():
    a, b = _item(0.0, 0.0, 5.0, 5.0), _item(20.0, 0.0, 25.0, 5.0)  # no overlap
    clusters = group_by_overlap([a, b], _bbox)
    assert len(clusters) == 2


def test_group_by_overlap_transitive_chain():
    a = _item(0.0, 0.0, 5.0, 5.0)
    b = _item(4.0, 0.0, 9.0, 5.0)  # overlaps a
    c = _item(8.0, 0.0, 13.0, 5.0)  # overlaps b, not a
    clusters = group_by_overlap([a, b, c], _bbox)
    assert len(clusters) == 1
    assert len(clusters[0]) == 3


def test_group_by_overlap_empty_input():
    assert group_by_overlap([], _bbox) == []


def test_group_by_overlap_correctly_merges_where_cluster_spatial_threshold_zero_would_not():
    """`cluster_spatial(items, get_bbox, threshold=0.0)` looks like it
    should express "pure overlap", but its grid cell size is `max(threshold,
    1e-6)` -- a zero threshold collapses to a degenerate 1e-6-sized cell,
    which trips `cluster_spatial`'s own `_MAX_CELLS_PER_ITEM` cap and
    silently reduces every realistically-sized bbox down to its centroid
    alone, missing true overlaps between items whose centroids land in
    different cells. `group_by_overlap` exists specifically to do this
    correctly -- this test pins that difference down."""
    a, b = _item(0.0, 0.0, 5.0, 5.0), _item(3.0, 0.0, 8.0, 5.0)  # overlap

    broken = cluster_spatial([a, b], _bbox, threshold=0.0)
    assert len(broken) == 2  # the bug: overlapping boxes NOT merged

    correct = group_by_overlap([a, b], _bbox)
    assert len(correct) == 1  # group_by_overlap gets it right
