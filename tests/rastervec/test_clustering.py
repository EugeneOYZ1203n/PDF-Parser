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


def test_cluster_by_pairwise_distance_matches_cluster_by_dimension():
    clustering = Clustering()
    groups = [[_item(0, 0, 10, 10), _item(0, 0, 13, 13), _item(0, 0, 100, 5)]]

    def dimension_distance(a, b):
        ax0, ay0, ax1, ay1 = _bbox(a)
        bx0, by0, bx1, by1 = _bbox(b)
        aw, ah = ax1 - ax0, ay1 - ay0
        bw, bh = bx1 - bx0, by1 - by0
        return max(abs(aw - bw) / max(aw, bw, 1e-6), abs(ah - bh) / max(ah, bh, 1e-6))

    expected = clustering.cluster_by_dimension(groups, _bbox, tolerance=0.35)
    actual = clustering.cluster_by_pairwise_distance(groups, dimension_distance, threshold=0.35)

    assert sorted(map(len, actual)) == sorted(map(len, expected))
    assert len(actual) == 2  # the two ~similar boxes merge; the thin one stays separate


def test_cluster_by_pairwise_distance_empty_group_list():
    clustering = Clustering()
    assert clustering.cluster_by_pairwise_distance([], lambda a, b: 0.0, threshold=1.0) == []


def test_group_by_overlap_merges_partial_overlap():
    a = _item(0, 0, 10, 10)
    b = _item(5, 5, 15, 15)  # partial overlap with a

    groups = Clustering().group_by_overlap([[a, b]], _bbox)

    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_group_by_overlap_keeps_full_containment_separate():
    outer = _item(0, 0, 10, 10)
    inner = _item(2, 2, 4, 4)  # fully inside outer -> not merged

    groups = Clustering().group_by_overlap([[outer, inner]], _bbox)

    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 1]


def test_group_by_overlap_keeps_disjoint_separate():
    a = _item(0, 0, 1, 1)
    b = _item(100, 100, 101, 101)

    groups = Clustering().group_by_overlap([[a, b]], _bbox)

    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 1]


def test_group_by_overlap_default_tolerance_keeps_near_miss_separate():
    a = _item(0, 0, 10, 10)
    b = _item(12, 0, 20, 10)  # gap = 2, no tolerance requested -> stays separate

    groups = Clustering().group_by_overlap([[a, b]], _bbox)

    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 1]


def test_group_by_overlap_tolerance_merges_near_miss():
    a = _item(0, 0, 10, 10)
    b = _item(12, 0, 20, 10)  # gap = 2 <= tolerance -> merges

    groups = Clustering().group_by_overlap([[a, b]], _bbox, tolerance=3.0)

    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_group_by_overlap_tolerance_still_excludes_full_containment():
    outer = _item(0, 0, 10, 10)
    inner = _item(2, 2, 4, 4)  # fully inside outer -> never merges, any tolerance

    groups = Clustering().group_by_overlap([[outer, inner]], _bbox, tolerance=100.0)

    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 1]
