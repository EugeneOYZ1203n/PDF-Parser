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


def test_cluster_spatial_threshold_zero_and_group_by_overlap_agree_on_overlap():
    """`cluster_spatial(items, get_bbox, threshold=0.0)` has a degenerate
    1e-6-sized grid cell, so every realistically-sized bbox exceeds
    `_MAX_CELLS_PER_ITEM` and goes through the direct big-item check -- it
    used to be collapsed to its centre cell instead, which missed true
    overlaps. Both functions now merge an overlapping pair (`group_by_overlap`
    remains the fast tool for pure overlap)."""
    a, b = _item(0.0, 0.0, 5.0, 5.0), _item(3.0, 0.0, 8.0, 5.0)  # overlap

    assert len(cluster_spatial([a, b], _bbox, threshold=0.0)) == 1
    assert len(group_by_overlap([a, b], _bbox)) == 1


def _cluster_sets(clusters):
    return sorted(sorted(i["seq"] for i in c) for c in clusters)


def test_cluster_spatial_merges_a_small_item_at_a_large_items_edge():
    # 1000 x 1000 spans ~10,000 cells at threshold 10 -- far over the cap. The
    # small box sits 3 pt off its right edge, ~500 pt from its centre.
    big = _item(0.0, 0.0, 1000.0, 1000.0, seq=0)
    small = _item(1003.0, 100.0, 1008.0, 105.0, seq=1)
    far = _item(2000.0, 100.0, 2005.0, 105.0, seq=2)

    assert _cluster_sets(cluster_spatial([big, small, far], _bbox, threshold=10.0)) == [[0, 1], [2]]
    # Order-independent: the big item last.
    assert _cluster_sets(cluster_spatial([small, far, big], _bbox, threshold=10.0)) == [[0, 1], [2]]


def test_cluster_spatial_merges_two_large_items():
    a = _item(0.0, 0.0, 1000.0, 1000.0, seq=0)
    b = _item(1005.0, 0.0, 2000.0, 1000.0, seq=1)
    c = _item(5000.0, 0.0, 6000.0, 1000.0, seq=2)

    assert _cluster_sets(cluster_spatial([a, b, c], _bbox, threshold=10.0)) == [[0, 1], [2]]


def test_cluster_spatial_big_item_path_honours_extra_close_and_area_cap():
    big = _item(0.0, 0.0, 1000.0, 1000.0, seq=0)
    small = _item(1003.0, 100.0, 1008.0, 105.0, seq=1)

    vetoed = cluster_spatial([big, small], _bbox, threshold=10.0, extra_close=lambda _a, _b: False)
    assert _cluster_sets(vetoed) == [[0], [1]]
    capped = cluster_spatial([big, small], _bbox, threshold=10.0, max_union_area=1_000_000.0)
    assert _cluster_sets(capped) == [[0], [1]]  # union would be 1008 x 1000
    allowed = cluster_spatial([big, small], _bbox, threshold=10.0, max_union_area=2_000_000.0)
    assert _cluster_sets(allowed) == [[0, 1]]


def test_cluster_spatial_matches_brute_force_single_linkage():
    import random

    from rastervec.commons.helpers.geometry import rect_gap

    rng = random.Random(4)
    for _trial in range(30):
        items = []
        for k in range(rng.randint(2, 60)):
            x, y = rng.uniform(0, 3000), rng.uniform(0, 3000)
            if rng.random() < 0.15:  # big enough for the direct path
                w, h = rng.uniform(450, 1500), rng.uniform(450, 1500)
            else:
                w, h = rng.uniform(0, 40), rng.uniform(0, 40)
            items.append(_item(x, y, x + w, y + h, seq=k))
        threshold = rng.choice([5.0, 10.0, 25.0])

        parent = list(range(len(items)))

        def find(i):
            while parent[i] != i:
                i = parent[i]
            return i

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if rect_gap(items[i]["bbox"], items[j]["bbox"]) <= threshold:
                    parent[find(i)] = find(j)
        groups: dict = {}
        for i, it in enumerate(items):
            groups.setdefault(find(i), []).append(it)

        assert _cluster_sets(cluster_spatial(items, _bbox, threshold=threshold)) == _cluster_sets(groups.values())
