"""`commons.helpers.clustering.UnionFind`."""

from raster_parser.commons.helpers.clustering import UnionFind


def test_union_find_joins_transitively():
    uf = UnionFind(4)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2) != uf.find(3)
