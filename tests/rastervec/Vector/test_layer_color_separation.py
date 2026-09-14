from __future__ import annotations

from rastervec.Vector.layer_color_separation import separate_by_width


def test_separate_by_width_groups_by_exact_width(vector):
    v1 = vector(width=1.0)
    v2 = vector(width=1.0)
    v3 = vector(width=2.5)

    groups = separate_by_width([v1, v2, v3])

    assert set(groups.keys()) == {1.0, 2.5}
    assert groups[1.0] == [v1, v2]
    assert groups[2.5] == [v3]


def test_separate_by_width_groups_none_width(vector):
    v1 = vector(width=None)
    v2 = vector(width=1.0)

    groups = separate_by_width([v1, v2])

    assert groups[None] == [v1]
    assert groups[1.0] == [v2]
