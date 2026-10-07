from __future__ import annotations

import math
import random

from rastervec.P3_Vector_Parsing.LatestVectorClassification import pattern_lattice as pl
from rastervec.P3_Vector_Parsing.LatestVectorClassification.config import (
    PATTERN_MAX_GROUP,
    PATTERN_MIN_BUCKET,
)

# grid side whose lattice group is past PATTERN_MAX_GROUP (the drawing size threshold)
_N = math.isqrt(PATTERN_MAX_GROUP) + 1


def _tick(vector, x, y, dx=1.0, dy=0.0, seqno=0):
    return vector(bbox=(min(x, x + dx), min(y, y + dy), max(x, x + dx), max(y, y + dy)),
                  items=[("l", (x, y), (x + dx, y + dy))], seqno=seqno)


def _grid(vector, nx, ny, x0=0.0, y0=0.0, step=(5.0, 4.0), seq0=0, jitter=0.0, rng=None):
    out = []
    for j in range(ny):
        for i in range(nx):
            jx = rng.uniform(-jitter, jitter) if rng else 0.0
            jy = rng.uniform(-jitter, jitter) if rng else 0.0
            out.append(_tick(vector, x0 + i * step[0] + jx, y0 + j * step[1] + jy, seqno=seq0 + len(out)))
    return out


def test_similarity_key_is_rotation_invariant(vector):
    assert pl.similarity_key(_tick(vector, 0, 0, 1, 0)) == pl.similarity_key(_tick(vector, 5, 5, 0, 1))
    assert pl.similarity_key(_tick(vector, 0, 0, 1, 0)) != pl.similarity_key(_tick(vector, 0, 0, 5, 0))


def test_similarity_key_item_order_matters(vector):
    a = vector(items=[("l", (0, 0), (1, 0)), ("l", (1, 0), (1, 7))])
    b = vector(items=[("l", (0, 0), (0, 7)), ("l", (0, 7), (1, 7))])
    assert pl.similarity_key(a) != pl.similarity_key(b)


def test_similarity_key_hashes_lengths_into_buckets(vector):
    # same 3 pt bucket -> similar; across a bucket boundary -> not (accepted)
    assert pl.similarity_key(_tick(vector, 0, 0, 1.0)) == pl.similarity_key(_tick(vector, 0, 0, 2.9))
    assert pl.similarity_key(_tick(vector, 0, 0, 2.9)) != pl.similarity_key(_tick(vector, 0, 0, 3.1))


def test_anchor_per_kind(vector):
    assert pl.anchor(vector(items=[("re", (2.0, 3.0, 5.0, 7.0))])) == (2.0, 3.0)
    assert pl.anchor(vector(items=[("qu", ((1, 2), (3, 2), (3, 4), (1, 4)))])) == (1, 2)
    assert pl.anchor(vector(items=[])) is None


def test_grid_dropped_as_one_group(vector):
    grid = _grid(vector, _N, _N)
    other = _tick(vector, 100, 100, 7, 7)
    kept, groups = pl.pattern_drawing(grid + [other])
    assert kept == [other]
    assert len(groups) == 1 and len(groups[0]) == _N * _N


def test_rotated_motif_shares_bucket(vector):
    grid = _grid(vector, 6, 6)
    rotated = [_tick(vector, 200 + 3 * i, 0, 0, 1) for i in range(3)]
    buckets = pl.group_by_similarity(grid + rotated)
    assert len(buckets) == 1 and len(buckets[0]) == 39


def test_1d_row_dropped(vector):
    row = [_tick(vector, 3.0 * i, 0, 1, 0, seqno=i) for i in range(PATTERN_MAX_GROUP + 5)]
    kept, groups = pl.pattern_drawing(row)
    assert kept == [] and len(groups) == 1


def test_small_bucket_untouched(vector):
    row = [_tick(vector, 3.0 * i, 0) for i in range(PATTERN_MIN_BUCKET)]
    kept, groups = pl.pattern_drawing(row)
    assert kept == row and groups == []


def test_separated_patches_are_separate_groups(vector):
    a = _grid(vector, 5, 5)
    b = _grid(vector, 5, 5, x0=500.0, y0=500.0, seq0=100)
    groups = pl.ordered_groups(a + b)
    assert sorted(len(g) for g in groups) == [25, 25]


def test_coincident_duplicates_absorbed(vector):
    grid = _grid(vector, _N, _N)
    dupes = [_tick(vector, 0, 0, seqno=99), _tick(vector, 10, 8, seqno=98)]
    kept, groups = pl.pattern_drawing(grid + dupes)
    assert kept == [] and len(groups) == 1 and len(groups[0]) == _N * _N + 2


def test_small_groups_kept(vector):
    # 3 separated 3x3 patches: 27 similar Vectors, but every lattice group has 9 members
    patches = [_grid(vector, 3, 3, x0=300.0 * k, seq0=10 * k) for k in range(3)]
    flat = [v for p in patches for v in p]
    assert len(flat) > PATTERN_MIN_BUCKET and 9 <= PATTERN_MAX_GROUP
    kept, groups = pl.pattern_drawing(flat)
    assert kept == flat and groups == []


def test_jittered_lattice_found(vector):
    grid = _grid(vector, _N, _N, jitter=0.1, rng=random.Random(0))
    _kept, groups = pl.pattern_drawing(grid)
    assert sum(len(g) for g in groups) == _N * _N


def test_deterministic(vector):
    grid = _grid(vector, 7, 5, jitter=0.1, rng=random.Random(1))
    first = [[id(v) for v in g] for g in pl.ordered_groups(grid)]
    second = [[id(v) for v in g] for g in pl.ordered_groups(grid)]
    assert first == second


def test_off_grid_member_rejected(vector):
    grid = _grid(vector, 6, 6, step=(20.0, 20.0))
    grid[14] = _tick(vector, 2 * 20.0 + 3.5, 2 * 20.0, seqno=14)  # 3.5 pt off its grid point
    groups = pl.ordered_groups(grid)
    assert len(groups[0]) == 35 and grid[14] not in groups[0]


def test_gap_stops_1d_flood(vector):
    row = [_tick(vector, 5.0 * i, 0, seqno=i) for i in range(30) if i != 15]
    groups = pl.ordered_groups(row)
    assert sorted(len(g) for g in groups) == [14, 15]


def test_global_drift_splits_row(vector):
    xs, x, step = [], 0.0, 5.0
    for _ in range(40):
        xs.append(x)
        x += step
        step *= 1.03  # every hop locally fine, the row as a whole is no lattice
    row = [_tick(vector, xi, 0, seqno=k) for k, xi in enumerate(xs)]
    assert max(len(g) for g in pl.ordered_groups(row)) < len(row)


def test_noisy_first_pair_refitted(vector):
    row = [_tick(vector, 5.0 * i, 0, seqno=i) for i in range(PATTERN_MIN_BUCKET + 5)]
    row[1] = _tick(vector, 5.15, 0, seqno=1)  # the seed's v1 is 3% too long
    groups = pl.ordered_groups(row)
    assert len(groups) == 1 and len(groups[0]) == len(row)
    grid = _grid(vector, 8, 8)
    grid[1] = _tick(vector, 5.15, 0, seqno=1)
    assert [len(g) for g in pl.ordered_groups(grid)] == [64]


def _blockers(vector, n, x=2.0, y=0.0):
    """n tiny rects (not similar to the grid's ticks) across the link at y."""
    out = []
    for k in range(n):
        r = (x + 0.8 * k, y - 0.15, x + 0.8 * k + 0.3, y + 0.15)
        out.append(vector(bbox=r, items=[("re", r)], seqno=1000 + k))
    return out


def test_content_between_members_keeps_group(vector):
    grid = _grid(vector, _N, _N)
    blockers = _blockers(vector, 3)
    kept, groups = pl.pattern_drawing(grid + blockers)
    assert groups == [] and len(kept) == len(grid) + 3


def test_little_content_between_members_still_drawing(vector):
    grid = _grid(vector, _N, _N)
    blockers = _blockers(vector, 2)
    kept, groups = pl.pattern_drawing(grid + blockers)
    assert kept == blockers and len(groups) == 1 and len(groups[0]) == _N * _N


def test_content_outside_links_ignored(vector):
    grid = _grid(vector, _N, _N)
    far = _blockers(vector, 5, x=200.0, y=200.0)
    big = vector(bbox=(-50.0, -50.0, 100.0, 100.0), items=[("re", (-50.0, -50.0, 100.0, 100.0))], seqno=2000)
    kept, groups = pl.pattern_drawing(grid + far + [big])
    assert len(groups) == 1 and len(groups[0]) == _N * _N and len(kept) == 6


def test_stacked_duplicates_stay_singletons(vector):
    stack = [_tick(vector, 0, 0, seqno=k) for k in range(PATTERN_MIN_BUCKET + 10)]
    assert [len(g) for g in pl.ordered_groups(stack)] == [1] * len(stack)
    kept, groups = pl.pattern_drawing(stack)
    assert kept == stack and groups == []


def test_stacked_duplicates_dont_hide_lattice(vector):
    # more duplicates at the seed than PATTERN_KNN: the basis query reaches past them
    grid = _grid(vector, _N, _N)
    stack = [_tick(vector, 0, 0, seqno=-1 - k) for k in range(40)]
    _kept, groups = pl.pattern_drawing(stack + grid)
    assert len(groups) == 1 and len(groups[0]) == _N * _N + 40
