from __future__ import annotations

import pymupdf as fitz
import pytest

from rastervec.helpers import vector_classification as vc
from rastervec.helpers.clustering import Clustering
from rastervec.models import VectorPath
from rastervec.reader import Reader
from rastervec.vector import Vector, _is_dashed


def _make_path(
    *,
    seq=0,
    item_index=0,
    kind="re",
    bbox=(0, 0, 1, 1),
    fill_rule="",
    stroke_color=None,
    fill_color=None,
    stroke_opacity=None,
    fill_opacity=None,
    stroke_width=None,
    dashes=None,
    closed=None,
    layer=None,
    page_index=0,
) -> VectorPath:
    return VectorPath(
        seq=seq,
        item_index=item_index,
        kind=kind,
        fill_rule=fill_rule,
        points=[(bbox[0], bbox[1]), (bbox[2], bbox[3])],
        bbox=bbox,
        stroke_color=stroke_color,
        fill_color=fill_color,
        stroke_opacity=stroke_opacity,
        fill_opacity=fill_opacity,
        stroke_width=stroke_width,
        dashes=dashes,
        closed=closed,
        layer=layer,
        page_index=page_index,
    )


def _build_test_page(tmp_pdf_path) -> "Reader":
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)

    bg = page.new_shape()
    bg.draw_rect(fitz.Rect(0, 0, 200, 200))
    bg.finish(color=None, fill=(1, 1, 1))
    bg.commit()

    for i in range(4):
        shape = page.new_shape()
        shape.draw_rect(fitz.Rect(10 + i * 5, 10, 10 + i * 5 + 3, 16))
        shape.finish(color=None, fill=(0, 0, 0))
        shape.commit()

    line = page.new_shape()
    line.draw_line((100, 100), (180, 180))
    line.finish(color=(0, 0, 0), width=2)
    line.commit()

    path = tmp_pdf_path(doc)
    return Reader(path)


def test_extract_paths_basic(tmp_pdf_path):
    with _build_test_page(tmp_pdf_path) as reader:
        page = reader.get_page(0)
        paths = Vector().extract_paths(page)

    assert len(paths) > 0
    kinds = {p.kind for p in paths}
    assert kinds <= {"l", "re", "qu", "c"}
    assert all(p.page_index == 0 for p in paths)


def test_extract_paths_rect_bbox_matches(tmp_pdf_path):
    with _build_test_page(tmp_pdf_path) as reader:
        page = reader.get_page(0)
        paths = Vector().extract_paths(page)

    small_rects = [p for p in paths if p.kind == "re" and (p.bbox[2] - p.bbox[0]) < 10]
    assert len(small_rects) == 4
    first = sorted(small_rects, key=lambda p: p.bbox[0])[0]
    assert first.bbox == pytest.approx((10, 10, 13, 16))
    assert first.fill_color == (0.0, 0.0, 0.0)


def test_separate_by_layer_groups_by_layer_field():
    paths = [
        _make_path(layer="A"),
        _make_path(layer="A"),
        _make_path(layer="B"),
        _make_path(layer=None),
    ]

    groups = Vector().separate_by_layer(paths)

    assert set(groups.keys()) == {"A", "B", ""}
    assert len(groups["A"]) == 2
    assert len(groups["B"]) == 1
    assert len(groups[""]) == 1


def test_separate_by_color_groups_by_stroke_fill_and_opacity():
    paths = [
        _make_path(stroke_color=(1, 0, 0), fill_color=(0, 1, 0)),
        _make_path(stroke_color=None, fill_color=(0, 1, 0)),
        _make_path(stroke_color=None, fill_color=None),
        # same stroke/fill as paths[0] but different opacity -> its own group.
        _make_path(stroke_color=(1, 0, 0), fill_color=(0, 1, 0), stroke_opacity=0.5),
    ]

    groups = Vector().separate_by_color(paths)

    assert groups[(1, 0, 0), (0, 1, 0), None, None] == [paths[0]]
    assert groups[None, (0, 1, 0), None, None] == [paths[1]]
    assert groups[None, None, None, None] == [paths[2]]
    assert groups[(1, 0, 0), (0, 1, 0), 0.5, None] == [paths[3]]


class _Meta:
    width = 200
    height = 200


class _Page:
    meta = _Meta()


@pytest.mark.parametrize(
    "dashes, expected",
    [
        (None, False),
        ("", False),
        ("[] 0", False),
        ("[3 2] 0", True),
    ],
)
def test_is_dashed(dashes, expected):
    assert _is_dashed(dashes) is expected


# ----------------------------------------------------------------------
# Vector.cluster() -- pipeline shape / step-level behavior.
# ----------------------------------------------------------------------


def test_cluster_step_count_and_labels():
    lone = _make_path(kind="l", bbox=(0, 0, 3, 6), fill_color=(0, 0, 0))

    steps = Vector().cluster([lone], _Page())

    assert [s.label for s in steps] == [
        "Large items",
        "Vector signatures",
        "Seq dedupe + overlap merge",
        "Tiny groups",
        "Large groups",
        "Spatial cluster",
        "Mixed fill-rule clusters",
        "Group stats",
        "Perimeter-only clusters",
        "Density clusters",
        "High-frequency clusters",
        "Constant-spacing clusters",
        "Low-variety clusters",
    ]


def test_classify_is_cluster_final_kept_category():
    lone = _make_path(kind="l", bbox=(0, 0, 3, 6), fill_color=(0, 0, 0))

    steps = Vector().cluster([lone], _Page())
    assert steps[-1].categories["kept"].groups == Vector().classify([lone], _Page())


def test_classify_drops_trivial_lone_cluster():
    # A single-path cluster is dropped somewhere along the chain (which
    # exact step depends on the current threshold tuning -- e.g. low
    # variety no longer unconditionally rejects it, since the required
    # unique-signature count now scales down for small clusters too).
    lone = _make_path(kind="l", bbox=(0, 0, 3, 6), fill_color=(0, 0, 0))

    clusters = Vector().classify([lone], _Page())

    assert clusters == []


def test_cluster_drops_oversized_item_and_group():
    # max dimension 40, 20% of the 200x200 page's smaller side -- exceeds
    # MAX_DIMENSION_FRACTION (10%), so filter_large_items drops it at step 1.
    oversized = _make_path(kind="l", bbox=(0, 0, 40, 40), stroke_color=(0, 0, 0))
    small = _make_path(seq=50, kind="l", bbox=(150, 150, 152, 152), stroke_color=(0, 0, 0))

    steps = Vector().cluster([oversized, small], _Page())

    dropped_at_large_items = steps[0].categories["dropped_oversized"].groups
    assert any(oversized in g for g in dropped_at_large_items)
    assert all(small not in g for g in dropped_at_large_items)
    kept_after_large_items = [p for g in steps[0].categories["kept"].groups for p in g]
    assert kept_after_large_items == [small]


def test_cluster_drops_duplicate_run_of_five_or_more():
    # 5 identical rects (same shape+size, i.e. same vector_signature) --
    # DUPLICATE_RUN_MIN_LENGTH (5) worth of consecutive seq -- get dropped
    # as a run at the seq-dedupe step; a differently-shaped item passes
    # through untouched.
    duplicates = [
        _make_path(seq=i, kind="re", bbox=(i * 20, 0, i * 20 + 3, 6), fill_color=(0, 0, 0))
        for i in range(5)
    ]
    other = _make_path(seq=50, kind="l", bbox=(150, 150, 158, 158), stroke_color=(0, 0, 0))

    steps = Vector().cluster(duplicates + [other], _Page())
    dedupe_step = steps[2]
    assert dedupe_step.label == "Seq dedupe + overlap merge"

    dropped_paths = [p for g in dedupe_step.categories["duplicate_runs"].groups for p in g]
    assert {id(p) for p in dropped_paths} == {id(p) for p in duplicates}

    kept_paths = [p for g in dedupe_step.categories["kept"].groups for p in g]
    assert other in kept_paths
    assert all(d not in kept_paths for d in duplicates)


def test_cluster_keeps_duplicate_run_under_minimum():
    quad = [
        _make_path(seq=i, kind="re", bbox=(i * 20, 0, i * 20 + 3, 6), fill_color=(0, 0, 0))
        for i in range(4)
    ]

    steps = Vector().cluster(quad, _Page())
    dedupe_step = steps[2]

    assert dedupe_step.categories["duplicate_runs"].groups == []
    kept_paths = [p for g in dedupe_step.categories["kept"].groups for p in g]
    assert {id(p) for p in kept_paths} == {id(p) for p in quad}


def test_cluster_drops_entire_run_longer_than_minimum():
    # 12 consecutive duplicates -- more than DUPLICATE_RUN_MIN_LENGTH (5) --
    # are dropped in their entirety (no partial keep within one run), and a
    # trailing, differently-shaped item that follows the run passes through.
    duplicates = [
        _make_path(seq=i, kind="re", bbox=(i * 20, 0, i * 20 + 3, 6), fill_color=(0, 0, 0))
        for i in range(12)
    ]
    trailing = _make_path(seq=50, kind="l", bbox=(300, 0, 308, 8), stroke_color=(0, 0, 0))

    steps = Vector().cluster(duplicates + [trailing], _Page())
    dedupe_step = steps[2]

    dropped_paths = [p for g in dedupe_step.categories["duplicate_runs"].groups for p in g]
    assert {id(p) for p in dropped_paths} == {id(p) for p in duplicates}

    kept_paths = [p for g in dedupe_step.categories["kept"].groups for p in g]
    assert kept_paths == [trailing]


def test_cluster_spatial_step_reports_debug_categories():
    # A is 18 wide x 4 tall; B is 4 wide x 18 tall, close by (gap 4px, well
    # under SPATIAL_CLUSTER_THRESHOLD (16)). Their length magnitudes match
    # exactly (18 == 18), but A's length runs horizontally while B's runs
    # vertically -- not parallel -- so the fully constrained "kept"
    # category keeps them separate. Both debug categories ignore part of
    # that constraint and merge them: "debug_unconstrained" ignores the
    # size/parallel rule entirely (plain bbox-gap rule); "debug_no_parallel"
    # keeps the size/validity rule but drops the parallel requirement, so a
    # same-magnitude match across axes (18 vs 18) is enough.
    a = _make_path(seq=0, kind="l", bbox=(0, 0, 18, 4), stroke_color=(0, 0, 0))
    b = _make_path(seq=1, kind="l", bbox=(22, 0, 26, 18), stroke_color=(0, 0, 0))

    steps = Vector().cluster([a, b], _Page())
    spatial_step = next(s for s in steps if s.label == "Spatial cluster")

    kept_groups = spatial_step.categories["kept"].groups
    assert spatial_step.categories["kept"].role == "kept"
    assert len(kept_groups) == 2  # kept separate: matching sides aren't parallel

    # Neither a nor b merged with anything, so each is its own trivial
    # "dominant" piece (nothing to split against) -- the split step is a
    # no-op here, not a drop.
    dominant_cat = spatial_step.categories["dominant_tag_split"]
    assert dominant_cat.role == "info"
    assert {id(p) for g in dominant_cat.groups for p in g} == {id(a), id(b)}
    assert spatial_step.categories["minority_tag_split"].groups == []

    unconstrained_cat = spatial_step.categories["debug_unconstrained"]
    assert unconstrained_cat.role == "info"
    assert len(unconstrained_cat.groups) == 1
    assert {id(p) for g in unconstrained_cat.groups for p in g} == {id(a), id(b)}

    no_parallel_cat = spatial_step.categories["debug_no_parallel"]
    assert no_parallel_cat.role == "info"
    assert len(no_parallel_cat.groups) == 1
    assert {id(p) for g in no_parallel_cat.groups for p in g} == {id(a), id(b)}


def test_cluster_group_stats_step_is_pass_through_with_stats():
    a = _make_path(seq=0, kind="l", bbox=(0, 0, 3, 6), stroke_color=(0, 0, 0))
    b = _make_path(seq=1, kind="re", bbox=(2, 0, 6, 6), fill_color=(0, 0, 0))

    steps = Vector().cluster([a, b], _Page())
    stats_step = next(s for s in steps if s.label == "Group stats")

    # Pure pass-through: the same groups as the previous step's kept category.
    spatial_step = steps[steps.index(stats_step) - 1]
    assert _sorted_ids(stats_step.categories["kept"].groups) == _sorted_ids(
        spatial_step.categories["kept"].groups
    )

    assert stats_step.group_stats is not None
    for g in stats_step.categories["kept"].groups:
        stats = stats_step.group_stats[id(g)]
        assert stats.member_count == len(g)
        assert stats.unique_signature_count >= 1


def test_cluster_drops_perimeter_only_cluster():
    # Four items forming a ring around the edges of a 20x20 box -- nothing
    # in the shrunk-in (20% margin) center region, so
    # filter_perimeter_only_clusters should drop the whole cluster as
    # border/ring geometry, not text.
    ring = [
        _make_path(seq=0, kind="l", bbox=(0, 0, 20, 1), stroke_color=(0, 0, 0)),
        _make_path(seq=1, kind="l", bbox=(0, 19, 20, 20), stroke_color=(0, 0, 0)),
        _make_path(seq=2, kind="l", bbox=(0, 0, 1, 20), stroke_color=(0, 0, 0)),
        _make_path(seq=3, kind="l", bbox=(19, 0, 20, 20), stroke_color=(0, 0, 0)),
    ]
    centered = _make_path(seq=50, kind="l", bbox=(150, 150, 154, 154), stroke_color=(0, 0, 0))

    steps = Vector().cluster(ring + [centered], _Page())
    perimeter_step = next(s for s in steps if s.label == "Perimeter-only clusters")

    kept_paths = [p for g in perimeter_step.categories["kept"].groups for p in g]
    assert all(p not in kept_paths for p in ring)
    assert centered in kept_paths

    dropped_at_perimeter_step = [
        p for g in perimeter_step.categories["dropped_perimeter"].groups for p in g
    ]
    assert all(p in dropped_at_perimeter_step for p in ring)


def _sorted_ids(groups):
    return sorted(tuple(sorted(id(p) for p in g)) for g in groups)


# ----------------------------------------------------------------------
# helpers/vector_classification.py -- unit tests for the individual step
# functions, run directly (avoids entangling multiple filters' effects on
# small hand-built inputs).
# ----------------------------------------------------------------------


def test_cluster_spatial_groups_merges_when_matching_sides_are_parallel():
    # Group A is a 5x5 square (length axis defaults to x on a tie, width
    # axis y). Group B is 5 wide x 25 tall -- its long side (25) runs
    # vertically, its short side (5, the *same* magnitude as A's sides)
    # runs horizontally, same axis as A's. A's length matching B's width
    # -- both lying on the x axis -- is a genuinely parallel match, per
    # the user's own example ("length of one and width of another is
    # similar... must be parallel"). The 12px gap between them is also
    # above the old SPATIAL_CLUSTER_THRESHOLD (8) but within the new,
    # doubled one (16), so the merge additionally depends on the widened
    # range -- confirmed directly against the old threshold below.
    a = [_make_path(seq=0, kind="l", bbox=(0, 0, 5, 5), stroke_color=(0, 0, 0))]
    b = [_make_path(seq=1, kind="l", bbox=(17, 0, 22, 25), stroke_color=(0, 0, 0))]
    # Group C sits just as close to B but shares no comparably-sized,
    # parallel side with either A or B -- left separate.
    c = [_make_path(seq=2, kind="l", bbox=(22, 40, 82, 100), stroke_color=(0, 0, 0))]

    kept, _, _, debug_unconstrained, debug_no_parallel = vc.cluster_spatial_groups(
        [a, b, c], Clustering(), threshold=16.0, size_tolerance=0.10, min_side_fraction=0.10,
        tag_round_px=1.0,
    )

    kept_sets = {frozenset(id(p) for p in g) for g in kept}
    assert frozenset(id(p) for p in a + b) in kept_sets
    assert frozenset(id(p) for p in c) in kept_sets

    # debug_unconstrained ignores the size rule entirely -- everything
    # spatially close (a-b and b-c both within threshold) chains together.
    assert len(debug_unconstrained) == 1
    assert {id(p) for g in debug_unconstrained for p in g} == {id(p) for p in a + b + c}

    # debug_no_parallel still requires a magnitude match (just not a
    # parallel one) -- a-b still merges (already parallel anyway), but
    # c's sides (60/60) don't match either a's or b's within tolerance
    # regardless of axis, so c stays separate.
    no_parallel_sets = {frozenset(id(p) for p in g) for g in debug_no_parallel}
    assert frozenset(id(p) for p in a + b) in no_parallel_sets
    assert frozenset(id(p) for p in c) in no_parallel_sets

    old_threshold_result = Clustering().cluster_spatial(
        [a, b], get_bbox=vc._bbox_of, threshold=8.0,
    )
    assert len(old_threshold_result) == 2  # gap (12) exceeds the old threshold (8)


def test_cluster_spatial_groups_parallel_constraint_blocks_cross_axis_match():
    # A is a 25x5 (long, low) rect; B is 5x25 (tall, thin) -- same length
    # magnitude (25) as A, but B's long side runs vertically while A's
    # runs horizontally. Magnitude-only matching (ignoring axis) would
    # treat "A's length == B's length" as a hit; requiring the matched
    # sides to be parallel correctly rejects it, since a horizontal 25
    # and a vertical 25 aren't the same physical direction.
    a = [_make_path(seq=0, kind="l", bbox=(0, 0, 25, 5), stroke_color=(0, 0, 0))]
    b = [_make_path(seq=1, kind="l", bbox=(30, 0, 35, 25), stroke_color=(0, 0, 0))]

    kept, _, _, _, debug_no_parallel = vc.cluster_spatial_groups(
        [a, b], Clustering(), threshold=16.0, size_tolerance=0.10, min_side_fraction=0.10,
        tag_round_px=1.0,
    )

    kept_sets = {frozenset(id(p) for p in g) for g in kept}
    assert frozenset(id(p) for p in a) in kept_sets
    assert frozenset(id(p) for p in b) in kept_sets  # kept separate: axes don't match

    assert len(debug_no_parallel) == 1  # merged once the parallel requirement is dropped
    assert {id(p) for g in debug_no_parallel for p in g} == {id(p) for p in a + b}


def test_cluster_spatial_groups_splits_cluster_by_dominant_tag():
    # B is a bridge: it matches A and E on its length (10) -- that tag
    # ends up used by 2 edges (A-B, B-E) -- and matches D on its width
    # (2) -- used by only 1 edge (B-D). All four end up in one spatially
    # merged cluster; the split then separates it into the dominant piece
    # (A, B, E -- connected via the more common length-10 tag) and the
    # minority piece (D alone -- only ever touched via the less common
    # width-2 tag).
    a = [_make_path(seq=0, kind="l", bbox=(-15, 0, -5, 2), stroke_color=(0, 0, 0))]
    b = [_make_path(seq=1, kind="l", bbox=(0, 0, 10, 2), stroke_color=(0, 0, 0))]
    e = [_make_path(seq=2, kind="l", bbox=(15, 0, 25, 2), stroke_color=(0, 0, 0))]
    d = [_make_path(seq=3, kind="l", bbox=(-1, -8, 11, -6), stroke_color=(0, 0, 0))]

    kept, dominant_groups, minority_groups, _, _ = vc.cluster_spatial_groups(
        [a, b, e, d], Clustering(), threshold=6.0, size_tolerance=0.10,
        min_side_fraction=0.10, tag_round_px=1.0,
    )

    assert len(dominant_groups) == 1
    assert {id(p) for p in dominant_groups[0]} == {id(p) for g in (a, b, e) for p in g}

    assert len(minority_groups) == 1
    assert {id(p) for p in minority_groups[0]} == {id(p) for p in d}

    # kept is dominant+minority pieces flattened together -- nothing dropped.
    kept_sets = {frozenset(id(p) for p in g) for g in kept}
    assert frozenset(id(p) for g in (a, b, e) for p in g) in kept_sets
    assert frozenset(id(p) for p in d) in kept_sets
    assert sum(len(g) for g in kept) == sum(len(g) for g in (a, b, e, d))


def test_cluster_spatial_groups_no_split_when_cluster_has_one_tag():
    # A and B merge via a single edge/tag -- nothing to split against, so
    # the whole merged cluster is the dominant piece and there's no
    # minority piece for it.
    a = [_make_path(seq=0, kind="l", bbox=(0, 0, 5, 5), stroke_color=(0, 0, 0))]
    b = [_make_path(seq=1, kind="l", bbox=(6, 0, 11, 5), stroke_color=(0, 0, 0))]

    kept, dominant_groups, minority_groups, _, _ = vc.cluster_spatial_groups(
        [a, b], Clustering(), threshold=6.0, size_tolerance=0.10,
        min_side_fraction=0.10, tag_round_px=1.0,
    )

    assert len(dominant_groups) == 1
    assert {id(p) for p in dominant_groups[0]} == {id(p) for g in (a, b) for p in g}
    assert minority_groups == []
    assert len(kept) == 1


def test_group_sides_excludes_short_side_below_min_fraction():
    # A thin stroke: 100 long, 2 wide -- width/length = 2%, below the 10%
    # validity floor, so the short side is excluded entirely.
    thin_stroke_bbox = (0.0, 0.0, 100.0, 2.0)
    assert vc._group_sides(thin_stroke_bbox, min_side_fraction=0.10) == [(100.0, "x")]

    # 100 long, 15 wide -- 15% >= the floor, so both sides are valid.
    normal_bbox = (0.0, 0.0, 100.0, 15.0)
    assert vc._group_sides(normal_bbox, min_side_fraction=0.10) == [(100.0, "x"), (15.0, "y")]


def test_any_side_close_respects_narrower_tolerance():
    a = [_make_path(seq=0, kind="l", bbox=(0, 0, 20, 20), stroke_color=(0, 0, 0))]
    b = [_make_path(seq=1, kind="l", bbox=(0, 0, 23, 23), stroke_color=(0, 0, 0))]
    # relative diff = 3/23 ~= 13% -- passes the old 20% tolerance but
    # fails the new, narrower 10%.
    assert vc._any_side_close(a, b, size_tolerance=0.20, min_side_fraction=0.10, require_parallel=True) is True
    assert vc._any_side_close(a, b, size_tolerance=0.10, min_side_fraction=0.10, require_parallel=True) is False


def test_compute_group_stats_counts_members_and_unique_signatures():
    g1 = [
        _make_path(seq=0, kind="re", bbox=(0, 0, 3, 6), fill_color=(0, 0, 0)),
        _make_path(seq=1, kind="re", bbox=(10, 0, 13, 6), fill_color=(0, 0, 0)),  # same shape as g1[0]
        _make_path(seq=2, kind="l", bbox=(20, 0, 24, 8), stroke_color=(0, 0, 0)),  # different shape
    ]
    _, signature_counts = vc.compute_vector_signatures([g1], round_px=0.5)

    _, stats = vc.compute_group_stats([g1], signature_counts, round_px=0.5)

    s = stats[id(g1)]
    assert s.member_count == 3
    assert s.unique_signature_count == 2


def test_filter_high_frequency_clusters_drops_top_fraction_only():
    # 7 distinct signatures total; one occurs 20 times (a repeated filler
    # shape), the rest occur once each. With top_fraction=0.05 the cutoff
    # lands on the single highest-count signature (20), so only a cluster
    # made entirely of that shape is dropped.
    filler_paths = [
        _make_path(seq=i, kind="re", bbox=(0, 0, 3, 6), fill_color=(0, 0, 0))
        for i in range(20)
    ]
    unique_paths = [
        _make_path(seq=100 + i, kind="l", bbox=(0, 0, 4 + i, 4 + i), stroke_color=(0, 0, 0))
        for i in range(6)
    ]
    all_groups = [[p] for p in filler_paths + unique_paths]
    _, signature_counts = vc.compute_vector_signatures(all_groups, round_px=0.5)

    filler_group = [filler_paths]
    unique_groups = [[p] for p in unique_paths]

    kept, dropped = vc.filter_high_frequency_clusters(
        filler_group + unique_groups, signature_counts, round_px=0.5, top_fraction=0.05,
    )

    assert dropped == [filler_paths]
    assert _sorted_ids(kept) == _sorted_ids(unique_groups)


def test_filter_mixed_fill_rule_clusters_drops_mixed_group():
    mixed = [
        _make_path(seq=0, kind="re", bbox=(0, 0, 3, 6), fill_rule="f"),
        _make_path(seq=1, kind="re", bbox=(10, 0, 13, 6), fill_rule="s"),
    ]
    uniform = [
        _make_path(seq=2, kind="re", bbox=(0, 20, 3, 26), fill_rule="f"),
        _make_path(seq=3, kind="re", bbox=(10, 20, 13, 26), fill_rule="f"),
    ]

    kept, dropped = vc.filter_mixed_fill_rule_clusters([mixed, uniform])

    assert dropped == [mixed]
    assert kept == [uniform]


def test_filter_density_clusters_drops_sparse_cluster():
    # bbox (0,0,40,40): default_grid_size=4 clamped to [5,40]px cells
    # lands right back on 4x4 (10x10px cells, within bounds -- no
    # adjustment needed) -> 16 cells. Members sit only in the two
    # opposite-corner cells -- 14/16 (87.5%) empty, above
    # max_empty_fraction (40%) -- dropped.
    sparse = [
        _make_path(seq=0, kind="l", bbox=(0, 0, 5, 5), stroke_color=(0, 0, 0)),
        _make_path(seq=1, kind="l", bbox=(35, 35, 40, 40), stroke_color=(0, 0, 0)),
    ]
    # One member per cell -- every cell touched, 0 empty -- kept.
    dense = [
        _make_path(
            seq=r * 4 + c, kind="l",
            bbox=(c * 10 + 1, r * 10 + 1, c * 10 + 9, r * 10 + 9),
            stroke_color=(0, 0, 0),
        )
        for r in range(4) for c in range(4)
    ]

    kept, dropped = vc.filter_density_clusters(
        [sparse, dense], default_grid_size=4, min_cell_px=5.0, max_cell_px=40.0, max_empty_fraction=0.40,
    )

    assert dropped == [sparse]
    assert kept == [dense]


def test_filter_constant_spacing_clusters_drops_regular_pattern():
    # 10 identical rects, each a perfect 5px translation of the previous --
    # constant spacing, a regular repeated pattern -- dropped.
    pattern = [
        _make_path(seq=i, kind="re", bbox=(i * 5, 0, i * 5 + 2, 2), fill_color=(0, 0, 0))
        for i in range(10)
    ]
    kept, dropped = vc.filter_constant_spacing_clusters(
        [pattern], round_px=0.5, spacing_tolerance=0.10, min_repeat_count=3,
        pattern_fraction_threshold=0.70,
    )
    assert dropped == [pattern]
    assert kept == []

    # Same shape repeated (>= min_repeat_count), but the gap jumps from 5px
    # to 15px between the 2nd and 3rd member -- irregular spacing -- kept.
    irregular = [
        _make_path(seq=100, kind="re", bbox=(0, 0, 2, 2), fill_color=(0, 0, 0)),
        _make_path(seq=101, kind="re", bbox=(5, 0, 7, 2), fill_color=(0, 0, 0)),
        _make_path(seq=102, kind="re", bbox=(20, 0, 22, 2), fill_color=(0, 0, 0)),
    ]
    kept2, dropped2 = vc.filter_constant_spacing_clusters(
        [irregular], round_px=0.5, spacing_tolerance=0.10, min_repeat_count=3,
        pattern_fraction_threshold=0.70,
    )
    assert kept2 == [irregular]
    assert dropped2 == []


def test_filter_constant_spacing_clusters_drops_at_pattern_fraction_threshold():
    # Shape A: 7 members, each a perfect 5px translation of the previous --
    # patterned (spacing checked separately per shape). Shape B: 3 members
    # of a different shape, with irregular spacing -- not patterned. Only
    # shape A's sub-group is patterned, but its 7 members make up exactly
    # 70% of the cluster's 10 total members, meeting PATTERN_FRACTION_
    # THRESHOLD -- the whole cluster is dropped even though not every
    # sub-group is itself a pattern.
    shape_a = [
        _make_path(seq=i, kind="re", bbox=(i * 5, 0, i * 5 + 2, 2), fill_color=(0, 0, 0))
        for i in range(7)
    ]
    shape_b = [
        _make_path(seq=100, kind="re", bbox=(0, 100, 3, 103), fill_color=(0, 0, 0)),
        _make_path(seq=101, kind="re", bbox=(5, 100, 8, 103), fill_color=(0, 0, 0)),
        _make_path(seq=102, kind="re", bbox=(50, 100, 53, 103), fill_color=(0, 0, 0)),
    ]
    cluster = shape_a + shape_b

    kept, dropped = vc.filter_constant_spacing_clusters(
        [cluster], round_px=0.5, spacing_tolerance=0.10, min_repeat_count=3,
        pattern_fraction_threshold=0.70,
    )
    assert dropped == [cluster]
    assert kept == []

    # Drop shape A down to 6 members -- patterned fraction is now 6/9 ~=
    # 67%, below the threshold -- kept.
    below_threshold_cluster = shape_a[:6] + shape_b
    kept2, dropped2 = vc.filter_constant_spacing_clusters(
        [below_threshold_cluster], round_px=0.5, spacing_tolerance=0.10, min_repeat_count=3,
        pattern_fraction_threshold=0.70,
    )
    assert kept2 == [below_threshold_cluster]
    assert dropped2 == []


def test_required_unique_signature_count_scales_with_member_count():
    # Matches the production anchors: 5 members -> 1 required; 10 members
    # -> 2 required (log-scale ramp, fast early); 300+ members -> capped
    # at 10 required.
    assert vc._required_unique_signature_count(5, 5, 1, 300, 10) == 1
    assert vc._required_unique_signature_count(2, 5, 1, 300, 10) == 1  # below the floor, clamped
    assert vc._required_unique_signature_count(10, 5, 1, 300, 10) == 2
    assert vc._required_unique_signature_count(300, 5, 1, 300, 10) == 10
    assert vc._required_unique_signature_count(10_000, 5, 1, 300, 10) == 10  # capped


def test_filter_low_variety_clusters_drops_below_scaled_minimum():
    # Both clusters have 10 members -- _required_unique_signature_count(10,
    # 5, 1, 300, 10) == 2. "varied" has 2 distinct shapes (5 members each)
    # -- meets the requirement, kept. "uniform" is 10 copies of one shape
    # -- only 1 distinct shape, below the requirement, dropped.
    varied = [
        _make_path(seq=i, kind="re", bbox=(i * 10, 0, i * 10 + 3, 6), fill_color=(0, 0, 0))
        for i in range(5)
    ] + [
        _make_path(seq=100 + i, kind="l", bbox=(i * 10, 50, i * 10 + 4, 54), stroke_color=(0, 0, 0))
        for i in range(5)
    ]
    uniform = [
        _make_path(seq=200 + i, kind="re", bbox=(i * 10, 100, i * 10 + 3, 106), fill_color=(0, 0, 0))
        for i in range(10)
    ]

    groups = [varied, uniform]
    _, signature_counts = vc.compute_vector_signatures(groups, round_px=0.5)
    _, group_stats = vc.compute_group_stats(groups, signature_counts, round_px=0.5)

    kept, dropped = vc.filter_low_variety_clusters(
        groups, group_stats, min_member_count=5, min_required=1,
        max_member_count=300, max_required=10,
    )

    assert kept == [varied]
    assert dropped == [uniform]


def test_build_drawing_vectors_aggregates_by_seq():
    a = _make_path(seq=5, kind="l", bbox=(0, 0, 10, 0), stroke_color=(0, 0, 0), stroke_width=2)
    b = _make_path(seq=5, kind="l", bbox=(10, 0, 10, 10), stroke_color=(0, 0, 0), stroke_width=2)
    c = _make_path(seq=9, kind="re", bbox=(50, 50, 60, 60), fill_color=(1, 0, 0))

    result = Vector().build_drawing_vectors([a, b, c])

    by_seq = {dv.paths[0].seq: dv for dv in result}
    assert set(by_seq) == {5, 9}
    assert by_seq[5].bbox == pytest.approx((0, 0, 10, 10))
    assert by_seq[5].stroke_color == (0, 0, 0)
    assert by_seq[9].fill_color == (1, 0, 0)
