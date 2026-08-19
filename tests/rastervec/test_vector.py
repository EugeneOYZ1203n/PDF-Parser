from __future__ import annotations

import pymupdf as fitz
import pytest

from rastervec.models import VectorPath
from rastervec.reader import Reader
from rastervec.vector import Vector, _is_dashed


def _make_path(
    *,
    seq=0,
    item_index=0,
    kind="re",
    bbox=(0, 0, 1, 1),
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
        fill_rule="",
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


def test_filter_layout_panels_drops_single_item_re():
    lone_re = _make_path(seq=0, kind="re")  # only item in its drawing -> dropped

    result = Vector().filter_layout_panels([lone_re])

    assert result == []


def test_filter_layout_panels_keeps_multi_item_re():
    re_with_sibling = _make_path(seq=1, kind="re")
    sibling = _make_path(seq=1, kind="l")  # same seq -> re is not single-item, kept

    result = Vector().filter_layout_panels([re_with_sibling, sibling])

    assert re_with_sibling in result
    assert sibling in result


def test_filter_layout_panels_keeps_single_item_line():
    single_line = _make_path(seq=2, kind="l")  # single item but not re/qu -> kept

    result = Vector().filter_layout_panels([single_line])

    assert result == [single_line]


class _Meta:
    width = 200
    height = 200


class _Page:
    meta = _Meta()


def test_classify_returns_clusters_as_text_candidates_no_heuristic():
    # All four glyphs share one seq (one drawing, several items) so
    # filter_layout_panels -- which only drops a *lone*-item re/qu drawing
    # -- leaves them alone; a real CAD text-as-vector-paths drawing is
    # exactly this shape (one seq, many glyph items). "far_away" is a
    # single unrelated path that stays in its own separate cluster purely
    # because it's spatially far from the glyphs -- not because of any
    # drawing-vs-text heuristic (Vector no longer has one; classify()
    # returns every surviving cluster as-is, letting OCR itself decide).
    text_like = [
        _make_path(
            seq=0, item_index=i, bbox=(10 + i * 5, 10, 10 + i * 5 + 3, 16),
            fill_color=(0, 0, 0),
        )
        for i in range(4)
    ]
    far_away = _make_path(
        seq=100, kind="l", bbox=(100, 100, 180, 180), stroke_color=(0, 0, 0)
    )

    clusters = Vector().classify(text_like + [far_away], _Page())

    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 4]
    assert any(far_away in c for c in clusters if len(c) == 1)


def test_classify_keeps_single_item_cluster():
    # kind="l" (not re/qu) so filter_layout_panels' lone-panel rule doesn't
    # drop it before it reaches the final grouping step. A single-path
    # cluster is returned as-is like any other -- there's no minimum
    # member-count heuristic gating what counts as a text candidate.
    lone = _make_path(kind="l", bbox=(0, 0, 3, 6), fill_color=(0, 0, 0))

    clusters = Vector().classify([lone], _Page())

    assert clusters == [[lone]]


def test_filter_large_bbox_drops_oversized_path():
    big = _make_path(bbox=(0, 0, 190, 190))  # 36100 / 40000 = 90% of page
    small = _make_path(bbox=(10, 10, 20, 20))

    result = Vector().filter_large_bbox([big, small], _Page())

    assert big not in result
    assert small in result


def test_group_overlapping_merges_partial_overlap_not_full_containment():
    # page is 200x200 -> tolerance = max(0.5% * 200, 3px) = 3px
    a = _make_path(bbox=(0, 0, 10, 10))
    b = _make_path(bbox=(5, 5, 15, 15))  # partially overlaps a -> merge
    # fully inside a, and far enough from b (gap > tolerance) that it
    # can't sneak into the group via a near-miss merge with b instead
    c = _make_path(bbox=(0.5, 0.5, 1.5, 1.5))

    groups = Vector().group_overlapping([[a, b, c]], _Page())

    merged = [g for g in groups if a in g]
    assert len(merged) == 1
    assert sorted(merged[0], key=id) == sorted([a, b], key=id)
    assert any(g == [c] for g in groups)


def test_group_overlapping_merges_within_gap_tolerance():
    # page is 200x200 -> tolerance = max(0.5% * 200, 3px) = 3px
    a = _make_path(bbox=(0, 0, 10, 10))
    b = _make_path(bbox=(12, 0, 20, 10))  # gap = 2px <= 3px tolerance -> merge
    c = _make_path(bbox=(40, 0, 50, 10))  # gap = 20px > tolerance -> stays separate

    groups = Vector().group_overlapping([[a, b, c]], _Page())

    merged = [g for g in groups if a in g]
    assert len(merged) == 1
    assert sorted(merged[0], key=id) == sorted([a, b], key=id)
    assert any(g == [c] for g in groups)


def test_group_overlapping_tolerance_scales_with_page_size():
    class _BigMeta:
        width = 3000
        height = 3000

    class _BigPage:
        meta = _BigMeta()

    # tolerance = max(0.5% * 3000, 3px) = 15px
    a = _make_path(bbox=(0, 0, 10, 10))
    b = _make_path(bbox=(20, 0, 30, 10))  # gap = 10px -- merges only on the big page

    groups = Vector().group_overlapping([[a, b]], _BigPage())

    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_group_overlapping_explicit_tolerance_overrides_page_default():
    # page is 200x200 -> default tolerance would be 3px, too small for this
    # 10px gap; an explicit override should still let it merge.
    a = _make_path(bbox=(0, 0, 10, 10))
    b = _make_path(bbox=(20, 0, 30, 10))

    default_groups = Vector().group_overlapping([[a, b]], _Page())
    assert len(default_groups) == 2  # default 3px tolerance: stays separate

    overridden_groups = Vector().group_overlapping([[a, b]], _Page(), tolerance=15.0)
    assert len(overridden_groups) == 1  # explicit 15px tolerance: merges


def test_default_overlap_tolerance_matches_group_overlapping_default():
    assert Vector.default_overlap_tolerance(_Page()) == 3.0  # max(0.5% * 200, 3px)


def test_cluster_spatial_explicit_threshold_overrides_instance_default():
    a = _make_path(bbox=(0, 0, 2, 2))
    b = _make_path(bbox=(20, 0, 22, 2))  # gap=18px, past default threshold=8.0

    default_result = Vector().cluster_spatial([a, b])
    assert len(default_result) == 2

    overridden_result = Vector().cluster_spatial([a, b], threshold=20.0)
    assert len(overridden_result) == 1


def test_cluster_by_seq_explicit_max_gap_overrides_instance_default():
    a = _make_path(seq=0, bbox=(0, 0, 2, 2))
    b = _make_path(seq=10, bbox=(0, 0, 2, 2))  # seq gap=10, past default max_gap=3

    default_result = Vector().cluster_by_seq([[a, b]])
    assert len(default_result) == 2

    overridden_result = Vector().cluster_by_seq([[a, b]], max_gap=20)
    assert len(overridden_result) == 1


def test_cluster_groups_by_dimension_explicit_tolerance_overrides_instance_default():
    g1 = [_make_path(bbox=(0, 0, 10, 10))]
    g2 = [_make_path(bbox=(0, 0, 13, 13))]  # 30% bigger, past default tolerance=0.35... just within

    default_result = Vector().cluster_groups_by_dimension([g1, g2])
    assert len(default_result) == 1  # 0.3 relative diff <= default 0.35 -> merges

    overridden_result = Vector().cluster_groups_by_dimension([g1, g2], tolerance=0.1)
    assert len(overridden_result) == 2  # tighter tolerance: stays separate


def test_cluster_step_params_are_keyed_by_step_not_ordinal_position():
    # cluster_spatial and cluster_spatial_union_find both default to the
    # same spatial_threshold instance attribute, but per-call step_params
    # overrides must stay independent between them.
    a = _make_path(bbox=(0, 0, 2, 2))
    b = _make_path(bbox=(15, 0, 17, 2))  # gap=13px

    order = ["cluster_spatial", "cluster_spatial_union_find", "none", "none"]
    step_params = {
        "cluster_spatial": {"threshold": 20.0},  # merges at step 1
        "cluster_spatial_union_find": {"threshold": 1.0},  # would split if it re-derived, but it can't re-split
    }
    snapshots, _dropped = Vector().cluster([a, b], _Page(), order, step_params=step_params)

    assert len(snapshots[0]) == 1  # cluster_spatial(threshold=20) merged them
    # cluster_spatial_union_find never re-splits an existing group (see its
    # own docstring/tests), so the already-merged group survives step 2
    # even with a tiny threshold override.
    assert len(snapshots[1]) == 1


def test_filter_large_group_bbox_drops_oversized_group():
    big_group = [_make_path(bbox=(0, 0, 190, 190))]  # 90% of 200x200 page
    small_group = [_make_path(bbox=(10, 10, 20, 20))]

    result = Vector().filter_large_group_bbox([big_group, small_group], _Page())

    assert small_group in result
    assert big_group not in result


def test_filter_aspect_ratio_drops_long_thin_group():
    line_like = [_make_path(bbox=(0, 0, 100, 2))]
    square_like = [_make_path(bbox=(0, 0, 10, 10))]

    result = Vector().filter_aspect_ratio([line_like, square_like])

    assert square_like in result
    assert line_like not in result


def test_cluster_spatial_union_find_merges_close_groups_without_splitting():
    close_a = [_make_path(bbox=(0, 0, 2, 2))]
    close_b = [_make_path(bbox=(4, 0, 6, 2))]  # gap=2 from close_a -> within default threshold=8.0
    far = [_make_path(bbox=(100, 100, 102, 102))]

    result = Vector().cluster_spatial_union_find([close_a, close_b, far])

    sizes = sorted(len(g) for g in result)
    assert sizes == [1, 2]  # close_a+close_b merge into one group; far stays alone


def test_cluster_spatial_union_find_never_splits_an_incoming_group():
    # A single incoming group already spans a huge bbox gap internally (as
    # if an earlier operation grouped these paths for some other reason) --
    # unlike cluster_spatial (which re-flattens and re-derives from
    # scratch), cluster_spatial_union_find treats the whole incoming group
    # as one atomic unit and never re-splits it.
    pre_grouped = [
        _make_path(bbox=(0, 0, 2, 2)), _make_path(bbox=(500, 500, 502, 502)),
    ]
    other = [_make_path(bbox=(1000, 1000, 1002, 1002))]

    result = Vector().cluster_spatial_union_find([pre_grouped, other])

    sizes = sorted(len(g) for g in result)
    assert sizes == [1, 2]
    pre_grouped_ids = {id(p) for p in pre_grouped}
    assert any(len(g) == 2 and {id(p) for p in g} == pre_grouped_ids for g in result)


def test_cluster_groups_by_dimension_merges_similar_sized_groups():
    g1 = [_make_path(bbox=(0, 0, 10, 10))]
    g2 = [_make_path(bbox=(50, 50, 60, 60))]  # same size as g1, far away -> still merges
    g3 = [_make_path(bbox=(0, 0, 100, 100))]  # very different size -> stays separate

    result = Vector().cluster_groups_by_dimension([g1, g2, g3])

    sizes = sorted(len(g) for g in result)
    assert sizes == [1, 2]


def _sorted_ids(groups):
    return sorted(tuple(sorted(id(p) for p in g)) for g in groups)


def test_cluster_default_order_matches_pipeline_steps_and_classify():
    a = _make_path(seq=0, bbox=(10, 10, 13, 16), fill_color=(0, 0, 0))
    b = _make_path(seq=0, item_index=1, bbox=(15, 10, 18, 16), fill_color=(0, 0, 0))
    c = _make_path(seq=100, kind="l", bbox=(100, 100, 180, 180), stroke_color=(0, 0, 0))
    paths = [a, b, c]

    assert list(Vector.PIPELINE_STEPS) == [
        "filter_layout_panels", "filter_large_bbox", "cluster_spatial",
        "none", "none", "none", "filter_large_group_bbox", "filter_aspect_ratio",
    ]

    kept_snapshots, dropped_snapshots = Vector().cluster(paths, _Page())
    assert len(kept_snapshots) == len(dropped_snapshots) == 8

    # classify() is a thin wrapper around cluster()'s default order --
    # its result must be exactly cluster()'s final kept snapshot.
    assert _sorted_ids(kept_snapshots[-1]) == _sorted_ids(Vector().classify(paths, _Page()))


def test_cluster_order_changes_result():
    # Same size, far apart -- dimension-clustering merges them regardless
    # of position; spatial clustering (threshold=8.0 default) keeps them
    # separate since they're far apart.
    a = _make_path(bbox=(0, 0, 10, 10))
    b = _make_path(bbox=(50, 50, 60, 60))
    paths = [a, b]

    default_final = Vector().cluster(paths, _Page())[0][-1]
    assert len(default_final) == 2  # far apart -- stays split through the default pipeline

    full_order = ["cluster_spatial", "cluster_by_seq", "group_overlapping", "cluster_groups_by_dimension"]
    full_final = Vector().cluster(paths, _Page(), full_order)[0][-1]
    assert len(full_final) == 1  # dimension-clustering (last) merges same-size groups regardless of position

    reordered = ["cluster_groups_by_dimension", "cluster_spatial", "cluster_by_seq", "group_overlapping"]
    reordered_final = Vector().cluster(paths, _Page(), reordered)[0][-1]
    # dimension merges them first, but spatial (now 2nd) re-flattens and
    # re-clusters from scratch -- far apart, so it splits them back up.
    assert len(reordered_final) == 2


def test_cluster_group_overlapping_merges_from_scratch_as_sole_step():
    # Regression test: group_overlapping used to be a no-op when it's the
    # only active step, because Clustering.group_by_overlap only ever
    # splits within an *existing* incoming group and cluster()'s initial
    # state is one singleton group per path -- a group of size 1 can never
    # be split, so nothing merged and the tolerance param had no effect.
    # _apply_pipeline_step now flattens into a single incoming group first,
    # matching cluster_spatial's from-scratch behavior.
    a = _make_path(bbox=(0, 0, 5, 5))
    b = _make_path(bbox=(5.5, 0, 10, 5))  # 0.5 gap -- within default tolerance
    paths = [a, b]

    order = ["group_overlapping", "none", "none", "none"]
    merged = Vector().cluster(paths, _Page(), order)[0][-1]
    assert len(merged) == 1

    tight = Vector().cluster(
        paths, _Page(), order, step_params={"group_overlapping": {"tolerance": 0.1}}
    )[0][-1]
    assert len(tight) == 2  # 0.5 gap > 0.1 tolerance -- stays split


def test_cluster_group_overlapping_cluster_scope_never_splits_incoming_group():
    # bbox_scope="cluster" treats each incoming group as one atomic unit
    # (like cluster_spatial_union_find), so two already-grouped paths never
    # get split apart even though they're far from each other individually.
    a = _make_path(bbox=(0, 0, 5, 5))
    b = _make_path(bbox=(100, 100, 105, 105))
    c = _make_path(bbox=(100.5, 100, 105.5, 105))  # close to b only

    # First cluster_spatial groups b+c together (close) and leaves a alone;
    # then group_overlapping in "cluster" scope must never split b+c apart,
    # regardless of a's distance from that group.
    order = ["cluster_spatial", "group_overlapping", "none", "none"]
    result = Vector().cluster(
        [a, b, c], _Page(), order,
        step_params={"group_overlapping": {"bbox_scope": "cluster"}},
    )[0][-1]

    bc_group = [g for g in result if any(p in g for p in (b, c))]
    assert len(bc_group) == 1
    assert set(id(p) for p in bc_group[0]) == {id(b), id(c)}


def test_cluster_none_step_is_identity():
    a = _make_path(bbox=(0, 0, 10, 10))
    b = _make_path(bbox=(50, 50, 60, 60))
    paths = [a, b]

    order = ["none", "cluster_spatial", "none", "none"]
    kept_snapshots, _dropped_snapshots = Vector().cluster(paths, _Page(), order)

    # "none" makes no change: step 0's output is just each path alone,
    # same as the initial singleton groups cluster() starts from.
    assert sorted(len(g) for g in kept_snapshots[0]) == [1, 1]
    # steps 2 and 3 ("none" again) leave step 1's (cluster_spatial's) result untouched.
    assert kept_snapshots[1] == kept_snapshots[2] == kept_snapshots[3]


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
