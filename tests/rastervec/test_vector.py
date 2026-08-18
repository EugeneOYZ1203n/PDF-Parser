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


def test_separate_by_color_prefers_stroke_over_fill():
    paths = [
        _make_path(stroke_color=(1, 0, 0), fill_color=(0, 1, 0)),
        _make_path(stroke_color=None, fill_color=(0, 1, 0)),
        _make_path(stroke_color=None, fill_color=None),
    ]

    groups = Vector().separate_by_color(paths)

    assert groups[(1, 0, 0)] == [paths[0]]
    assert groups[(0, 1, 0)] == [paths[1]]
    assert groups[None] == [paths[2]]


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


def test_classify_groups_small_close_filled_paths_as_text():
    # All four glyphs share one seq (one drawing, several items) so
    # filter_layout_panels -- which only drops a *lone*-item re/qu drawing
    # -- leaves them alone; a real CAD text-as-vector-paths drawing is
    # exactly this shape (one seq, many glyph items).
    text_like = [
        _make_path(
            seq=0, item_index=i, bbox=(10 + i * 5, 10, 10 + i * 5 + 3, 16),
            fill_color=(0, 0, 0),
        )
        for i in range(4)
    ]
    drawing_like = _make_path(
        seq=100, kind="l", bbox=(100, 100, 180, 180), stroke_color=(0, 0, 0)
    )

    drawing_paths, text_clusters = Vector().classify(text_like + [drawing_like], _Page())

    assert drawing_like in drawing_paths
    assert len(text_clusters) == 1
    assert len(text_clusters[0]) == 4


def test_classify_single_item_cluster_is_not_text():
    # kind="l" (not re/qu) so filter_layout_panels' lone-panel rule doesn't
    # drop it before it reaches the final grouping/classification step.
    lone = _make_path(kind="l", bbox=(0, 0, 3, 6), fill_color=(0, 0, 0))

    drawing_paths, text_clusters = Vector().classify([lone], _Page())

    assert drawing_paths == [lone]
    assert text_clusters == []


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


def test_cluster_groups_by_dimension_merges_similar_sized_groups():
    g1 = [_make_path(bbox=(0, 0, 10, 10))]
    g2 = [_make_path(bbox=(50, 50, 60, 60))]  # same size as g1, far away -> still merges
    g3 = [_make_path(bbox=(0, 0, 100, 100))]  # very different size -> stays separate

    result = Vector().cluster_groups_by_dimension([g1, g2, g3])

    sizes = sorted(len(g) for g in result)
    assert sizes == [1, 2]


def test_cluster_default_order_matches_manual_chain():
    a = _make_path(seq=0, bbox=(10, 10, 13, 16), fill_color=(0, 0, 0))
    b = _make_path(seq=0, item_index=1, bbox=(15, 10, 18, 16), fill_color=(0, 0, 0))
    c = _make_path(seq=100, kind="l", bbox=(100, 100, 180, 180), stroke_color=(0, 0, 0))
    paths = [a, b, c]

    snapshots = Vector().cluster(paths, _Page())
    assert len(snapshots) == 4
    assert len(snapshots[0]) >= 1  # sanity: every step produced something

    vector = Vector()
    spatial = vector.cluster_spatial(paths)
    seq_clusters = vector.cluster_by_seq(spatial)
    overlap_groups = vector.group_overlapping(seq_clusters, _Page())
    dimension_groups = vector.cluster_groups_by_dimension(overlap_groups)

    def _sorted_ids(groups):
        return sorted(tuple(sorted(id(p) for p in g)) for g in groups)

    assert _sorted_ids(snapshots[-1]) == _sorted_ids(dimension_groups)


def test_cluster_order_changes_result():
    # Same size, far apart -- dimension-clustering merges them regardless
    # of position; spatial clustering (threshold=8.0 default) keeps them
    # separate since they're far apart.
    a = _make_path(bbox=(0, 0, 10, 10))
    b = _make_path(bbox=(50, 50, 60, 60))
    paths = [a, b]

    default_order = list(Vector.CLUSTER_STEPS)
    assert default_order[-1] == "cluster_groups_by_dimension"
    default_final = Vector().cluster(paths, _Page(), default_order)[-1]
    assert len(default_final) == 1  # dimension-clustering (last) merges them

    reordered = ["cluster_groups_by_dimension", "cluster_spatial", "cluster_by_seq", "group_overlapping"]
    reordered_final = Vector().cluster(paths, _Page(), reordered)[-1]
    # dimension merges them first, but spatial (now 2nd) re-flattens and
    # re-clusters from scratch -- far apart, so it splits them back up.
    assert len(reordered_final) == 2


def test_cluster_none_step_is_identity():
    a = _make_path(bbox=(0, 0, 10, 10))
    b = _make_path(bbox=(50, 50, 60, 60))
    paths = [a, b]

    order = ["none", "cluster_spatial", "none", "none"]
    snapshots = Vector().cluster(paths, _Page(), order)

    # "none" makes no change: step 0's output is just each path alone,
    # same as the initial singleton groups cluster() starts from.
    assert sorted(len(g) for g in snapshots[0]) == [1, 1]
    # steps 2 and 3 ("none" again) leave step 1's (cluster_spatial's) result untouched.
    assert snapshots[1] == snapshots[2] == snapshots[3]


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
