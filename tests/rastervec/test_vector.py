from __future__ import annotations

import pymupdf as fitz
import pytest

from rastervec.models import VectorPath
from rastervec.reader import Reader
from rastervec.vector import Vector, _is_dashed
from rastervec.vector_classification import DEFAULT_PIPELINE, StepConfig


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


class _Meta:
    width = 200
    height = 200


class _Page:
    meta = _Meta()


def test_classify_returns_clusters_as_text_candidates_no_heuristic():
    # All four glyphs share one seq (one drawing, several items) so the
    # default pipeline's layout-panel filter -- which only drops a *lone*-
    # item re/qu drawing -- leaves them alone; a real CAD text-as-vector-
    # paths drawing is exactly this shape (one seq, many glyph items).
    # "far_away" is a single unrelated path that stays in its own separate
    # cluster purely because it's spatially far from the glyphs -- not
    # because of any drawing-vs-text heuristic (there isn't one; classify()
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
    # kind="l" (not re/qu) so the layout-panel filter's lone-panel rule
    # doesn't drop it before it reaches the final grouping step. A
    # single-path cluster is returned as-is like any other -- there's no
    # minimum member-count heuristic gating what counts as a text
    # candidate.
    lone = _make_path(kind="l", bbox=(0, 0, 3, 6), fill_color=(0, 0, 0))

    clusters = Vector().classify([lone], _Page())

    assert clusters == [[lone]]


def _sorted_ids(groups):
    return sorted(tuple(sorted(id(p) for p in g)) for g in groups)


def test_cluster_default_pipeline_matches_classify():
    a = _make_path(seq=0, bbox=(10, 10, 13, 16), fill_color=(0, 0, 0))
    b = _make_path(seq=0, item_index=1, bbox=(15, 10, 18, 16), fill_color=(0, 0, 0))
    c = _make_path(seq=100, kind="l", bbox=(100, 100, 180, 180), stroke_color=(0, 0, 0))
    paths = [a, b, c]

    kept_snapshots, dropped_snapshots = Vector().cluster(paths, _Page())
    assert len(kept_snapshots) == len(dropped_snapshots) == len(DEFAULT_PIPELINE) == 5

    # classify() is a thin wrapper around cluster()'s default steps -- its
    # result must be exactly cluster()'s final kept snapshot.
    assert _sorted_ids(kept_snapshots[-1]) == _sorted_ids(Vector().classify(paths, _Page()))


def test_cluster_step_order_changes_result():
    # Same size, far apart -- dimension-based grouping merges them
    # regardless of position; spatial clustering (default threshold=8.0)
    # keeps them separate since they're far apart.
    a = _make_path(bbox=(0, 0, 10, 10))
    b = _make_path(bbox=(50, 50, 60, 60))
    paths = [a, b]

    default_final = Vector().cluster(paths, _Page())[0][-1]
    assert len(default_final) == 2  # far apart -- stays split through the default pipeline

    dimension_last = [
        StepConfig(kind="cluster", metric="spatial_gap", method="global", threshold=8.0),
        StepConfig(kind="group", metric="dimension_similarity", scope="cluster", threshold=0.35),
    ]
    dimension_last_final = Vector().cluster(paths, _Page(), dimension_last)[0][-1]
    assert len(dimension_last_final) == 1  # dimension-grouping (last) merges same-size groups

    dimension_first = [
        StepConfig(kind="group", metric="dimension_similarity", scope="cluster", threshold=0.35),
        StepConfig(kind="cluster", metric="spatial_gap", method="global", threshold=8.0),
    ]
    dimension_first_final = Vector().cluster(paths, _Page(), dimension_first)[0][-1]
    # dimension merges them first, but spatial clustering (now 2nd) is a
    # from-scratch pass -- far apart, so it splits them back up.
    assert len(dimension_first_final) == 2


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
