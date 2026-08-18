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


def test_filter_background_fill_drops_dominant_large_fill():
    background = _make_path(bbox=(0, 0, 200, 200), fill_color=(1, 1, 1))
    small_content = _make_path(bbox=(10, 10, 13, 16), fill_color=(0, 0, 0))

    class _Meta:
        width = 200
        height = 200

    class _Page:
        meta = _Meta()

    result = Vector().filter_background_fill([background, small_content], _Page())

    assert background not in result
    assert small_content in result


def test_filter_background_fill_no_dominant_fill_keeps_everything():
    a = _make_path(bbox=(0, 0, 10, 10), fill_color=(1, 0, 0))
    b = _make_path(bbox=(20, 20, 30, 30), fill_color=(0, 1, 0))

    class _Meta:
        width = 200
        height = 200

    class _Page:
        meta = _Meta()

    result = Vector().filter_background_fill([a, b], _Page())

    assert result == [a, b]


def test_classify_groups_small_close_filled_paths_as_text():
    text_like = [
        _make_path(seq=i, bbox=(10 + i * 5, 10, 10 + i * 5 + 3, 16), fill_color=(0, 0, 0))
        for i in range(4)
    ]
    drawing_like = _make_path(
        seq=100, kind="l", bbox=(100, 100, 180, 180), stroke_color=(0, 0, 0)
    )

    drawing_paths, text_clusters = Vector().classify(text_like + [drawing_like])

    assert drawing_like in drawing_paths
    assert len(text_clusters) == 1
    assert len(text_clusters[0]) == 4


def test_classify_single_item_cluster_is_not_text():
    lone = _make_path(bbox=(0, 0, 3, 6), fill_color=(0, 0, 0))

    drawing_paths, text_clusters = Vector().classify([lone])

    assert drawing_paths == [lone]
    assert text_clusters == []


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
