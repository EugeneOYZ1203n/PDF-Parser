from __future__ import annotations

import pymupdf as fitz
import pytest

from rastervec.models import VectorPath
from rastervec.Reader.reader import Reader
from rastervec.Vector.vector import Vector


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


def test_extract_records_carries_drawing_level_fields(tmp_pdf_path):
    with _build_test_page(tmp_pdf_path) as reader:
        page = reader.get_page(0)
        records = Vector().extract_records(page)

    assert len(records) > 0
    assert all(isinstance(r.items, list) and r.items for r in records)
    assert all(r.groups is None and r.role is None for r in records)
    small_rect_records = [
        r for r in records
        if any(p.kind == "re" and (p.bbox[2] - p.bbox[0]) < 10 for p in r.items)
    ]
    assert len(small_rect_records) == 4


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
