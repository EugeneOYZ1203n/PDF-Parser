"""`commons.v1_commons.render`: the original reconstructor's replay, on shared `Vector`s."""

from __future__ import annotations

import pymupdf

from raster_parser.commons.v1_commons.render import draw_vectors, finish_kwargs


def test_finish_kwargs_defaults_for_unset_fields(vector) -> None:
    kw = finish_kwargs(vector(type="f", color=None, fill=(0, 0, 0)))
    assert kw["width"] == 0 and kw["closePath"] is False
    assert kw["stroke_opacity"] == 1.0 and kw["fill_opacity"] == 1.0
    assert "color" not in kw and kw["fill"] == (0, 0, 0) and "dashes" not in kw


def test_replay_items_matches_the_shared_replay_with_the_offset_negated() -> None:
    from raster_parser.commons.renderer._shapes import replay_item
    from raster_parser.commons.v1_commons.render import replay_items

    items = [
        ("l", (10.0, 10.0), (90.0, 10.5)),
        ("re", (20.0, 20.0, 40.0, 40.0)),
        ("qu", ((50.0, 20.0), (70.0, 20.0), (70.0, 40.0), (50.0, 40.0))),
        ("c", (10.0, 60.0), (30.0, 80.0), (50.0, 40.0), (70.0, 60.0)),
    ]
    with pymupdf.open() as doc:
        page = doc.new_page()
        a, b = page.new_shape(), page.new_shape()
        replay_items(a, items, 3.25, -1.5)
        for it in items:
            replay_item(b, it, -3.25, 1.5)
        assert a.draw_cont == b.draw_cont


def test_replay_items_rejects_unknown_ops() -> None:
    import pytest

    from raster_parser.commons.v1_commons.render import replay_items

    with pymupdf.open() as doc:
        with pytest.raises(ValueError, match="unhandled drawing op"):
            replay_items(doc.new_page().new_shape(), [("zz", (0, 0))])


def test_draw_vectors_replays_every_item(vector, tmp_path) -> None:
    v = vector(
        type="s",
        items=[
            ("l", (10.0, 10.0), (90.0, 10.0)),
            ("re", (20.0, 20.0, 40.0, 40.0)),
            ("qu", ((50.0, 20.0), (70.0, 20.0), (70.0, 40.0), (50.0, 40.0))),
            ("c", (10.0, 60.0), (30.0, 80.0), (50.0, 40.0), (70.0, 60.0)),
        ],
        bbox=(10.0, 10.0, 90.0, 80.0),
        width=1.0,
    )
    out = draw_vectors([v], tmp_path / "out.pdf", page_width=100, page_height=100)
    with pymupdf.open(out) as doc:
        (d,) = doc[0].get_drawings()
        assert [it[0] for it in d["items"]] == ["l", "re", "qu", "c"]
        assert pymupdf.Rect(d["items"][2][1].rect) == pymupdf.Rect(
            50, 20, 70, 40
        )  # a box, not a bow-tie
