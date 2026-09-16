import colorsys

from rastervec.Evaluation.inspector.layers import (
    OverlayItem,
    rgb_to_hex,
    seqno_rainbow_color_map,
    seqno_rainbow_colorer,
)


def _item(seqno) -> OverlayItem:
    attrs = {} if seqno is None else {"seqno": seqno}
    return OverlayItem(bbox=(0, 0, 1, 1), attrs=attrs)


def test_seqno_rainbow_color_map_empty():
    assert seqno_rainbow_color_map([]) == {}


def test_seqno_rainbow_color_map_single_item_is_red():
    item = _item(5)
    color_map = seqno_rainbow_color_map([item])
    assert color_map[id(item)] == "#ff0000"


def test_seqno_rainbow_color_map_all_equal_seqno_is_red():
    items = [_item(3), _item(3), _item(3)]
    color_map = seqno_rainbow_color_map(items)
    assert all(color == "#ff0000" for color in color_map.values())


def test_seqno_rainbow_color_map_min_red_max_violet():
    low, mid, high = _item(0), _item(5), _item(10)
    color_map = seqno_rainbow_color_map([low, mid, high])

    assert color_map[id(low)] == "#ff0000"
    assert color_map[id(high)] == rgb_to_hex(colorsys.hsv_to_rgb(0.75, 1.0, 1.0))
    assert color_map[id(mid)] not in (color_map[id(low)], color_map[id(high)])


def test_seqno_rainbow_color_map_excludes_missing_seqno():
    with_seqno = _item(1)
    without_seqno = _item(None)

    color_map = seqno_rainbow_color_map([with_seqno, without_seqno])

    assert id(with_seqno) in color_map
    assert id(without_seqno) not in color_map


def test_seqno_rainbow_colorer_uses_fallback_for_missing_item():
    low, high = _item(0), _item(10)
    colorer = seqno_rainbow_colorer([low, high], fallback="#123456")

    assert colorer(low) == "#ff0000"
    assert colorer(_item(20)) == "#123456"
