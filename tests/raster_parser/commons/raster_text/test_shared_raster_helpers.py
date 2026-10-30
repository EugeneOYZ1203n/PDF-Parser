"""The raster prelude and the raster backends use one colour conversion and one debug-raster
placement, not their own copies."""

from __future__ import annotations


def test_no_local_colour_or_placement_copies():
    from raster_parser.commons.raster_text import raster_text
    from raster_parser.phase_raster_vectorize.junction import adapter

    for module in (raster_text, adapter):
        assert not hasattr(module, "_hex_rgb") and not hasattr(module, "_rgb_hex")
    assert not hasattr(adapter, "_place_raster")
    assert adapter.place_raster is raster_text.place_raster


def test_rgb255_to_hex():
    from raster_parser.commons.helpers.geometry import rgb255_to_hex

    assert rgb255_to_hex((255, 0, 16.7)) == "#ff0010"
    assert rgb255_to_hex((1, 2, 3, 4)) == "#010203"
