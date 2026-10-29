"""`commons.helpers.layer_separation`: layer / colour / width buckets."""

from __future__ import annotations

from raster_parser.commons.helpers.layer_separation import (
    separate_by_color,
    separate_by_layer,
    separate_by_width,
    width_key,
)


def test_layer_and_colour_buckets(vector):
    a = vector(layer="A", color=(0, 0, 0))
    b = vector(layer=None, color=(0, 0, 0))
    c = vector(layer="A", color=(1, 0, 0))
    assert separate_by_layer([a, b, c]) == {"A": [a, c], "": [b]}
    assert list(separate_by_color([a, b, c]).values()) == [[a, b], [c]]


def test_width_buckets_round_and_ignore_fill_only_widths(vector):
    thin = vector(type="s", width=0.5)
    thin_noise = vector(type="s", width=0.501)
    fill = vector(type="f", width=3.0, fill=(0, 0, 0), color=None)
    assert width_key(fill) is None
    assert separate_by_width([thin, thin_noise, fill]) == {0.5: [thin, thin_noise], None: [fill]}


def test_width_buckets_take_a_custom_key(vector):
    thin = vector(type="s", width=0.5)
    thin_noise = vector(type="s", width=0.501)
    exact = separate_by_width([thin, thin_noise], key=lambda v: v.width)
    assert exact == {0.5: [thin], 0.501: [thin_noise]}
