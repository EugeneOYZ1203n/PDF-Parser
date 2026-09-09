"""Layer/color separation: splits extracted Vectors into buckets so
Vector_Classification's classification chain (see that package) only ever
runs within one (layer, color) bucket, never across buckets -- two vectors
in different layers, or with different stroke/fill colors, are never
spatially merged together, regardless of how close they are on the page.
"""
from __future__ import annotations

from collections import defaultdict

from rastervec.logging_setup import get_logger
from rastervec.models import Vector

_LOG = get_logger("layer_color_separation")


def separate_by_layer(vectors: list[Vector]) -> dict[str, list[Vector]]:
    groups: dict[str, list[Vector]] = defaultdict(list)
    for v in vectors:
        groups[v.layer or ""].append(v)
    _LOG.debug("separated %d vector(s) into %d layer(s)", len(vectors), len(groups))
    return dict(groups)


def separate_by_color(vectors: list[Vector]) -> dict[tuple, list[Vector]]:
    groups: dict[tuple, list[Vector]] = defaultdict(list)

    for v in vectors:
        key = (v.color, v.fill, v.stroke_opacity, v.fill_opacity)
        groups[key].append(v)

    _LOG.debug(
        "separated %d vector(s) into %d color/opacity groups",
        len(vectors),
        len(groups),
    )
    return dict(groups)
