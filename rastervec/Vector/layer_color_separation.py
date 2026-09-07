"""Layer/color separation: splits extracted VectorPaths into buckets so
Vector_Classification's classification chain (see that package) only ever
runs within one (layer, color) bucket, never across buckets -- two paths
in different layers, or with different stroke/fill colors, are never
spatially merged together, regardless of how close they are on the page.
"""
from __future__ import annotations

from collections import defaultdict

from rastervec.logging_setup import get_logger
from rastervec.models import VectorPath

_LOG = get_logger("layer_color_separation")


def separate_by_layer(paths: list[VectorPath]) -> dict[str, list[VectorPath]]:
    groups: dict[str, list[VectorPath]] = defaultdict(list)
    for path in paths:
        groups[path.layer or ""].append(path)
    _LOG.debug("separated %d path(s) into %d layer(s)", len(paths), len(groups))
    return dict(groups)


def separate_by_color(paths: list[VectorPath]) -> dict[tuple, list[VectorPath]]:
    groups: dict[tuple, list[VectorPath]] = defaultdict(list)

    for path in paths:
        key = (
            path.stroke_color,
            path.fill_color,
            path.stroke_opacity,
            path.fill_opacity,
        )
        groups[key].append(path)

    _LOG.debug(
        "separated %d path(s) into %d color/opacity groups",
        len(paths),
        len(groups),
    )
    return dict(groups)
