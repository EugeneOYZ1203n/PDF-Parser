"""Layer / colour / width separation: split Vectors into buckets so that grouping only ever runs
within one bucket -- two vectors in different layers, with different stroke/fill colours (and
opacities) or different stroke widths are never grouped together, however close they are.

Used by `v2_vector_class` (clustering). The width axis
takes a `key`: the default `width_key` rounds and ignores fill-only widths; the labelling tool
passes the exact `Vector.width`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable

from raster_parser.commons.logging_setup import get_logger
from raster_parser.models.vector import Vector

# per-call bucket counts: DEBUG is opt-in (see `get_logger`)
_LOG = get_logger("layer_separation", default_level=logging.INFO)


def separate_by_layer(vectors: list[Vector]) -> dict[str, list[Vector]]:
    groups: dict[str, list[Vector]] = defaultdict(list)
    for v in vectors:
        groups[v.layer or ""].append(v)
    _LOG.debug("separated %d vector(s) into %d layer(s)", len(vectors), len(groups))
    return dict(groups)


def separate_by_color(vectors: list[Vector]) -> dict[tuple, list[Vector]]:
    groups: dict[tuple, list[Vector]] = defaultdict(list)
    for v in vectors:
        groups[(v.color, v.fill, v.stroke_opacity, v.fill_opacity)].append(v)
    _LOG.debug("separated %d vector(s) into %d color/opacity groups", len(vectors), len(groups))
    return dict(groups)


def width_key(v: Vector) -> float | None:
    """The default width bucket: `None` for a fill-only vector (type "f" -- its `width` is
    leftover graphics state, never painted, so it must not split e.g. one CAD-text word's
    glyphs apart) or a vector with no width; otherwise the stroke width rounded to 0.01 pt so
    float noise doesn't split one visual weight into several buckets."""
    if v.type == "f" or v.width is None:
        return None
    return round(v.width, 2)


def separate_by_width(
    vectors: list[Vector],
    key: "Callable[[Vector], float | None]" = width_key,
) -> dict[float | None, list[Vector]]:
    """Groups by `key(vector)` (default `width_key`)."""
    groups: dict[float | None, list[Vector]] = defaultdict(list)
    for v in vectors:
        groups[key(v)].append(v)
    _LOG.debug("separated %d vector(s) into %d width group(s)", len(vectors), len(groups))
    return dict(groups)
