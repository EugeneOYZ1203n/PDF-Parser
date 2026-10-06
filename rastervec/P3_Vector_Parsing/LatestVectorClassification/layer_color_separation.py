"""Layer/color/width separation: splits extracted Vectors into buckets so
the pipeline's clustering step only ever runs within one bucket, never
across buckets -- two vectors in different layers, with different
stroke/fill colors, or (for `separate_by_width`, applied by
`classify_vectors` on top of the other two) different stroke widths, are
never spatially merged together, regardless of how close they are on the
page.
"""
from __future__ import annotations

from collections import defaultdict

from rastervec.commons.logging_setup import get_logger
from rastervec.commons.models import Vector

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


def width_key(v: Vector) -> float | None:
    """`separate_by_width`'s bucket key: `None` for a fill-only vector
    (type "f" -- its `width` is leftover graphics state, never painted, so
    it must not split e.g. one CAD-text word's glyphs apart) or a vector
    with no width; otherwise the stroke width rounded to 0.01 pt so float
    noise doesn't split one visual weight into several buckets."""
    if v.type == "f" or v.width is None:
        return None
    return round(v.width, 2)


def separate_by_width(vectors: list[Vector]) -> dict[float | None, list[Vector]]:
    """Groups by `width_key` -- the third separation axis on top of
    `separate_by_layer`/`separate_by_color`, so stroked content sharing a
    layer/color but drawn with different stroke weights is never merged
    into the same cluster."""
    groups: dict[float | None, list[Vector]] = defaultdict(list)
    for v in vectors:
        groups[width_key(v)].append(v)
    _LOG.debug("separated %d vector(s) into %d width group(s)", len(vectors), len(groups))
    return dict(groups)
