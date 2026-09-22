"""Color utilities for the inspector: `rgb_to_hex` plus the seqno-rainbow
coloring scheme used to visualize a drawing's own item draw-order."""
from __future__ import annotations

import colorsys
from typing import Callable

from rastervec.Evaluation.inspector.layer_types import OverlayItem


def rgb_to_hex(color) -> str:
    """Convert an RGB tuple in [0, 1] to a Tkinter hex colour."""

    if not color:
        return "#000000"

    return "#%02x%02x%02x" % tuple(
        round(
            max(
                0.0,
                min(1.0, c),
            )
            * 255
        )
        for c in color
    )


def seqno_rainbow_color_map(items: list[OverlayItem]) -> dict[int, str]:
    """Map each item (by id()) to a red(min seqno)->violet(max seqno) color.

    Hue sweeps 0 degrees (red) through orange/yellow/green/blue to 270
    degrees (violet), linearly by each item's "seqno" attr relative to the
    min/max seqno found in `items`. Items missing a usable seqno are
    omitted from the result.
    """

    seqnos = [
        item.attrs.get("seqno")
        for item in items
        if isinstance(item.attrs.get("seqno"), int)
    ]

    if not seqnos:
        return {}

    lo = min(seqnos)
    hi = max(seqnos)
    spread = hi - lo

    result: dict[int, str] = {}

    for item in items:
        seqno = item.attrs.get("seqno")

        if not isinstance(seqno, int):
            continue

        fraction = 0.0 if spread == 0 else (seqno - lo) / spread
        hue = fraction * 0.75  # 0.75 * 360 = 270 degrees (violet)

        result[id(item)] = rgb_to_hex(colorsys.hsv_to_rgb(hue, 1.0, 1.0))

    return result


def seqno_rainbow_colorer(
    items: list[OverlayItem],
    fallback: str,
) -> Callable[[OverlayItem], str]:
    """Return a per-item color function, computed once over `items`."""

    color_map = seqno_rainbow_color_map(items)

    def _color(item: OverlayItem) -> str:
        return color_map.get(id(item), fallback)

    return _color
