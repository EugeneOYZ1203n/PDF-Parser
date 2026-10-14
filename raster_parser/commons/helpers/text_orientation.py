"""Text direction <-> the old `rotate` / `glyph_orientation` values.

`rotate` is PyMuPDF's `insert_text(rotate=...)` angle in unrotated page space (counter-clockwise,
y down), so it maps to the unit direction `(cos r, -sin r)`. Going back, `rotate` is the
direction's angle snapped to a quarter turn, and `glyph_orientation` is how the text reads once
the page's `/Rotate` is applied: "vertical" at a displayed 90/270, "horizontal" at 0/180,
"other" when the direction is more than `AXIS_TOL_DEG` off every axis.
(Moved from the removed `models/dto_conversion.py`.)
"""
from __future__ import annotations

from math import atan2, cos, degrees, radians, sin

AXIS_TOL_DEG = 10.0


def rotate_to_direction(rotate: int) -> tuple[float, float]:
    r = radians(int(rotate) % 360)
    return (round(cos(r), 12) + 0.0, round(-sin(r), 12) + 0.0)


def direction_to_rotate(direction: tuple[float, float]) -> int:
    dx, dy = direction
    angle = degrees(atan2(-dy, dx)) % 360.0
    return int(round(angle / 90.0)) % 4 * 90


def glyph_orientation_for(direction: tuple[float, float], page_rotation: int = 0) -> str:
    dx, dy = direction
    angle = degrees(atan2(-dy, dx)) % 360.0
    off_axis = abs(angle - round(angle / 90.0) * 90.0)
    if off_axis > AXIS_TOL_DEG:
        return "other"
    displayed = (direction_to_rotate(direction) - int(page_rotation)) % 180
    return "vertical" if displayed == 90 else "horizontal"
