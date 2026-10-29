"""Text direction from the old `rotate` value.

`rotate` is PyMuPDF's `insert_text(rotate=...)` angle in unrotated page space (counter-clockwise,
y down), so it maps to the unit direction `(cos r, -sin r)`.
"""

from __future__ import annotations

from math import cos, radians, sin

# Digits the direction is rounded to, so quarter turns give exact 0 / +-1 components.
_DIRECTION_DIGITS = 12


def rotate_to_direction(rotate: int) -> tuple[float, float]:
    r = radians(int(rotate) % 360)
    return (round(cos(r), _DIRECTION_DIGITS) + 0.0, round(-sin(r), _DIRECTION_DIGITS) + 0.0)
