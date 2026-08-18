"""Pure-math geometry helpers shared across pipeline stages.

Ported from pdf_model.py (the Tkinter inspector tool) so both tools use the
same, already-verified math rather than duplicating it independently.
"""
from __future__ import annotations

from math import atan2, degrees, hypot, sqrt

import pymupdf as fitz


def round_color(
    color: tuple[float, ...] | None,
) -> tuple[float, ...] | None:
    if not color:
        return None
    return tuple(round(c, 3) for c in color)


def point_angle(p1: fitz.Point, p2: fitz.Point) -> float:
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return degrees(atan2(dy, dx))


def line_length(p1: fitz.Point, p2: fitz.Point) -> float:
    return hypot(p2.x - p1.x, p2.y - p1.y)


def quad_angle(quad: fitz.Quad) -> float:
    return point_angle(quad.ul, quad.ur)


def quad_metadata(quad: fitz.Quad) -> dict:
    width = line_length(quad.ul, quad.ur)
    height = line_length(quad.ul, quad.ll)
    return {
        "angle": round(quad_angle(quad), 3),
        "width": round(width, 3),
        "height": round(height, 3),
    }


def rect_metadata(rect: fitz.Rect) -> dict:
    return {
        "x": round(rect.x0, 3),
        "y": round(rect.y0, 3),
        "width": round(rect.width, 3),
        "height": round(rect.height, 3),
        "area": round(rect.get_area(), 3),
        "aspect_ratio": (
            round(rect.width / rect.height, 4)
            if abs(rect.height) > 1e-9
            else None
        ),
    }


def format_matrix(matrix: fitz.Matrix) -> tuple:
    """Convert a PyMuPDF Matrix (a b c d e f) to a compact tuple."""
    return (
        round(matrix.a, 5),
        round(matrix.b, 5),
        round(matrix.c, 5),
        round(matrix.d, 5),
        round(matrix.e, 5),
        round(matrix.f, 5),
    )


def matrix_rotation(matrix: fitz.Matrix) -> float:
    """Estimate visual rotation from the matrix's x-axis."""
    return degrees(atan2(matrix.b, matrix.a))


def matrix_scale(matrix: fitz.Matrix) -> tuple[float, float]:
    """Estimate X/Y scale from a transformation matrix."""
    sx = sqrt(matrix.a * matrix.a + matrix.b * matrix.b)
    sy = sqrt(matrix.c * matrix.c + matrix.d * matrix.d)
    return (sx, sy)


def rect_gap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Euclidean gap between two axis-aligned (x0, y0, x1, y1) boxes.

    0.0 if they overlap or touch. Used for spatial clustering, where the
    gap between shapes (not the distance between their centers) is what
    determines whether they're "close."
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return hypot(dx, dy)


def union_bbox(
    boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    """Smallest axis-aligned (x0, y0, x1, y1) box containing every box in
    `boxes`. Used to get one bbox for a cluster/group of items from their
    individual bboxes."""
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return (x0, y0, x1, y1)


def make_oriented_quad(bbox: fitz.Rect, dx: float, dy: float) -> fitz.Quad:
    """Build a quad around bbox, oriented along the text direction (dx, dy).

    bbox.width/height are the *axis-aligned* extents, which only match the
    text's along-direction/normal-direction extents when the text is
    horizontal. For rotated text they can be swapped or otherwise wrong, so
    the bbox corners are projected onto the (dx, dy) / normal axes instead
    to recover the correct along/normal extents regardless of orientation.
    """
    length = hypot(dx, dy)
    if length < 1e-9:
        dx, dy = 1.0, 0.0
        length = 1.0
    dx /= length
    dy /= length

    nx = -dy
    ny = dx

    corners = [
        (bbox.x0, bbox.y0),
        (bbox.x1, bbox.y0),
        (bbox.x1, bbox.y1),
        (bbox.x0, bbox.y1),
    ]

    along_vals = [x * dx + y * dy for x, y in corners]
    normal_vals = [x * nx + y * ny for x, y in corners]

    along_min, along_max = min(along_vals), max(along_vals)
    normal_min, normal_max = min(normal_vals), max(normal_vals)

    half_along = (along_max - along_min) / 2.0
    half_normal = (normal_max - normal_min) / 2.0

    along_center = (along_max + along_min) / 2.0
    normal_center = (normal_max + normal_min) / 2.0

    cx = along_center * dx + normal_center * nx
    cy = along_center * dy + normal_center * ny

    def point(along: float, normal: float) -> fitz.Point:
        return fitz.Point(
            cx + dx * along + nx * normal,
            cy + dy * along + ny * normal,
        )

    ul = point(-half_along, -half_normal)
    ur = point(half_along, -half_normal)
    ll = point(-half_along, half_normal)
    lr = point(half_along, half_normal)

    return fitz.Quad(ul, ur, ll, lr)
