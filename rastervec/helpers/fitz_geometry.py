"""Geometry helpers that operate on live PyMuPDF objects (`fitz.Point`,
`fitz.Quad`, `fitz.Rect`, `fitz.Matrix`).

Kept separate from `helpers/geometry.py` (which is pure tuple math and does
not import pymupdf) so a module that only needs the tuple helpers -- e.g.
`models.py`, the evaluation metric suite -- never pulls pymupdf in
transitively. Used by the two Tkinter tools (`Evaluation/inspector/`,
`Evaluation/Labelling/manual_label.py`) for hover-metadata formatting.
"""
from __future__ import annotations

from math import atan2, degrees, hypot, sqrt

import pymupdf as fitz


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


def fitz_item(item: tuple) -> tuple:
    """The exact inverse of `plain_item`: convert one `Vector.items` entry
    (plain float tuples) back into the fitz-object item shape
    `get_drawings()['items']` itself uses -- `fitz.Point`/`Rect`/`Quad`
    per kind. Used by `models.Vector.to_pymupdf()`; trailing elements past
    the geometry (e.g. a "re" item's orientation flag) pass through
    unchanged, mirroring `plain_item`."""
    kind = item[0]
    if kind == "l":
        return (kind, fitz.Point(item[1]), fitz.Point(item[2]))
    if kind == "re":
        return (kind, fitz.Rect(item[1]), *item[2:])
    if kind == "qu":
        ul, ur, lr, ll = item[1]
        return (kind, fitz.Quad(fitz.Point(ul), fitz.Point(ur), fitz.Point(ll), fitz.Point(lr)))
    if kind == "c":
        return (kind, *[fitz.Point(p) for p in item[1:5]], *item[5:])
    return item


def plain_item(item: tuple) -> tuple:
    """Convert one raw `get_drawings()['items']` tuple (carrying fitz
    `Point`/`Rect`/`Quad` objects) into the same `(kind, *geometry)` shape
    with plain float tuples instead, so `models.Vector.items` stays
    fitz-free while still perfectly mirroring PyMuPDF's own item structure
    -- no VectorPath-style decomposition, just a type conversion of the
    same tuple. Any trailing elements past the geometry (e.g. a "re" item's
    orientation flag on newer PyMuPDF) pass through unchanged."""
    kind = item[0]
    if kind == "l":
        p1, p2 = fitz.Point(item[1]), fitz.Point(item[2])
        return (kind, (p1.x, p1.y), (p2.x, p2.y))
    if kind == "re":
        rect = fitz.Rect(item[1])
        return (kind, (rect.x0, rect.y0, rect.x1, rect.y1), *item[2:])
    if kind == "qu":
        quad = fitz.Quad(item[1])
        return (
            kind,
            ((quad.ul.x, quad.ul.y), (quad.ur.x, quad.ur.y), (quad.lr.x, quad.lr.y), (quad.ll.x, quad.ll.y)),
        )
    if kind == "c":
        points = [fitz.Point(p) for p in item[1:5]]
        return (kind, *[(p.x, p.y) for p in points], *item[5:])
    return item
