"""Pure tuple-math geometry helpers shared across pipeline stages.

No pymupdf import -- everything here operates on plain
`(x0, y0, x1, y1)` bboxes, `(x, y)` points and `(r, g, b, ...)` colours, so
any stage's output stays testable without a live PyMuPDF document. Helpers
that need `fitz.Point`/`Quad`/`Rect`/`Matrix` objects live in
`helpers/fitz_geometry.py` instead.
"""
from __future__ import annotations

from math import hypot

# Axis-aligned bounding box: (x0, y0, x1, y1) in PDF page space.
BBox = tuple[float, float, float, float]
# A 2-D point: (x, y).
Point = tuple[float, float]
# Four corners of a (possibly rotated) quad, in (ul, ur, lr, ll) order.
Quad = tuple[Point, Point, Point, Point]

# PDF user-space units per inch -- the constant behind every `dpi / 72.0`
# render-zoom in the renderer / OCR / FAST stages.
PDF_POINTS_PER_INCH = 72.0


def round_color(
    color: tuple[float, ...] | None,
) -> tuple[float, ...] | None:
    if not color:
        return None
    return tuple(round(c, 3) for c in color)


def rect_gap(a: BBox, b: BBox) -> float:
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


def union_bbox(boxes: list[BBox]) -> BBox:
    """Smallest axis-aligned (x0, y0, x1, y1) box containing every box in
    `boxes`. Used to get one bbox for a cluster/group of items from their
    individual bboxes."""
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return (x0, y0, x1, y1)


def dims(bbox: BBox) -> tuple[float, float]:
    """(width, height) of an axis-aligned box."""
    x0, y0, x1, y1 = bbox
    return (x1 - x0, y1 - y0)


def max_dimension(bbox: BBox) -> float:
    """The larger of a box's width/height; never negative."""
    x0, y0, x1, y1 = bbox
    return max(x1 - x0, 0.0, y1 - y0)


def is_dashed(dashes: str | None) -> bool:
    """PyMuPDF's "dashes" is a PDF dash-array string like "[] 0" (no dash)
    or "[3 2] 0" (dashed). An empty array means solid -- a plain
    `bool(dashes)` check is wrong here since "[] 0" is itself a non-empty,
    truthy string. Shared by Vector/vector.py and Vector_Classification/
    classification.py (both need the same solid-vs-dashed classification
    from the same raw PyMuPDF field)."""
    if not dashes:
        return False
    return not dashes.strip().startswith("[]")


def bbox_area(b: BBox) -> float:
    """Area of an axis-aligned (x0, y0, x1, y1) box; 0.0 for a degenerate
    (zero- or negative-extent) box."""
    return max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)


def bbox_intersection_area(a: BBox, b: BBox) -> float:
    """Overlap area of two axis-aligned (x0, y0, x1, y1) boxes; 0.0 if they
    don't overlap. Single source of truth for the overlap term in bbox_iou
    and bbox_coverage (Evaluation/Evaluate's metrics.py)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def bboxes_intersect(a: BBox, b: BBox) -> bool:
    """True if two axis-aligned boxes overlap or touch (shared edge counts)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 <= bx1 and bx0 <= ax1 and ay0 <= by1 and by0 <= ay1


def bbox_contains(bbox: BBox, x: float, y: float) -> bool:
    """True if the point (x, y) lies within (or on the edge of) `bbox`."""
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def bbox_fully_contains(a: BBox, b: BBox) -> bool:
    """True if a and b overlap and one fully contains (or equals) the
    other. No intersection returns False -- used by clustering to keep a
    fully-enclosed item (e.g. text inside a drawing's frame) from ever
    merging with the box that encloses it."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter_area = (ix1 - ix0) * (iy1 - iy0)
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    if area_a <= 0.0 or area_b <= 0.0:
        return False
    return inter_area >= area_a - 1e-9 or inter_area >= area_b - 1e-9


def bbox_coverage(a: BBox, b: BBox) -> float:
    """Fraction of box `a`'s area that box `b` covers -- intersection area
    over area(a), 0.0 when `a` is degenerate. Asymmetric: call
    bbox_coverage(gt, pred) for "how much of the ground truth is covered",
    bbox_coverage(pred, gt) for "how much of the prediction lands on text".
    Used by Evaluation/Evaluate's metrics.py for many-to-one matching."""
    area_a = bbox_area(a)
    if area_a <= 0.0:
        return 0.0
    return bbox_intersection_area(a, b) / area_a


def bbox_iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two axis-aligned (x0, y0, x1, y1) boxes,
    0.0 if they don't overlap. Used to match a predicted cluster bbox
    against a ground-truth text bbox (Evaluation/Labelling's auto_label.py,
    Evaluation/Evaluate's metrics.py)."""
    intersection = bbox_intersection_area(a, b)
    if intersection <= 0.0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def make_oriented_quad(bbox: BBox, dx: float, dy: float) -> Quad:
    """Build a quad around `bbox`, oriented along the text direction
    `(dx, dy)`, returned as ``(ul, ur, lr, ll)`` `(x, y)` tuples.

    A bbox's width/height are the *axis-aligned* extents, which only match
    the text's along-direction/normal-direction extents when the text is
    horizontal. For rotated text they can be swapped or otherwise wrong, so
    the bbox corners are projected onto the (dx, dy) / normal axes instead
    to recover the correct along/normal extents regardless of orientation.
    """
    x0, y0, x1, y1 = bbox
    length = hypot(dx, dy)
    if length < 1e-9:
        dx, dy = 1.0, 0.0
        length = 1.0
    dx /= length
    dy /= length

    nx = -dy
    ny = dx

    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
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

    def point(along: float, normal: float) -> Point:
        return (cx + dx * along + nx * normal, cy + dy * along + ny * normal)

    ul = point(-half_along, -half_normal)
    ur = point(half_along, -half_normal)
    ll = point(-half_along, half_normal)
    lr = point(half_along, half_normal)

    return (ul, ur, lr, ll)
