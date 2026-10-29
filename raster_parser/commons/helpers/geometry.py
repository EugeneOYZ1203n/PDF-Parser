"""Pure tuple-math geometry helpers shared across pipeline stages.

No pymupdf import -- everything here operates on plain
`(x0, y0, x1, y1)` bboxes, `(x, y)` points and `(r, g, b, ...)` colours, so
any stage's output stays testable without a live PyMuPDF document. Helpers
that need `pymupdf.Point`/`Quad`/`Rect`/`Matrix` objects live in
`helpers/pymupdf_geometry.py` instead.
"""

from __future__ import annotations

from math import hypot

# Axis-aligned bounding box: (x0, y0, x1, y1) in PDF page space.
BBox = tuple[float, float, float, float]
# A 2-D point: (x, y).
Point = tuple[float, float]
# Four corners of a (possibly rotated) quad, in (ul, ur, lr, ll) order.
Quad = tuple[Point, Point, Point, Point]

# PDF user space has 72 points per inch, so a render at `dpi` maps 1 pt to `dpi / 72` px.
PDF_POINTS_PER_INCH = 72.0


def round_color(
    color: tuple[float, ...] | None,
) -> tuple[float, ...] | None:
    if not color:
        return None
    return tuple(round(c, 3) for c in color)


def rgb01_to_hex(color: tuple[float, ...]) -> str:
    """An RGB colour with components in [0, 1] as `#rrggbb`; out-of-range components clamp."""
    return "#%02x%02x%02x" % tuple(min(255, max(0, round(c * 255))) for c in color)


def union_bbox(boxes: list[BBox]) -> BBox:
    """Smallest axis-aligned (x0, y0, x1, y1) box containing every box in
    `boxes`. Used to get one bbox for a cluster/group of items from their
    individual bboxes."""
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return (x0, y0, x1, y1)


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


def compute_origin(
    bbox: BBox,
    direction: Point,
    baseline_point: Point | None = None,
) -> Point:
    """A baseline leading-edge point for `bbox`, oriented along `direction`
    -- the same along/normal projection `make_oriented_quad` uses, generalized
    from native text's old per-word `_word_origin` so native and OCR `Text`
    populate `origin` the same way.

    Along the direction, the point sits at the bbox's leading (minimum-
    projection) edge. Perpendicular to it:

    - if `baseline_point` is given (native text's real `get_text("dict")`
      span `origin`), the point keeps that baseline's own normal offset --
      so the result lands on the actual glyph baseline, correct for
      rotated/vertical text, not just the bbox centre;
    - otherwise (OCR results, which have no independent baseline sample)
      it falls back to the bbox's own normal-axis centre.
    """
    x0, y0, x1, y1 = bbox
    dx, dy = direction
    length = hypot(dx, dy)
    if length < 1e-9:
        dx, dy = 1.0, 0.0
        length = 1.0
    dx, dy = dx / length, dy / length
    nx, ny = -dy, dx
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    along_min = min(x * dx + y * dy for x, y in corners)
    if baseline_point is not None:
        normal_offset = baseline_point[0] * nx + baseline_point[1] * ny
    else:
        normal_vals = [x * nx + y * ny for x, y in corners]
        normal_offset = (min(normal_vals) + max(normal_vals)) / 2.0
    return (along_min * dx + normal_offset * nx, along_min * dy + normal_offset * ny)


# --------------------------------------------------------------------------
# Vector.items geometry -- operates on the plain-tuple item shape stored on
# Vector.items ((kind, *geometry), pymupdf-free -- see helpers/pymupdf_geometry.py's
# plain_item for the one-time raw-get_drawings()-item conversion). Used by
# vector filters that need a Vector's own sub-item geometry
# for scoring, without Vector ever being decomposed into standalone items.
# --------------------------------------------------------------------------


def item_points(item: tuple) -> list[Point]:
    """Every point of one Vector.items entry, in the same order PyMuPDF's
    own item tuple carries them."""
    kind = item[0]
    if kind == "l":
        return [item[1], item[2]]
    if kind == "re":
        x0, y0, x1, y1 = item[1]
        return [(x0, y0), (x1, y1)]
    if kind == "qu":
        return list(item[1])
    if kind == "c":
        return list(item[1:5])
    return []


def item_bbox(item: tuple) -> BBox:
    """Axis-aligned bbox of one Vector.items entry, computed on demand
    (never cached on Vector itself)."""
    points = item_points(item)
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))
