"""Pure tuple-math geometry helpers shared across pipeline stages.

No pymupdf import -- everything here operates on plain
`(x0, y0, x1, y1)` bboxes, `(x, y)` points and `(r, g, b, ...)` colours, so
any stage's output stays testable without a live PyMuPDF document. Helpers
that need `fitz.Point`/`Quad`/`Rect`/`Matrix` objects live in
`helpers/fitz_geometry.py` instead.
"""
from __future__ import annotations

from math import cos, hypot, radians, sin

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


def compute_origin(bbox: BBox, direction: Point) -> Point:
    """A baseline leading-edge point for `bbox`, oriented along `direction`
    -- the same along/normal projection `make_oriented_quad` uses, generalized
    from native text's old per-word `_word_origin` so native and OCR `Text`
    populate `origin` the same way. Uses the bbox's own normal-axis center
    (no separate baseline offset input, unlike the old span-origin-aware
    version) since OCR results have no independent baseline sample."""
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
    normal_vals = [x * nx + y * ny for x, y in corners]
    normal_center = (min(normal_vals) + max(normal_vals)) / 2.0
    return (along_min * dx + normal_center * nx, along_min * dy + normal_center * ny)


# --------------------------------------------------------------------------
# Vector.items geometry -- operates on the plain-tuple item shape stored on
# Vector.items ((kind, *geometry), fitz-free -- see helpers/fitz_geometry.py's
# plain_item for the one-time raw-get_drawings()-item conversion). Used by
# Vector_Classification filters that need a Vector's own sub-item geometry
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


def transform_point(p: Point, offset: Point, rotation_deg: float) -> Point:
    """Rotate `p` about (0, 0) by `rotation_deg` (degrees, counter-clockwise
    in PDF's y-down... actually y-up-agnostic -- just the standard 2-D
    rotation matrix), then translate by `offset`."""
    theta = radians(rotation_deg)
    cos_t, sin_t = cos(theta), sin(theta)
    x, y = p
    return (x * cos_t - y * sin_t + offset[0], x * sin_t + y * cos_t + offset[1])


def transform_item(item: tuple, offset: Point, rotation_deg: float) -> tuple:
    """Rotate+translate every point of one Vector.items entry, keeping its
    (kind, *extra) shape -- e.g. a "re" item's trailing orientation field (if
    present) passes through untouched, only its rect corners move."""
    kind = item[0]

    def tp(p: Point) -> Point:
        return transform_point(p, offset, rotation_deg)

    if kind == "l":
        return (kind, tp(item[1]), tp(item[2]))
    if kind == "re":
        x0, y0, x1, y1 = item[1]
        (nx0, ny0), (nx1, ny1) = tp((x0, y0)), tp((x1, y1))
        new_rect = (min(nx0, nx1), min(ny0, ny1), max(nx0, nx1), max(ny0, ny1))
        return (kind, new_rect, *item[2:])
    if kind == "qu":
        return (kind, tuple(tp(p) for p in item[1]))
    if kind == "c":
        return (kind, *[tp(p) for p in item[1:5]], *item[5:])
    return item


def transform_bbox(bbox: BBox, offset: Point, rotation_deg: float) -> BBox:
    """Rotate-then-translate an axis-aligned bbox's four corners by
    `(rotation_deg, offset)` (see `transform_point`'s exact convention) and
    return the new axis-aligned bbox of the transformed corners. Used to
    restore OCR `Text` geometry from a `UniqueSegment`'s canonical frame
    back onto one real `SegmentMeta` occurrence (see
    `pipelines/sub_pipelines/ocr.py::restore_cluster_texts`)."""
    x0, y0, x1, y1 = bbox
    corners = [transform_point(p, offset, rotation_deg) for p in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def transform_direction(direction: Point, rotation_deg: float) -> Point:
    """Rotate a direction unit vector by `rotation_deg` about the origin
    -- no translation, since a direction is a vector, not a point."""
    return transform_point(direction, offset=(0.0, 0.0), rotation_deg=rotation_deg)


def transform_vector(v, *, offset: Point, rotation_deg: float):
    """Rotate+translate every item of `v` (and its `rect`/`scissor`),
    returning a new Vector -- never touches item *structure*, only geometry.
    Used to normalize a Segment's Vectors to a canonical (origin, upright)
    frame, and to invert that transform when restoring a UniqueSegment's OCR
    Text back onto each of its real page-space occurrences."""
    from dataclasses import replace

    new_items = [transform_item(item, offset, rotation_deg) for item in v.items]
    points = [p for item in new_items for p in item_points(item)]
    if points:
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        new_rect = (min(xs), min(ys), max(xs), max(ys))
    else:
        new_rect = v.rect

    new_scissor = None
    if v.scissor:
        sx0, sy0 = transform_point((v.scissor[0], v.scissor[1]), offset, rotation_deg)
        sx1, sy1 = transform_point((v.scissor[2], v.scissor[3]), offset, rotation_deg)
        new_scissor = (min(sx0, sx1), min(sy0, sy1), max(sx0, sx1), max(sy0, sy1))

    return replace(v, items=new_items, rect=new_rect, scissor=new_scissor)
