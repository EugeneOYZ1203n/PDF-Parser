"""The one standard drawing layer every PDF in the repo is built from.

Two functions per primitive kind:

- a **spec builder** (`text_spec`, `bbox_spec`, `quad_spec`,
  `polyline_spec`, `arrow_spec`, `dot_spec`, `vector_spec`, `image_spec`)
  turning a domain object into a frozen, fitz-free `*Spec` -- exactly the
  numbers PyMuPDF needs, inspectable in tests;
- a **drawer** (`draw_text`, `draw_path`, `draw_dot`, `draw_vectors`,
  `draw_image`) putting a spec onto a page.

`render_specs_pdf(page_meta, specs)` is the page-level entry point: a fresh
page sized to `page_meta`, every spec drawn (images -> vectors -> paths/dots
-> text, input order within each), one `Shape` + one `commit()` for all
paths/dots and one for all text (a commit per item adds a content stream
per item -- quadratic), and `/Rotate` applied last so nothing is ever placed
in rotated space. All coordinates are unrotated page space (see CLAUDE.md
"Coordinate spaces").

Text placement rule (`text_spec`): rotate to the text's own angle, then
scale to fit its own box -- never shear, never re-space words. The box is
the text's *oriented* box: its detect quad (`Text.quad_points`) projected
onto the text direction when present, else recovered from the axis-aligned
bbox envelope (`oriented_box`). Font size = box height / (ascender -
descender), so the glyph span fills the box height; a horizontal scale
along the text direction then fills the box width.

`insert_text(point, morph=(fixpoint, M))` maps everything it draws --
glyphs and `point` alike -- through `FLIP * M * FLIP` about `fixpoint`
(FLIP = y-mirror; confirmed empirically on PyMuPDF 1.28). So the page-space
(y-down) map `scale(sx, 1)` then `rotate(theta)` is passed as
`M = Matrix(sx, 1) * Matrix(-theta)` -- a diagonal times an orthogonal
matrix, which cannot shear -- and the string is laid out unmorphed with its
glyph box centred on the fixpoint, so the fixpoint *is* the box centre.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
import pymupdf as fitz

from rastervec.commons.renderer._shapes import replay_drawing_paths

if TYPE_CHECKING:
    from rastervec.commons.models import PageMeta, Text, Vector

Point = tuple[float, float]
Rgb = tuple[float, float, float]
Bbox = tuple[float, float, float, float]

# (text, page-space bbox, rotation degrees[, (r, g, b) 0..1]) -- a plain
# text box with no `Text` behind it (ground-truth labels, overlays).
TextBox = Union[tuple[str, Bbox, float], tuple[str, Bbox, float, Rgb]]

FONTNAME = "helv"
_FONT = fitz.Font(FONTNAME)
_ASC = _FONT.ascender
_DESC = _FONT.descender
_SPAN = _ASC - _DESC

# |cos^2 - sin^2| below this (within ~3 degrees of 45) makes the bbox
# envelope inversion ill-conditioned -- fall back to the font's own aspect.
_ENVELOPE_MIN_DET = 0.1


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TextSpec:
    text: str
    point: Point
    fontsize: float
    color: Rgb
    morph_fixpoint: Point
    morph_matrix: tuple[float, float, float, float, float, float]
    fontname: str = FONTNAME


@dataclass(frozen=True)
class PathSpec:
    """A bbox (`kind="rect"`), closed polygon or open polyline."""
    points: tuple[Point, ...]
    closed: bool
    color: Rgb | None
    fill: Rgb | None = None
    width: float = 1.0
    dashes: str | None = None
    fill_opacity: float | None = None
    kind: str = "poly"  # "rect" | "poly"


@dataclass(frozen=True)
class DotSpec:
    center: Point
    radius: float
    fill: Rgb


@dataclass(frozen=True, eq=False)
class VectorSpec:
    vector: "Vector"


@dataclass(frozen=True, eq=False)
class ImageSpec:
    rect: Bbox
    width: int
    height: int
    samples: bytes
    n: int  # colour channels: 1 (gray) or 3 (RGB)
    alpha: bool


Spec = Union[TextSpec, PathSpec, DotSpec, VectorSpec, ImageSpec]


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
def text_color(color: int | None) -> Rgb:
    """A packed sRGB int (as `Text.color` carries) -> (r, g, b) 0..1."""
    if color is None:
        return (0.0, 0.0, 0.0)
    return (((color >> 16) & 255) / 255, ((color >> 8) & 255) / 255, (color & 255) / 255)


def oriented_box(
    bbox: Bbox, angle_deg: float, *, quad: "tuple[Point, ...] | None" = None, text: str = "",
) -> tuple[float, float, float, float]:
    """`(cx, cy, w, h)`: the text's own centre, length along `angle_deg` and
    height across it.

    With `quad` (a detect quad, page space) its corners are projected onto
    the direction and its normal. Otherwise the axis-aligned `bbox` is the
    envelope of a `w x h` rectangle turned by the angle --
    `W = w|c| + h|s|`, `H = w|s| + h|c|` -- solved for `w, h`; near 45
    degrees (or on a non-positive solution, a bbox that isn't a clean
    envelope) the font's own natural aspect `k = w/h` for `text` is assumed
    instead, giving `h = (W + H) / ((k + 1)(|c| + |s|))`."""
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    if quad:
        along = [x * dx + y * dy for x, y in quad]
        normal = [-x * dy + y * dx for x, y in quad]
        a_mid = (min(along) + max(along)) / 2.0
        n_mid = (min(normal) + max(normal)) / 2.0
        return (
            a_mid * dx - n_mid * dy, a_mid * dy + n_mid * dx,
            max(along) - min(along), max(normal) - min(normal),
        )

    x0, y0, x1, y1 = bbox
    big_w, big_h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    c, s = abs(dx), abs(dy)
    det = c * c - s * s
    if abs(det) >= _ENVELOPE_MIN_DET:
        w = (big_w * c - big_h * s) / det
        h = (big_h * c - big_w * s) / det
        if w > 0 and h > 0:
            return (cx, cy, w, h)
    natural = _FONT.text_length(text, fontsize=1.0) if text else 0.0
    k = natural / _SPAN if natural > 0 else 1.0
    h = (big_w + big_h) / ((k + 1.0) * (c + s))
    return (cx, cy, k * h, h)


def text_spec(item: "Text | TextBox", *, color: Rgb | None = None) -> TextSpec | None:
    """The one Text -> PyMuPDF mapping. `item` is a `Text` (angle from its
    `direction`, box from `quad_points` else `bbox`, colour from its packed
    `color`) or a `TextBox` tuple. `color` overrides either. `None` for
    blank text or a degenerate box."""
    if isinstance(item, tuple):
        text, bbox, angle = item[0], item[1], float(item[2])
        quad = None
        rgb = item[3] if len(item) > 3 else (0.0, 0.0, 0.0)
    else:
        text, bbox, angle = item.text, item.bbox, item.angle()
        quad = getattr(item, "quad_points", None)
        rgb = text_color(item.color)
    if color is not None:
        rgb = color
    if not text or not text.strip():
        return None

    cx, cy, w, h = oriented_box(tuple(bbox), angle, quad=quad, text=text)
    if w <= 0 or h <= 0:
        return None
    fontsize = h / _SPAN
    natural = _FONT.text_length(text, fontsize=fontsize)
    if natural <= 0:
        return None
    m = fitz.Matrix(w / natural, 0, 0, 1, 0, 0) * fitz.Matrix(-angle)
    return TextSpec(
        text=text,
        point=(cx - natural / 2.0, cy + (_ASC + _DESC) / 2.0 * fontsize),
        fontsize=fontsize,
        color=tuple(float(v) for v in rgb[:3]),
        morph_fixpoint=(cx, cy),
        morph_matrix=(m.a, m.b, m.c, m.d, m.e, m.f),
    )


def draw_text(shape: "fitz.Shape", spec: TextSpec) -> None:
    shape.insert_text(
        fitz.Point(*spec.point), spec.text,
        fontsize=spec.fontsize, fontname=spec.fontname, color=spec.color,
        rotate=0, morph=(fitz.Point(*spec.morph_fixpoint), fitz.Matrix(*spec.morph_matrix)),
    )


# ---------------------------------------------------------------------------
# Paths: bbox / quad / polyline / arrow
# ---------------------------------------------------------------------------
def bbox_spec(
    bbox: Bbox, color: Rgb | None, *, width: float = 1.5, dashes: str | None = None,
    fill: Rgb | None = None, fill_opacity: float | None = None,
) -> PathSpec:
    x0, y0, x1, y1 = (float(v) for v in bbox)
    return PathSpec(
        points=((x0, y0), (x1, y0), (x1, y1), (x0, y1)), closed=True, color=color,
        fill=fill, width=width, dashes=dashes, fill_opacity=fill_opacity, kind="rect",
    )


def quad_spec(
    points, color: Rgb | None, *, width: float = 1.5, dashes: str | None = None,
    fill: Rgb | None = None, fill_opacity: float | None = None,
) -> PathSpec:
    """A closed polygon -- a detect quad (rotated, or clipped non-rectangular)."""
    return PathSpec(
        points=tuple((float(x), float(y)) for x, y in points), closed=True, color=color,
        fill=fill, width=width, dashes=dashes, fill_opacity=fill_opacity,
    )


def polyline_spec(points, color: Rgb, *, width: float = 1.0, dashes: str | None = None) -> PathSpec:
    return PathSpec(
        points=tuple((float(x), float(y)) for x, y in points), closed=False, color=color,
        width=width, dashes=dashes,
    )


def arrow_spec(
    center: Point, angle_deg: float, length: float, color: Rgb, *,
    width: float = 1.5, head_frac: float = 0.3, head_deg: float = 25.0,
) -> list[PathSpec]:
    """A shaft centred on `center` pointing along `angle_deg` (y-down,
    `helpers.geometry.transform_direction`'s convention -- the same one
    `Text.angle()` uses) plus a two-wing head at the tip."""
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    cx, cy = center
    tail = (cx - dx * length / 2.0, cy - dy * length / 2.0)
    tip = (cx + dx * length / 2.0, cy + dy * length / 2.0)
    head = length * head_frac
    wings = []
    for sign in (1.0, -1.0):
        r = math.radians(sign * head_deg)
        wx = -dx * math.cos(r) + dy * math.sin(r)
        wy = -dx * math.sin(r) - dy * math.cos(r)
        wings.append((tip[0] + wx * head, tip[1] + wy * head))
    return [
        polyline_spec((tail, tip), color, width=width),
        polyline_spec((wings[0], tip, wings[1]), color, width=width),
    ]


def draw_path(shape: "fitz.Shape", spec: PathSpec) -> None:
    """Draws and finishes one path (one `finish` per path: translucent
    fills still darken where they overlap)."""
    if len(spec.points) < 2:
        return
    if spec.kind == "rect":
        (x0, y0), _, (x1, y1), _ = spec.points
        shape.draw_rect(fitz.Rect(x0, y0, x1, y1))
    else:
        pts = [fitz.Point(*p) for p in spec.points]
        shape.draw_polyline(pts + [pts[0]] if spec.closed else pts)
    kwargs: dict = {"color": spec.color, "width": spec.width, "closePath": spec.closed}
    if spec.fill is not None:
        kwargs["fill"] = spec.fill
    if spec.fill_opacity is not None:
        kwargs["fill_opacity"] = spec.fill_opacity
    if spec.dashes:
        kwargs["dashes"] = spec.dashes
    shape.finish(**kwargs)


# ---------------------------------------------------------------------------
# Dots
# ---------------------------------------------------------------------------
def dot_spec(center: Point, color: Rgb, radius: float = 0.8) -> DotSpec:
    return DotSpec(center=(float(center[0]), float(center[1])), radius=radius, fill=color)


def draw_dot(shape: "fitz.Shape", spec: DotSpec) -> None:
    shape.draw_circle(fitz.Point(*spec.center), spec.radius)
    shape.finish(color=None, fill=spec.fill, width=0)


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------
def vector_spec(vector: "Vector", *, recolor: Rgb | None = None, min_width: float = 1.0) -> VectorSpec:
    """`recolor=None` keeps the vector's real paint (final output); a colour
    turns it into a plain stroked copy in that colour (debug layers), its
    real fill/dashes/blend dropped so the colour isn't drowned out."""
    if recolor is None:
        return VectorSpec(vector)
    import dataclasses

    return VectorSpec(dataclasses.replace(
        vector, type="s", color=tuple(recolor), fill=None, dashes=None,
        width=max(vector.width or 0.0, min_width), closePath=False,
        blendmode="Normal", opacity=1.0, stroke_opacity=1.0, fill_opacity=None,
    ))


def draw_vectors(page: "fitz.Page", specs: "list[VectorSpec]") -> None:
    """Page-level (not Shape-level): `replay_drawing_paths` commits its own
    Shape per (blendmode, opacity) run so it can wrap blended runs in an
    ExtGState."""
    if specs:
        replay_drawing_paths(page, [s.vector for s in specs])


# ---------------------------------------------------------------------------
# Raster images
# ---------------------------------------------------------------------------
def orient_for_page(array: np.ndarray, transform) -> np.ndarray:
    """Reorient an image-pixel-frame `array` so rows run down and columns
    run right in page space, given the placement's `get_image_info`
    transform (unit square -> page). Quarter turns/flips exact; a general
    skew snaps to its nearest quarter turn."""
    if transform is None:
        return array
    a, b, c, d, _e, _f = transform
    if abs(a) >= abs(b):  # image x -> page x
        out = array
        if a < 0:
            out = out[:, ::-1]
        if d < 0:
            out = out[::-1]
    else:                 # image x -> page y (quarter turn)
        out = np.swapaxes(array, 0, 1)
        if b < 0:
            out = out[::-1]
        if c < 0:
            out = out[:, ::-1]
    return out


def image_spec(array, rect: Bbox, *, transform=None) -> ImageSpec:
    """`array`: HxW gray, HxWx3 RGB or HxWx4 RGBA uint8 (or a PIL image),
    stretched over page-space `rect`. `transform`, when the array is still
    in an embedded image's own pixel frame, reorients it first."""
    arr = np.asarray(array)
    arr = np.ascontiguousarray(orient_for_page(arr, transform), dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        n, alpha = 1, False
    else:
        n, alpha = 3, arr.shape[2] == 4
        arr = np.ascontiguousarray(arr[:, :, :4] if alpha else arr[:, :, :3])
    return ImageSpec(
        rect=tuple(float(v) for v in rect), width=w, height=h,
        samples=arr.tobytes(), n=n, alpha=alpha,
    )


def draw_image(page: "fitz.Page", spec: ImageSpec) -> None:
    cs = fitz.csGRAY if spec.n == 1 else fitz.csRGB
    pixmap = fitz.Pixmap(cs, spec.width, spec.height, spec.samples, int(spec.alpha))
    page.insert_image(fitz.Rect(*spec.rect), pixmap=pixmap, keep_proportion=False)


# ---------------------------------------------------------------------------
# Page level
# ---------------------------------------------------------------------------
def draw_specs(page: "fitz.Page", specs: "list[Spec]") -> None:
    """Draw `specs` onto an existing (unrotated) page, in layer order:
    images, vectors, paths + dots, text."""
    images = [s for s in specs if isinstance(s, ImageSpec)]
    vectors = [s for s in specs if isinstance(s, VectorSpec)]
    marks = [s for s in specs if isinstance(s, (PathSpec, DotSpec))]
    texts = [s for s in specs if isinstance(s, TextSpec)]

    for spec in images:
        draw_image(page, spec)
    draw_vectors(page, vectors)
    if marks:
        shape = page.new_shape()
        for spec in marks:
            if isinstance(spec, PathSpec):
                draw_path(shape, spec)
            else:
                draw_dot(shape, spec)
        shape.commit()
    if texts:
        shape = page.new_shape()
        for spec in texts:
            draw_text(shape, spec)
        shape.commit()


def new_page_doc(page_meta: "PageMeta") -> "tuple[fitz.Document, fitz.Page]":
    """A fresh one-page document in unrotated page space (`/Rotate` is the
    caller's last step -- see `render_specs_pdf`)."""
    doc = fitz.open()
    page = doc.new_page(width=page_meta.width, height=page_meta.height)
    return doc, page


def render_specs_pdf(page_meta: "PageMeta", specs: "list[Spec]", *, deflate: bool = False) -> bytes:
    """One-page PDF bytes of `specs` on a fresh page sized to `page_meta`,
    `/Rotate` set last."""
    doc, page = new_page_doc(page_meta)
    try:
        draw_specs(page, [s for s in specs if s is not None])
        page.set_rotation(page_meta.rotation)
        return doc.tobytes(deflate=deflate)
    finally:
        doc.close()
