"""Helpers behind `filter_showcase.ipynb` -- presentation PNGs showing how
four of P3 LatestVectorClassification's filters work on one page:
collinear drawing, pattern lattice, length outliers and crossings.

Nothing here re-implements a filter: `run` calls `classify_vectors`
(`keep_steps=True`) and every image reads either its per-step categories or
the very functions those steps call (`group_collinear`, `_lattice_groups`,
`length_outliers`, `crossed_grid`, ...) on the same per-bucket inputs, so a
picture shows exactly what the pipeline did. The one new piece of geometry
is `crossing_points` -- the *locations* of the crossings that
`line_crossing_counts` only counts.

Rendering goes through `commons/renderer/draw.py`: recoloured highlight /
context layers as `stroke_batch_spec` polylines (bulk-fast, widths in
pixels so full-page and crop images read the same), the "after filter"
images as `vector_spec` real paint (what Phase 4 draws), the original as
the source page's own pixmap. Crops are page-space (unrotated) boxes mapped
through `rotation_matrix` before clipping, so rotated pages crop right.

Page-wide images (collinear, pattern) are zoomed by `zoom_window`: the
square window holding the most of both what the filter dropped and what it
kept (score = min of the two counts).
"""
from __future__ import annotations

import colorsys
import io
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont

from rastervec.commons.helpers.geometry import union_bbox
from rastervec.commons.models import PageMeta, Vector
from rastervec.commons.paths import output_dir
from rastervec.commons.renderer.draw import (
    dot_batch_spec, render_specs_pdf, stroke_batch_spec, vector_spec,
)
from rastervec.P1_Reading_Native.reader import Reader
from rastervec.P1_Reading_Native.vector_extract import extract_vectors
from rastervec.P3_Vector_Parsing.LatestVectorClassification import pattern_lattice as pl
from rastervec.P3_Vector_Parsing.LatestVectorClassification.classify_vectors import (
    CROSSED_CATEGORY, PATTERN_BUCKET_CATEGORY, PATTERN_CATEGORY, PATTERN_REJECTED_CATEGORY,
    ClassificationResult, classify_vectors, crossed_grid, length_outliers,
)
from rastervec.P3_Vector_Parsing.LatestVectorClassification.config import (
    ANGLE_TOL_DEG, COLLINEAR_OFFSET_TOL_PT, CROSS_EPS_PT, GLOBAL_ANGLE_MIN_GROUP_SIZE,
    LENGTH_OUTLIER_STD, MIN_CROSSINGS, MIN_PARALLEL_GROUP_SIZE, PATTERN_MAX_GROUP,
    PATTERN_MAX_GROUP_1D, PATTERN_MIN_BUCKET, STRAIGHT_TOL_PT,
)
from rastervec.P3_Vector_Parsing.LatestVectorClassification.line_geometry import (
    axial_distance, crossing_matrix, group_angle, group_collinear, group_parallel,
    item_pieces, split_straight,
)

RGB = tuple[float, float, float]
Bbox = tuple[float, float, float, float]

CONTEXT: RGB = (0.85, 0.85, 0.85)   # page vectors outside the subject
SUBJECT: RGB = (0.55, 0.55, 0.55)   # the subject's own, un-highlighted vectors
RED: RGB = (0.85, 0.1, 0.1)
BLUE: RGB = (0.1, 0.35, 0.9)
ORANGE: RGB = (1.0, 0.5, 0.0)
AMBER: RGB = (0.85, 0.7, 0.0)
PURPLE: RGB = (0.55, 0.0, 0.75)
FAINT_RED: RGB = (1.0, 0.75, 0.75)
LINK: RGB = (0.15, 0.15, 0.15)

STEP_OVERSIZE, STEP_COLLINEAR, STEP_PATTERN, STEP_SEQNO, STEP_SPATIAL, STEP_OUTLIERS, STEP_CROSSINGS = range(7)


# --------------------------------------------------------------------------
# Pipeline run + step access
# --------------------------------------------------------------------------

@dataclass
class Run:
    """One page through `classify_vectors(keep_steps=True)`."""

    pdf_path: str
    page_meta: PageMeta
    vectors: list[Vector]
    result: ClassificationResult

    @property
    def buckets(self) -> list[tuple[tuple, list]]:
        """`(bucket key, steps)` per (layer, color, width) bucket."""
        return [(key, stage.steps) for key, stage in self.result.clustering.items()]


def run(pdf_path: str, page_index: int) -> Run:
    with Reader(str(pdf_path)) as reader:
        page = reader.get_page(page_index)
        vectors = extract_vectors(page)
        meta = page.meta

    class _Page:  # classify_vectors only reads `page.meta`
        pass

    _Page.meta = meta
    result = classify_vectors(vectors, _Page(), keep_steps=True)
    return Run(str(pdf_path), meta, vectors, result)


def flatten(entry) -> list[Vector]:
    """A category entry -- a Vector group or a tiered cluster -- as one flat list."""
    if entry and isinstance(entry[0], list):
        return [v for g in entry for v in g]
    return list(entry)


def step_kept(steps, step: int) -> list[Vector]:
    """One bucket's `kept` Vectors after `step`, flat."""
    return [v for entry in steps[step].categories["kept"].groups for v in flatten(entry)]


def page_kept(r: Run, step: int) -> list[Vector]:
    return [v for _key, steps in r.buckets for v in step_kept(steps, step)]


def page_category(r: Run, step: int, name: str) -> list[list[Vector]]:
    """Every bucket's entries of category `name` at `step`, each flattened."""
    out = []
    for _key, steps in r.buckets:
        cat = steps[step].categories.get(name)
        if cat is not None:
            out.extend(flatten(e) for e in cat.groups)
    return out


def step_clusters(steps, step: int) -> list[list[Vector]]:
    """One bucket's kept clusters after a clustering step, each flat."""
    return [flatten(c) for c in steps[step].categories["kept"].groups]


def ids(vectors) -> set[int]:
    return {id(v) for v in vectors}


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def centre(v: Vector) -> tuple[float, float]:
    x0, y0, x1, y1 = v.bbox
    return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def crop_of(vectors: list[Vector], pad: float = 0.15, *, min_side: float = 20.0) -> Bbox:
    """Union bbox of `vectors`, padded by `pad` x its size on every side and
    grown to at least `min_side` pt per axis."""
    x0, y0, x1, y1 = union_bbox([v.bbox for v in vectors])
    w, h = max(x1 - x0, min_side), max(y1 - y0, min_side)
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    hw, hh = w * (0.5 + pad), h * (0.5 + pad)
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def page_box(meta: PageMeta) -> Bbox:
    return (0.0, 0.0, float(meta.width), float(meta.height))


def zoom_window(
    dropped: list[Vector], kept: list[Vector], meta: PageMeta, frac: float = 0.25,
) -> Bbox | None:
    """The square window (side `frac` x the page's short side) holding the
    most of both `dropped` and `kept` Vectors, counted by bbox centre:
    score = min(#dropped, #kept), ties to the larger sum. Scanned on a grid
    of side/4 steps via summed-area tables. `None` when no window holds
    both (score 0)."""
    w, h = float(meta.width), float(meta.height)
    side = frac * min(w, h)
    if side <= 0 or not dropped or not kept:
        return None
    step = side / 4.0
    nx, ny = max(1, math.ceil(w / step)), max(1, math.ceil(h / step))

    def grid(vs: list[Vector]) -> np.ndarray:
        g = np.zeros((ny, nx), dtype=np.int64)
        for v in vs:
            cx, cy = centre(v)
            ix = min(max(int(cx // step), 0), nx - 1)
            iy = min(max(int(cy // step), 0), ny - 1)
            g[iy, ix] += 1
        return g

    def window_sums(g: np.ndarray) -> np.ndarray:
        k = 4
        s = np.zeros((ny + 1, nx + 1), dtype=np.int64)
        s[1:, 1:] = g.cumsum(0).cumsum(1)
        ky, kx = min(k, ny), min(k, nx)
        return s[ky:, kx:] - s[:-ky, kx:] - s[ky:, :-kx] + s[:-ky, :-kx]

    d, k = window_sums(grid(dropped)), window_sums(grid(kept))
    score = np.minimum(d, k).astype(float) + (d + k) / (10.0 * (d.max() + k.max() + 1))
    iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
    if min(d[iy, ix], k[iy, ix]) == 0:
        return None
    x0, y0 = min(ix * step, max(w - side, 0.0)), min(iy * step, max(h - side, 0.0))
    return (x0, y0, x0 + side, y0 + side)


def _cubic_point(cubic, t: float) -> tuple[float, float]:
    p0, p1, p2, p3 = cubic
    u = 1.0 - t
    a, b, c, d = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
    return (a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
            a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1])


def line_cubic_points(seg, cubic, eps: float) -> list[tuple[float, float]]:
    """The proper crossings of segment `seg` and cubic `cubic` -- exactly
    the ones `line_geometry.line_cubic_crossings` counts (same root solve,
    sign test and in-segment test), returned as points."""
    x0, y0, x1, y1 = seg
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length == 0.0:
        return []
    p0, p1, p2, p3 = (np.asarray(p, dtype=float) for p in cubic)
    a = -p0 + 3 * p1 - 3 * p2 + p3
    b = 3 * p0 - 6 * p1 + 3 * p2
    c = -3 * p0 + 3 * p1

    def lin(v):
        return (dx * v[1] - dy * v[0]) / length

    offset = (dx * y0 - dy * x0) / length
    coeffs = [lin(a), lin(b), lin(c), lin(p0) - offset]
    if all(abs(k) < 1e-12 for k in coeffs):
        return []

    def f(t):
        px, py = _cubic_point(cubic, t)
        return (dx * (py - y0) - dy * (px - x0)) / length

    roots = sorted(float(r.real) for r in np.roots(coeffs) if abs(r.imag) < 1e-9 and 0.0 < r.real < 1.0)
    bounds = [0.0] + roots + [1.0]
    out = []
    for k, t in enumerate(roots, start=1):
        left, right = f((bounds[k - 1] + t) / 2.0), f((t + bounds[k + 1]) / 2.0)
        if left * right >= 0 or abs(left) <= eps or abs(right) <= eps:
            continue
        px, py = _cubic_point(cubic, t)
        s = ((px - x0) * dx + (py - y0) * dy) / length
        if eps < s < length - eps:
            out.append((px, py))
    return out


def _seg_intersection(a, b) -> tuple[float, float]:
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def crossing_points(
    cluster: list[Vector], *, only: list[Vector] | None = None, eps: float = CROSS_EPS_PT,
) -> list[tuple[float, float]]:
    """Where foreign pieces properly cross the "l"-only Vectors of
    `cluster` (`only`: just these of them) -- the crossings
    `line_crossing_counts` counts, as points. Segment x segment via
    `crossing_matrix`; segment x curve via `line_cubic_points`. A foreign
    segment crossing two own segments yields two points (the count is 1)."""
    pieces = [item_pieces(v) for v in cluster]
    targets = ids(only) if only is not None else None
    out: list[tuple[float, float]] = []
    for i, v in enumerate(cluster):
        if targets is not None and id(v) not in targets:
            continue
        if not v.items or any(it[0] != "l" for it in v.items):
            continue
        own = np.asarray(pieces[i][0], dtype=float).reshape(-1, 4)
        if not len(own):
            continue
        foreign_segs = [s for j, (segs, _c) in enumerate(pieces) if j != i for s in segs]
        if foreign_segs:
            foreign = np.asarray(foreign_segs, dtype=float)
            m = crossing_matrix(own, foreign, eps)
            for a, b in zip(*np.nonzero(m)):
                out.append(_seg_intersection(own[a], foreign[b]))
        for j, (_segs, cubics) in enumerate(pieces):
            if j == i:
                continue
            for cubic in cubics:
                for s in own:
                    out.extend(line_cubic_points(tuple(s), cubic, eps))
    return out


# --------------------------------------------------------------------------
# Colours
# --------------------------------------------------------------------------

def hues(k: int) -> list[RGB]:
    """`k` well-spread saturated colours (golden-ratio hue steps)."""
    return [colorsys.hsv_to_rgb((i * 0.618033988749895 + 0.05) % 1.0, 0.9, 0.85) for i in range(k)]


def to_hex(rgb: RGB) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(round(c * 255) for c in rgb))


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _curve(p0, p1, p2, p3, n: int = 12) -> list:
    return [_cubic_point((p0, p1, p2, p3), i / n) for i in range(n + 1)]


def ring(c: tuple[float, float], radius: float, n: int = 48) -> list:
    """A closed circle polyline."""
    return [(c[0] + radius * math.cos(2 * math.pi * k / n), c[1] + radius * math.sin(2 * math.pi * k / n))
            for k in range(n + 1)]


def vector_polylines(v: Vector) -> list[list]:
    """`v`'s items as drawable polylines (curves sampled, rects/quads closed)."""
    out = []
    for item in v.items:
        kind = item[0]
        if kind == "l":
            out.append([tuple(item[1]), tuple(item[2])])
        elif kind == "c":
            out.append(_curve(*item[1:5]))
        elif kind == "re":
            x0, y0, x1, y1 = item[1]
            out.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])
        elif kind == "qu":
            q = [tuple(p) for p in item[1]]
            out.append(q + q[:1])
    return out


def _touches(v: Vector, box: Bbox) -> bool:
    x0, y0, x1, y1 = v.bbox
    return x0 <= box[2] and box[0] <= x1 and y0 <= box[3] and box[1] <= y1


@dataclass
class Frame:
    """One image's page-space crop and its pixel scale. `px(n)` = n pixels
    in pt, so line widths/dots look the same at any zoom."""

    meta: PageMeta
    clip: Bbox
    side_px: int = 1600

    @property
    def zoom(self) -> float:
        x0, y0, x1, y1 = self.clip
        return self.side_px / max(x1 - x0, y1 - y0, 1e-6)

    def px(self, n: float) -> float:
        return n / self.zoom

    def visible(self, vectors) -> list[Vector]:
        return [v for v in vectors if _touches(v, self.clip)]


@dataclass
class Layer:
    """Recoloured Vectors (`color` an RGB or `fn(v) -> RGB`), plus optional
    extra polylines and dots, all in one colour family."""

    vectors: list[Vector] = field(default_factory=list)
    color: object = SUBJECT
    width_px: float = 2.0
    polylines: list = field(default_factory=list)
    poly_color: RGB = LINK
    poly_width_px: float = 1.0
    dots: list = field(default_factory=list)
    dot_color: RGB = RED
    dot_radius_px: float = 3.5


def _layer_specs(frame: Frame, layer: Layer) -> list:
    specs = []
    vs = frame.visible(layer.vectors)
    if callable(layer.color):
        by_color: dict[RGB, list] = {}
        for v in vs:
            by_color.setdefault(tuple(layer.color(v)), []).extend(vector_polylines(v))
        specs.extend(stroke_batch_spec(p, c, width=frame.px(layer.width_px)) for c, p in by_color.items())
    elif vs:
        specs.append(stroke_batch_spec([p for v in vs for p in vector_polylines(v)], layer.color,
                                       width=frame.px(layer.width_px)))
    if layer.polylines:
        specs.append(stroke_batch_spec(layer.polylines, layer.poly_color, width=frame.px(layer.poly_width_px)))
    if layer.dots:
        specs.append(dot_batch_spec(layer.dots, layer.dot_color, radius=frame.px(layer.dot_radius_px)))
    return specs


def _pixmap(page: "fitz.Page", frame: Frame) -> Image.Image:
    clip = fitz.Rect(*frame.clip) * page.rotation_matrix
    pix = page.get_pixmap(matrix=fitz.Matrix(frame.zoom, frame.zoom), clip=clip, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def render_specs(frame: Frame, specs: list) -> Image.Image:
    doc = fitz.open("pdf", render_specs_pdf(frame.meta, specs))
    try:
        return _pixmap(doc[0], frame)
    finally:
        doc.close()


def highlight(frame: Frame, context: list[Vector], layers: list[Layer], *, context_width_px: float = 1.0) -> Image.Image:
    """`context` in light grey, then each layer on top in order."""
    specs = _layer_specs(frame, Layer(context, CONTEXT, context_width_px))
    for layer in layers:
        specs.extend(_layer_specs(frame, layer))
    return render_specs(frame, specs)


def output(frame: Frame, vectors: list[Vector]) -> Image.Image:
    """`vectors` in their real paint -- what Phase 4 draws."""
    return render_specs(frame, [vector_spec(v) for v in frame.visible(vectors)])


def original(pdf_path: str, frame: Frame) -> Image.Image:
    """The source page itself (text, images and all), cropped."""
    doc = fitz.open(pdf_path)
    try:
        return _pixmap(doc[frame.meta.index], frame)
    finally:
        doc.close()


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def titled(img: Image.Image, title: str, *, size: int = 22) -> Image.Image:
    """`img` with a white title strip (wrapped to the image width) on top."""
    font = _font(size)
    draw = ImageDraw.Draw(img)
    lines, cur = [], ""
    for word in title.split(" "):
        test = f"{cur} {word}".strip()
        if cur and draw.textlength(test, font=font) > img.width - 16:
            lines.append(cur)
            cur = word
        else:
            cur = test
    lines.append(cur)
    strip = (size + 6) * len(lines) + 10
    out = Image.new("RGB", (img.width, img.height + strip), "white")
    out.paste(img, (0, strip))
    d = ImageDraw.Draw(out)
    for k, line in enumerate(lines):
        d.text((8, 5 + k * (size + 6)), line, fill=(0, 0, 0), font=font)
    return out


def montage(cells: list[Image.Image], cols: int = 3, *, gap: int = 6) -> Image.Image:
    """Cells (same size) in a `cols`-wide grid on a light background."""
    w, h = cells[0].size
    rows = math.ceil(len(cells) / cols)
    out = Image.new("RGB", (cols * w + (cols + 1) * gap, rows * h + (rows + 1) * gap), (225, 225, 225))
    for k, c in enumerate(cells):
        r, q = divmod(k, cols)
        out.paste(c, (gap + q * (w + gap), gap + r * (h + gap)))
    return out


# --------------------------------------------------------------------------
# Output folder
# --------------------------------------------------------------------------

class Showcase:
    """`outputs/filter_showcase/<ts>__<stem>_p<N>/`; `save` writes a titled
    PNG and remembers it per section for inline display."""

    def __init__(self, r: Run, *, side_px: int = 1600, seed: int = 0):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.r = r
        self.side_px = side_px
        self.rng = random.Random(seed)
        self.root = output_dir("filter_showcase", f"{stamp}__{Path(r.pdf_path).stem}_p{r.page_meta.index}")
        self.saved: dict[str, list[Path]] = {}

    def frame(self, clip: Bbox | None, side_px: int | None = None) -> Frame:
        return Frame(self.r.page_meta, clip or page_box(self.r.page_meta), side_px or self.side_px)

    def save(self, section: str, name: str, img: Image.Image, title: str) -> Path:
        folder = self.root / section
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.png"
        titled(img, title).save(path)
        self.saved.setdefault(section, []).append(path)
        return path

    def write_text(self, section: str, name: str, text: str) -> Path:
        folder = self.root / section
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        path.write_text(text, encoding="utf-8")
        return path

    def original(self, frame: Frame) -> Image.Image:
        return original(self.r.pdf_path, frame)


# --------------------------------------------------------------------------
# 1. Collinear drawing
# --------------------------------------------------------------------------

@dataclass
class CollinearData:
    input: list[Vector]                 # step-0 kept, all buckets
    groups: list[list[Vector]]          # every collinear group with >= 2 members
    angles: list[float]                 # each group's mean angle
    drawing: list[list[Vector]]         # step-1 "drawing" category
    kept: list[Vector]                  # step-1 kept


def collinear_data(r: Run) -> CollinearData:
    groups, angles, inp = [], [], []
    for _key, steps in r.buckets:
        vs = step_kept(steps, STEP_OVERSIZE)
        inp.extend(vs)
        straight, _ = split_straight(vs, STRAIGHT_TOL_PT)
        for g in group_collinear(straight, ANGLE_TOL_DEG, COLLINEAR_OFFSET_TOL_PT):
            if len(g) >= 2:
                groups.append([v for v, _ in g])
                angles.append(group_angle(g))
    return CollinearData(inp, groups, angles, page_category(r, STEP_COLLINEAR, "drawing"),
                         page_kept(r, STEP_COLLINEAR))


def representatives(data: CollinearData, global_angles: list[float], rng: random.Random
                    ) -> list[tuple[float, list[Vector] | None]]:
    """Per global angle, one random group (>= `GLOBAL_ANGLE_MIN_GROUP_SIZE`
    members) whose own angle is nearest that global angle (`None` if none)."""
    by_angle: dict[int, list[int]] = {k: [] for k in range(len(global_angles))}
    for gi, a in enumerate(data.angles):
        if len(data.groups[gi]) < GLOBAL_ANGLE_MIN_GROUP_SIZE or not global_angles:
            continue
        k = min(range(len(global_angles)), key=lambda k: axial_distance(a, global_angles[k]))
        by_angle[k].append(gi)
    return [(a, data.groups[rng.choice(by_angle[k])] if by_angle[k] else None)
            for k, a in enumerate(global_angles)]


def showcase_collinear(sc: Showcase, *, frac: float = 0.25, clip: Bbox | None = None) -> None:
    r, sec = sc.r, "1_collinear"
    data = collinear_data(r)
    dropped = [v for g in data.drawing for v in g]
    window = clip or zoom_window(dropped, data.kept, r.page_meta, frac)
    if window is None:
        print("collinear: no window holds both dropped and kept vectors -- using the full page")
    f = sc.frame(window)
    drawing_ids = ids(dropped)

    sc.save(sec, "01_original", sc.original(f), "Original")

    colours = hues(len(data.groups))
    colour_of = {id(v): colours[i] for i, g in enumerate(data.groups) for v in g}
    grouped = [v for g in data.groups for v in g]
    sc.save(sec, "02_collinear_groups",
            highlight(f, r.vectors, [Layer(grouped, lambda v: colour_of[id(v)])]),
            f"Collinear groups: {len(data.groups)} groups of >= 2 straight vectors on one infinite line "
            f"(one colour per group, grey = singletons / not straight)")

    reps = representatives(data, r.result.global_angles, sc.rng)
    rep_colours = hues(len(reps))
    full = sc.frame(None)
    layers, lines = [], ["angle_deg\tcolour\tgroup_size\tgroup_bbox"]
    for (angle, g), c in zip(reps, rep_colours):
        if g is None:
            lines.append(f"{angle:.2f}\t-\t0\t-")
            continue
        bb = union_bbox([v.bbox for v in g])
        # ring groups that are only a few px on the full page; larger ones show by themselves
        half_diag = 0.5 * math.hypot(bb[2] - bb[0], bb[3] - bb[1])
        rings = [ring(((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2), max(half_diag + full.px(6), full.px(14)))
                 ] if half_diag < full.px(20) else []
        layers.append(Layer(g, c, 4.0, polylines=rings, poly_color=c, poly_width_px=2.0))
        lines.append(f"{angle:.2f}\t{to_hex(c)}\t{len(g)}\t" + ", ".join(f"{x:.1f}" for x in bb))
    sc.write_text(sec, "03_global_angles.txt", "\n".join(lines) + "\n")
    sc.save(sec, "03_representative_per_direction",
            highlight(full, r.vectors, layers),
            f"One random collinear group per global direction, ringed ({len(reps)} directions: "
            + ", ".join(f"{a:.1f} deg" for a, _ in reps) + ")")

    sc.save(sec, "04_drawing_groups",
            highlight(f, r.vectors, [Layer(dropped, RED, 2.5)]),
            f"Detected as drawing: {len(data.drawing)} collinear groups with > 50 members and length std < 5 pt")
    non_drawing = [g for g in data.groups if id(g[0]) not in drawing_ids]
    nd_colour = {id(v): colours[i] for i, g in enumerate(non_drawing) for v in g}
    sc.save(sec, "05_non_drawing_groups",
            highlight(f, r.vectors, [Layer([v for g in non_drawing for v in g], lambda v: nd_colour[id(v)])]),
            f"Not detected as drawing: {len(non_drawing)} collinear groups (too few members or lengths too varied)")

    sc.save(sec, "06_after_filter", output(f, data.kept),
            f"After the collinear filter: {len(data.kept)} of {len(data.input)} vectors kept")


# --------------------------------------------------------------------------
# 2. Pattern lattice
# --------------------------------------------------------------------------

@dataclass
class LatticeGroup:
    vectors: list[Vector]
    links: list[tuple[Vector, Vector]]
    one_d: bool
    sim: int  # index into PatternData.similar


@dataclass
class PatternData:
    input: list[Vector]
    similar: list[list[Vector]]         # similarity buckets with > PATTERN_MIN_BUCKET members
    groups: list[LatticeGroup]          # every lattice group (singletons included)
    pattern: list[list[Vector]]
    bucket: list[list[Vector]]
    rejected: list[list[Vector]]
    kept: list[Vector]


def pattern_data(r: Run) -> PatternData:
    """Recompute, per bucket, exactly what `pattern_drawing` computes
    (similarity buckets -> lattice groups) on that step's input."""
    inp, similar, groups = [], [], []
    for _key, steps in r.buckets:
        vs = step_kept(steps, STEP_COLLINEAR)
        inp.extend(vs)
        anchors = [pl.anchor(v) for v in vs]
        for sim in pl._similarity_index_groups(vs, anchors, PATTERN_MIN_BUCKET + 1):
            sim_vs = [vs[i] for i in sim]
            si = len(similar)
            similar.append(sim_vs)
            xy = np.asarray([anchors[i] for i in sim], dtype=float)
            for members, links, one_d in pl._lattice_groups(xy, [v.seqno for v in sim_vs]):
                groups.append(LatticeGroup(
                    [sim_vs[j] for j in members],
                    [(sim_vs[a], sim_vs[b]) for a, b in links.tolist()], one_d, si,
                ))
    return PatternData(
        inp, similar, groups,
        page_category(r, STEP_PATTERN, PATTERN_CATEGORY),
        page_category(r, STEP_PATTERN, PATTERN_BUCKET_CATEGORY),
        page_category(r, STEP_PATTERN, PATTERN_REJECTED_CATEGORY),
        page_kept(r, STEP_PATTERN),
    )


def similar_grid(r: Run, members: list[Vector], rng: random.Random, *, n: int = 9, cell_px: int = 320
                 ) -> Image.Image:
    """`n` random members, each alone and centred in its cell, every cell
    the same pt -> px scale (set by the largest member)."""
    pick = rng.sample(members, min(n, len(members)))
    side = max(max(v.bbox[2] - v.bbox[0], v.bbox[3] - v.bbox[1]) for v in pick) * 1.3 + 2.0
    cells = []
    for v in pick:
        cx, cy = centre(v)
        f = Frame(r.page_meta, (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2), cell_px)
        cells.append(output(f, [v]).resize((cell_px, cell_px)))
    return montage(cells)


def showcase_pattern(sc: Showcase, *, top_k: int = 5, frac: float = 0.25, clip: Bbox | None = None) -> None:
    r, sec = sc.r, "2_pattern"
    data = pattern_data(r)
    dropped_pattern = [v for g in data.pattern for v in g]
    dropped_bucket = [v for g in data.bucket for v in g]
    window = clip or zoom_window(dropped_pattern + dropped_bucket, data.kept, r.page_meta, frac)
    if window is None:
        print("pattern: no window holds both dropped and kept vectors -- using the full page")
    f = sc.frame(window)

    sc.save(sec, "01_original", sc.original(f), "Original")

    for k, members in enumerate(sorted(data.similar, key=len, reverse=True)[:top_k], start=1):
        sc.save(sec, f"02_similar_vectors_{k}", similar_grid(r, members, sc.rng),
                f"Similar vectors, bucket #{k} ({len(members)} members): same item kinds + per-item "
                f"lengths within 3 pt, any rotation -- 9 random members, shared scale")

    for k, g in enumerate(sorted(data.groups, key=lambda g: len(g.vectors), reverse=True)[:top_k], start=1):
        gf = sc.frame(crop_of(g.vectors, 0.15))
        bucket = data.similar[g.sim]
        links = [[centre(a), centre(b)] for a, b in g.links]
        img = highlight(gf, r.vectors, [
            Layer(bucket, SUBJECT, 1.5),
            Layer(g.vectors, BLUE, 2.0, polylines=links, poly_color=RED, poly_width_px=1.5,
                  dots=[centre(v) for v in g.vectors], dot_radius_px=3.0),
        ])
        kind = "1D row" if g.one_d else "2D lattice"
        sc.save(sec, f"03_lattice_group_{k}", img,
                f"Similarity lattice #{k}: {len(g.vectors)} members on one {kind} (blue; red = member centres "
                f"and links between lattice neighbours; dark grey = rest of the similarity bucket)")

    sc.save(sec, "04_large_groups_dropped",
            highlight(f, r.vectors, [Layer(dropped_pattern, RED, 2.5)]),
            f"Detected large lattices, dropped as drawing: {len(data.pattern)} groups "
            f"(> {PATTERN_MAX_GROUP} members, 1D > {PATTERN_MAX_GROUP_1D}, little foreign content between members)")

    dropped_ids = ids(dropped_pattern) | ids(dropped_bucket)
    rejected_ids = ids(v for g in data.rejected for v in g)
    failed, short = [], []
    for g in data.groups:
        if len(g.vectors) <= PATTERN_MAX_GROUP or id(g.vectors[0]) in dropped_ids:
            continue
        (failed if id(g.vectors[0]) in rejected_ids else short).append(g)
    sc.save(sec, "05_large_groups_not_dropped",
            highlight(f, r.vectors, [
                Layer([v for g in short for v in g.vectors], AMBER, 2.5),
                Layer([v for g in failed for v in g.vectors], ORANGE, 2.5),
            ]),
            f"Detected large lattices, kept: orange = too much other content between members "
            f"({len(failed)} groups); amber = 1D row of <= {PATTERN_MAX_GROUP_1D} ({len(short)} groups)")

    pattern_ids = ids(dropped_pattern)
    remaining = [v for v in data.input if id(v) not in pattern_ids]
    sc.save(sec, "06_remaining", output(f, remaining),
            f"Remaining after dropping the lattices: {len(remaining)} of {len(data.input)} vectors")

    sc.save(sec, "07_pattern_bucket",
            highlight(f, r.vectors, [Layer(dropped_pattern, FAINT_RED, 2.0), Layer(dropped_bucket, PURPLE, 2.5)]),
            f"Bucket rule: {len(data.bucket)} more large groups dropped (purple) because > 2 groups of the "
            f"same similarity bucket were already dropped (faint red)")

    sc.save(sec, "08_after_filter", output(f, data.kept),
            f"After the pattern filter: {len(data.kept)} of {len(data.input)} vectors kept")


# --------------------------------------------------------------------------
# 3. Length outliers
# --------------------------------------------------------------------------

def outlier_clusters(r: Run) -> list[tuple[list[Vector], list[Vector]]]:
    """`(cluster, outliers)` for every step-4 cluster with outliers."""
    out = []
    for _key, steps in r.buckets:
        for cluster in step_clusters(steps, STEP_SPATIAL):
            gone = length_outliers(cluster)
            if gone:
                out.append((cluster, gone))
    return out


def showcase_length_outliers(sc: Showcase, *, n: int = 3) -> None:
    r = sc.r
    cands = outlier_clusters(r)
    picks = sc.rng.sample(cands, min(n, len(cands)))
    if len(picks) < n:
        print(f"length outliers: only {len(picks)} cluster(s) had outliers")
    for k, (cluster, gone) in enumerate(picks, start=1):
        sec = f"3_length_outliers/cluster_{k}"
        f = sc.frame(crop_of(cluster, 0.15))
        sc.save(sec, "01_original", sc.original(f), f"Original (cluster {k}, {len(cluster)} vectors)")

        straight, _ = split_straight(cluster, STRAIGHT_TOL_PT)
        par = [g for g in group_parallel(straight, ANGLE_TOL_DEG) if len(g) >= MIN_PARALLEL_GROUP_SIZE]
        colours = hues(len(par))
        colour_of = {id(v): colours[i] for i, g in enumerate(par) for v, _ in g}
        lengths = np.asarray([fit.length for g in par for _, fit in g])
        mean, std = float(lengths.mean()), float(lengths.std())
        lo, hi = mean - LENGTH_OUTLIER_STD * std, mean + LENGTH_OUTLIER_STD * std
        sc.save(sec, "02_parallel_groups",
                highlight(f, r.vectors, [Layer(cluster, SUBJECT, 1.5),
                                         Layer([v for g in par for v, _ in g], lambda v: colour_of[id(v)], 2.5)]),
                f"{len(par)} parallel groups (one colour each); pooled length {mean:.1f} +- {std:.1f} pt, "
                f"kept band {max(lo, 0):.1f}-{hi:.1f} pt")
        gone_ids = ids(gone)
        sc.save(sec, "03_outliers",
                highlight(f, r.vectors, [Layer(cluster, SUBJECT, 1.5), Layer(gone, RED, 3.0)]),
                f"Length outliers: {len(gone)} vectors more than {LENGTH_OUTLIER_STD:g} std from the mean (red)")
        sc.save(sec, "04_after_filter", output(f, [v for v in cluster if id(v) not in gone_ids]),
                f"After removing the outliers: {len(cluster) - len(gone)} of {len(cluster)} vectors")


# --------------------------------------------------------------------------
# 4. Crossings
# --------------------------------------------------------------------------

def crossing_clusters(r: Run) -> list[tuple[list[Vector], list[Vector], list[Vector]]]:
    """`(cluster, dropped, flagged_kept)` for every step-5 cluster where the
    crossings step drops something *and* keeps some flagged vectors."""
    out = []
    for _key, steps in r.buckets:
        for cluster in step_clusters(steps, STEP_OUTLIERS):
            dropped, kept_flagged = crossed_grid(cluster)
            if dropped and kept_flagged:
                out.append((cluster, dropped, kept_flagged))
    return out


def showcase_crossings(sc: Showcase, *, n: int = 3) -> None:
    r = sc.r
    cands = crossing_clusters(r)
    picks = sc.rng.sample(cands, min(n, len(cands)))
    if len(picks) < n:
        print(f"crossings: only {len(picks)} cluster(s) had both dominant-grid drops and off-grid flagged lines")
    for k, (cluster, dropped, kept_flagged) in enumerate(picks, start=1):
        sec = f"4_crossings/cluster_{k}"
        f = sc.frame(crop_of(cluster, 0.15))
        sc.save(sec, "01_original", sc.original(f), f"Original (cluster {k}, {len(cluster)} vectors)")

        flagged = dropped + kept_flagged
        # two flagged lines crossing each other report the same point twice
        dots = list({(round(x, 2), round(y, 2)): (x, y) for x, y in crossing_points(cluster, only=flagged)}.values())
        sc.save(sec, "02_crossings",
                highlight(f, r.vectors, [Layer(cluster, SUBJECT, 1.5),
                                         Layer(flagged, ORANGE, 2.5, dots=dots, dot_color=RED)]),
                f"{len(flagged)} lines crossed >= {MIN_CROSSINGS} times (orange), "
                f"{len(dots)} crossings on them (red dots)")
        sc.save(sec, "03_dominant_vs_off_grid",
                highlight(f, r.vectors, [Layer(cluster, SUBJECT, 1.5), Layer(kept_flagged, BLUE, 2.5),
                                         Layer(dropped, RED, 2.5)]),
                f"Dominant grid (red, {len(dropped)}) vs off-grid flagged lines (blue, {len(kept_flagged)})")
        sc.save(sec, "04_dropped",
                highlight(f, r.vectors, [Layer(cluster, SUBJECT, 1.5), Layer(dropped, RED, 3.0)]),
                f"Dropped to drawing: {len(dropped)} dominant-grid lines")
        gone = ids(dropped)
        sc.save(sec, "05_after_filter", output(f, [v for v in cluster if id(v) not in gone]),
                f"After the crossings filter: {len(cluster) - len(dropped)} of {len(cluster)} vectors")
