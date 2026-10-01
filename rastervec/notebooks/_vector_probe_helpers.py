"""Shared helpers for the three vector-geometry probe notebooks
(`vector_intersection_lab.ipynb`, `dashed_line_collinear_lab.ipynb`,
`parallel_groups_lab.ipynb`) -- experimental VectorClassification signals,
kept out of `P3_Vector_Parsing/` until one of them proves useful.

The unit everywhere is one whole `Vector` (one `get_drawings()` drawing),
never an individual item:

- **crossings** -- `crossing_partner_counts(cluster)`: per Vector, how many
  *other* Vectors of the same cluster have at least one *proper* (interior,
  X-shaped) crossing against it. Items are flattened to straight segments
  for the test (`vector_segments`: "l" as-is, "re"/"qu" as their 4 edges,
  "c" as a `curve_steps`-segment polyline); touching endpoints, corners and
  T-junctions don't count, and a Vector crossing its own items doesn't
  count.
- **straight Vectors** -- `straight_line(v)`: a Vector whose items are all
  "l" and whose points all lie within `tol` of one line. Only these take
  part in collinear/parallel grouping; everything else is "excluded".
- **grouping** -- `group_parallel` (single-linkage on the folded [0, 180)
  angle, wrap-aware -- see `_chain_angles`) and `group_collinear`
  (`group_parallel`, then *anchored* grouping on each line's perpendicular
  offset `rho` within each angle group -- every member within `tol` of its
  group's smallest offset, so a group never spans more than `tol`; see
  `_anchored_1d` for why offsets can't be single-linkage). "Same infinite
  line", no gap limit.

`DebugReport` writes a folder `scripts/pipeline_report_viewer.py` opens
directly: one single-page PDF per layer plus a `manifest.json` in the exact
shape the viewer's `_Panel` reads.
"""
from __future__ import annotations

import colorsys
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from rastervec.commons.models import PageMeta, Vector
from rastervec.commons.paths import output_dir
from rastervec.commons.renderer.pdf import render_boxes_pdf, render_vectors_pdf
from rastervec.P1_Reading_Native.reader import Reader
from rastervec.P1_Reading_Native.vector_extract import extract_vectors
from rastervec.P3_Vector_Parsing.VectorClassification.classify_vectors import classify_vectors
from rastervec.P3_Vector_Parsing.VectorClassification.layer_color_separation import (
    separate_by_color,
    separate_by_layer,
    separate_by_width,
)

Point = tuple[float, float]
Segment = tuple[float, float, float, float]  # x0, y0, x1, y1
BucketKey = tuple  # (layer, color-key, width-key)


# --------------------------------------------------------------------------
# Pipeline front half: extract -> bucket -> (cluster)
# --------------------------------------------------------------------------

def load_page_vectors(pdf_path: str, page_index: int) -> tuple[PageMeta, list[Vector]]:
    """Phase-1 raw vectors for one page. `Vector`s are plain data, so the
    document is closed again before returning."""
    with Reader(str(pdf_path)) as reader:
        page = reader.get_page(page_index)
        vectors = extract_vectors(page)
        return page.meta, vectors


def bucket_vectors(vectors: list[Vector]) -> dict[BucketKey, list[Vector]]:
    """layer -> color -> width buckets, flattened to one dict -- the same
    three P3 separation functions `classify_vectors` uses."""
    out: dict[BucketKey, list[Vector]] = {}
    for layer, by_layer in separate_by_layer(vectors).items():
        for color, by_color in separate_by_color(by_layer).items():
            for width, by_width in separate_by_width(by_color).items():
                out[(layer, color, width)] = by_width
    return out


def cluster_vectors(vectors: list[Vector], page_meta: PageMeta) -> list[tuple[BucketKey, list[Vector]]]:
    """Run P3 VectorClassification's own classification chain (layer/color/
    width buckets -> seqno-overlap merge -> constrained spatial clustering)
    and return every final cluster as `(bucket_key, flat Vector list)`."""

    class _Page:  # classify_vectors only threads `page` through, never reads it
        meta = page_meta

    result = classify_vectors(vectors, _Page())
    clusters: list[tuple[BucketKey, list[Vector]]] = []
    for key, stage in result.clustering.items():
        if not stage.steps:
            continue
        for cluster in stage.steps[-1].categories["kept"].groups:
            clusters.append((key, [v for group in cluster for v in group]))
    return clusters


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def _cubic(p0: Point, p1: Point, p2: Point, p3: Point, steps: int) -> list[Point]:
    pts = []
    for i in range(steps + 1):
        t = i / steps
        u = 1.0 - t
        a, b, c, d = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
        pts.append((
            a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
            a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
        ))
    return pts


def _ring(points: list[Point]) -> list[Segment]:
    return [
        (*points[i], *points[(i + 1) % len(points)])
        for i in range(len(points))
    ]


def vector_segments(v: Vector, curve_steps: int = 8) -> list[Segment]:
    """Every item of `v` as straight segments: "l" as-is, "re" (x0, y0, x1,
    y1) and "qu" (ul, ur, lr, ll -- `plain_item`'s order) as their 4 edges,
    "c" flattened to `curve_steps` segments. Zero-length segments dropped."""
    segs: list[Segment] = []
    for item in v.items:
        kind = item[0]
        if kind == "l":
            segs.append((*item[1], *item[2]))
        elif kind == "re":
            x0, y0, x1, y1 = item[1]
            segs.extend(_ring([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
        elif kind == "qu":
            segs.extend(_ring(list(item[1])))
        elif kind == "c":
            pts = _cubic(item[1], item[2], item[3], item[4], curve_steps)
            segs.extend((*pts[i], *pts[i + 1]) for i in range(len(pts) - 1))
    return [s for s in segs if s[0] != s[2] or s[1] != s[3]]


def _signed_dist(seg: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Signed perpendicular distance of points (px, py) from each segment's
    supporting line; `seg` (n, 4), px/py broadcastable against (n, ...)."""
    dx = seg[..., 2] - seg[..., 0]
    dy = seg[..., 3] - seg[..., 1]
    length = np.hypot(dx, dy)
    length = np.where(length == 0, 1.0, length)
    return (dx * (py - seg[..., 1]) - dy * (px - seg[..., 0])) / length


def segments_properly_cross(a: np.ndarray, b: np.ndarray, eps: float = 0.01) -> bool:
    """True if any segment of `a` (n, 4) properly crosses any segment of
    `b` (m, 4): each segment's endpoints lie strictly on opposite sides of
    the other's line, each by more than `eps` pt. So an endpoint touching
    the other segment (T-junction, shared corner, a dash end resting on a
    line) is never a crossing, and neither is collinear overlap."""
    if len(a) == 0 or len(b) == 0:
        return False
    A = a[:, None, :]  # (n, 1, 4)
    B = b[None, :, :]  # (1, m, 4)
    # Cheap per-pair bbox reject first.
    overlap = (
        (np.minimum(A[..., 0], A[..., 2]) <= np.maximum(B[..., 0], B[..., 2]))
        & (np.minimum(B[..., 0], B[..., 2]) <= np.maximum(A[..., 0], A[..., 2]))
        & (np.minimum(A[..., 1], A[..., 3]) <= np.maximum(B[..., 1], B[..., 3]))
        & (np.minimum(B[..., 1], B[..., 3]) <= np.maximum(A[..., 1], A[..., 3]))
    )
    if not overlap.any():
        return False
    ia, ib = np.nonzero(overlap)
    sa, sb = a[ia], b[ib]
    d1 = _signed_dist(sa, sb[:, 0], sb[:, 1])
    d2 = _signed_dist(sa, sb[:, 2], sb[:, 3])
    d3 = _signed_dist(sb, sa[:, 0], sa[:, 1])
    d4 = _signed_dist(sb, sa[:, 2], sa[:, 3])
    cross = (
        (d1 * d2 < 0) & (np.abs(d1) > eps) & (np.abs(d2) > eps)
        & (d3 * d4 < 0) & (np.abs(d3) > eps) & (np.abs(d4) > eps)
    )
    return bool(cross.any())


def _overlapping_pairs(bboxes: list[tuple[float, float, float, float]]):
    """Sweep-and-prune over x: every (i, j), i < j, whose bboxes overlap."""
    order = sorted(range(len(bboxes)), key=lambda i: bboxes[i][0])
    active: list[int] = []
    for i in order:
        x0, y0, x1, y1 = bboxes[i]
        active = [j for j in active if bboxes[j][2] >= x0]
        for j in active:
            if bboxes[j][1] <= y1 and y0 <= bboxes[j][3]:
                yield (min(i, j), max(i, j))
        active.append(i)


def crossing_partner_counts(
    cluster: list[Vector], *, eps: float = 0.01, curve_steps: int = 8,
) -> list[int]:
    """Per Vector of `cluster` (same order), the number of *other* Vectors
    in `cluster` it properly crosses at least once (see
    `segments_properly_cross`)."""
    segs = [np.asarray(vector_segments(v, curve_steps), dtype=float).reshape(-1, 4) for v in cluster]
    bboxes = []
    for s in segs:
        if len(s):
            xs, ys = s[:, [0, 2]], s[:, [1, 3]]
            bboxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
        else:
            bboxes.append((math.inf, math.inf, -math.inf, -math.inf))
    counts = [0] * len(cluster)
    for i, j in _overlapping_pairs(bboxes):
        if segments_properly_cross(segs[i], segs[j], eps):
            counts[i] += 1
            counts[j] += 1
    return counts


@dataclass(frozen=True)
class LineFit:
    """A straight Vector's line: `angle` folded to [0, 180) degrees (page
    space, y down), `midpoint` of its extent, `length` = extent along the
    line (t_max - t_min)."""

    angle: float
    midpoint: Point
    length: float


def straight_line(v: Vector, tol: float = 0.25) -> LineFit | None:
    """`LineFit` if every item of `v` is "l" and every point lies within
    `tol` pt of one line (through the two mutually farthest points), else
    `None`. A zero-extent Vector is not straight."""
    if not v.items or any(item[0] != "l" for item in v.items):
        return None
    pts = np.asarray([p for item in v.items for p in item[1:3]], dtype=float)
    first = pts[0]
    p0 = pts[np.argmax(np.hypot(*(pts - first).T))]
    p1 = pts[np.argmax(np.hypot(*(pts - p0).T))]
    span = float(np.hypot(*(p1 - p0)))
    if span == 0.0:
        return None
    angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])) % 180.0
    ux, uy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    rel = pts - p0
    if np.max(np.abs(rel[:, 0] * uy - rel[:, 1] * ux)) > tol:
        return None
    t = rel[:, 0] * ux + rel[:, 1] * uy
    t_min, t_max = float(t.min()), float(t.max())
    t_mid = (t_min + t_max) / 2.0
    return LineFit(
        angle=angle,
        midpoint=(float(p0[0] + t_mid * ux), float(p0[1] + t_mid * uy)),
        length=t_max - t_min,
    )


def split_straight(vectors: list[Vector], tol: float = 0.25) -> tuple[list[tuple[Vector, LineFit]], list[Vector]]:
    """`(straight, excluded)` -- `(Vector, LineFit)` pairs for straight
    Vectors, plain Vectors for everything else."""
    straight, excluded = [], []
    for v in vectors:
        fit = straight_line(v, tol)
        if fit is None:
            excluded.append(v)
        else:
            straight.append((v, fit))
    return straight, excluded


def _anchored_1d(values: list[float], tol: float) -> list[list[int]]:
    """Anchored grouping on a line: indices sorted by value; a group starts
    at its smallest value and takes every following value within `tol` of
    that anchor, so a group's total span is at most `tol`. Deliberately
    *not* single-linkage: on a real page (240118 p0) dense hatching put
    2227 diagonal segments' perpendicular offsets 0.06 pt apart (median),
    and single-linkage chained them into one 218 pt-wide "collinear" group."""
    order = sorted(range(len(values)), key=values.__getitem__)
    groups: list[list[int]] = []
    for i in order:
        if groups and values[i] - values[groups[-1][0]] <= tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _chain_angles(angles: list[float], tol: float) -> list[list[int]]:
    """Single-linkage on the [0, 180) circle: angles sorted, split wherever
    the gap to the next exceeds `tol`; the first and last groups merge when
    the wrap-around gap (last -> 180 -> first) is also within `tol`.
    Single-linkage (not `_anchored_1d`) on purpose: short CAD segments carry
    real angle jitter -- 240118 p0's 6 pt hatch segments spread over
    134-136 deg -- and a hard 1 deg span cap cut one hatch into 3 groups.
    The cost: a polygonised arc drawn as separate straight Vectors < `tol`
    apart chains into one group."""
    order = sorted(range(len(angles)), key=angles.__getitem__)
    groups: list[list[int]] = []
    for i in order:
        if groups and angles[i] - angles[groups[-1][-1]] <= tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    if len(groups) > 1 and angles[groups[0][0]] + 180.0 - angles[groups[-1][-1]] <= tol:
        groups[0] = groups.pop() + groups[0]
    return groups


def _mean_axis_angle(angles: list[float]) -> float:
    """Circular mean of axial angles (period 180), via doubled angles."""
    s = sum(math.sin(math.radians(2 * a)) for a in angles)
    c = sum(math.cos(math.radians(2 * a)) for a in angles)
    return (math.degrees(math.atan2(s, c)) / 2.0) % 180.0


def group_parallel(
    straight: list[tuple[Vector, LineFit]], angle_tol: float = 1.0,
) -> list[list[tuple[Vector, LineFit]]]:
    """Single-linkage groups of straight Vectors by folded angle
    (`_chain_angles` -- wrap-aware, so 179.8 deg and 0.1 deg join)."""
    angles = [fit.angle for _, fit in straight]
    return [[straight[i] for i in g] for g in _chain_angles(angles, angle_tol)]


def group_collinear(
    straight: list[tuple[Vector, LineFit]], angle_tol: float = 1.0, offset_tol: float = 0.5,
) -> list[list[tuple[Vector, LineFit]]]:
    """Same-infinite-line groups: `group_parallel`, then within each angle
    group anchored grouping on each line's perpendicular offset `rho` (its
    midpoint projected onto the group's mean-angle normal) within
    `offset_tol` pt. No along-line gap limit."""
    out = []
    for group in group_parallel(straight, angle_tol):
        theta = math.radians(_mean_axis_angle([fit.angle for _, fit in group]))
        nx, ny = -math.sin(theta), math.cos(theta)
        rhos = [fit.midpoint[0] * nx + fit.midpoint[1] * ny for _, fit in group]
        out.extend([group[i] for i in g] for g in _anchored_1d(rhos, offset_tol))
    return out


def group_stats(group: list[tuple[Vector, LineFit]]) -> tuple[int, float]:
    """`(member count, population std of member lengths in pt)`."""
    lengths = [fit.length for _, fit in group]
    return len(lengths), float(np.std(lengths)) if lengths else 0.0


# --------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------

RGB = tuple[float, float, float]
GREY: RGB = (0.55, 0.55, 0.55)
LIGHT_GREY: RGB = (0.8, 0.8, 0.8)

_CROSSING_ANCHORS: list[tuple[int, RGB]] = [
    (0, (1.0, 0.0, 0.0)),    # red
    (3, (1.0, 0.85, 0.0)),   # yellow
    (5, (0.0, 0.7, 0.0)),    # green
    (7, (0.0, 0.25, 1.0)),   # blue (7+)
]


def ramp_crossings(n: int) -> RGB:
    """0 red -> 3 yellow -> 5 green -> 7+ blue, linear RGB in between."""
    if n >= _CROSSING_ANCHORS[-1][0]:
        return _CROSSING_ANCHORS[-1][1]
    for (n0, c0), (n1, c1) in zip(_CROSSING_ANCHORS, _CROSSING_ANCHORS[1:]):
        if n0 <= n <= n1:
            t = (n - n0) / (n1 - n0)
            return tuple(a + (b - a) * t for a, b in zip(c0, c1))
    return _CROSSING_ANCHORS[0][1]


def green_red(t: float) -> RGB:
    """t=0 green -> t=1 red, through yellow (HSV hue 120 -> 0), so the
    midpoint is a clean yellow rather than linear-RGB olive."""
    t = min(1.0, max(0.0, t))
    return colorsys.hsv_to_rgb((1.0 - t) * 120.0 / 360.0, 1.0, 0.9)


def normalise(x: float, lo: float, hi: float) -> float:
    """`(x - lo) / (hi - lo)`, 0 when the range is empty."""
    return 0.0 if hi <= lo else (x - lo) / (hi - lo)


def golden_hues(k: int) -> list[float]:
    """`k` well-spread hues in [0, 1) (golden-ratio steps)."""
    return [(i * 0.618033988749895) % 1.0 for i in range(k)]


def angle_value(angle: float) -> float:
    """HSV value from a folded angle: triangle wave, 0/180 deg -> 0.25,
    90 deg -> 0.75, linear in between."""
    a = angle % 180.0
    return 0.25 + 0.5 * (1.0 - abs(a - 90.0) / 90.0)


def hsv_cluster_angle(hue: float, angle: float) -> RGB:
    return colorsys.hsv_to_rgb(hue, 1.0, angle_value(angle))


def to_hex(rgb: RGB) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(round(c * 255) for c in rgb))


# --------------------------------------------------------------------------
# Output: pipeline_report_viewer-compatible folder
# --------------------------------------------------------------------------

def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9+.-]+", "_", text).strip("_") or "layer"


class DebugReport:
    """One `outputs/<name>/<ts>__<pdf-stem>_p<N>/` folder: `add_vectors`/
    `add_boxes`/`add_layer` write one single-page PDF per layer
    (`<stage>__<label>.pdf`), `save_figure` a PNG, `finish` the
    `manifest.json` `scripts/pipeline_report_viewer.py` reads. Open with
    `python scripts/pipeline_report_viewer.py <out_dir>`."""

    def __init__(self, name: str, source_pdf: str, page_meta: PageMeta, *, line_width: float = 1.0):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.name = name
        self.source_pdf = str(Path(source_pdf).resolve())
        self.page_meta = page_meta
        self.line_width = line_width
        self.out_dir = output_dir(name, f"{stamp}__{Path(source_pdf).stem}_p{page_meta.index}")
        self.layers: list[dict] = []

    def add_layer(self, stage: str, label: str, color: RGB | str, pdf_bytes: bytes) -> None:
        file = f"{_slug(stage)}__{_slug(label)}.pdf"
        (self.out_dir / file).write_bytes(pdf_bytes)
        self.layers.append({
            "stage": stage, "layer": label, "file": file,
            "color": color if isinstance(color, str) else to_hex(color),
        })

    def add_vectors(self, stage: str, label: str, color: RGB | str, vectors: list[Vector], color_of) -> None:
        """One layer of `vectors` restroked in `color_of(v)`. Empty layers
        are skipped (nothing to toggle)."""
        if not vectors:
            return
        self.add_layer(stage, label, color, render_vectors_pdf(
            self.page_meta, vectors, color_of=color_of, width=self.line_width,
        ))

    def add_boxes(self, stage: str, label: str, color: RGB | str, boxes: list) -> None:
        if boxes:
            self.add_layer(stage, label, color, render_boxes_pdf(self.page_meta, boxes, width=0.75))

    def save_figure(self, fig, name: str) -> Path:
        path = self.out_dir / f"{_slug(name)}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        return path

    def finish(self) -> Path:
        manifest = {
            "source_pdf": self.source_pdf,
            "pages": [self.page_meta.index],
            "engine": "notebook",
            "variant": self.name,
            "layers": self.layers,
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return self.out_dir

    def viewer_command(self) -> str:
        return f'.venv/Scripts/python.exe scripts/pipeline_report_viewer.py "{self.out_dir}"'
