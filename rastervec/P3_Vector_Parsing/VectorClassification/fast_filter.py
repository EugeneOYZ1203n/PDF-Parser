"""Post-recognition FAST text/drawing split for VectorClassification.

After PaddleOCR detect, each cluster's detect quads are grouped
(`group_quads_anchored`), each group's crop of the cluster render goes
through FAST (`fast_detect.py`) for a text heatmap, and -- once recognition
has decided which groups hold real (non-blank) text -- every vector under an
accepted group is scored by how much of its *own* ink FAST highlighted
(`vector_ink_fraction`). Only vectors that pass stay text; everything else
in the OCR'd clusters goes to `drawing` (see `parse.py`). This replaces the
transitive bbox-connectivity rule (`parse._drawing_extra_vectors`), which
absorbed any leader/dimension line touching a label into text.

Pixel spaces: "cluster px" is the pixel grid of the cluster's own OCR render
(`render_cluster_with_dynamic_dpi(group_vectors, ...)` at `dpi`/`padding`);
a group's `crop_rect` and `heat` live in that grid (`heat[y - cy0, x - cx0]`).
A vector's ink mask is its own single-vector render at the same dpi, offset
into cluster px via `page_points_to_pixel`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from rastervec.commons.helpers.geometry import bboxes_intersect, union_bbox
from rastervec.commons.models import Vector
from rastervec.commons.renderer import (
    page_points_to_pixel, pixel_to_page_bbox, render_vector_cluster,
)

Bbox = tuple[float, float, float, float]
PixelRect = tuple[int, int, int, int]
# (ink mask, page-space point that the mask's pixel (0, 0) corresponds to)
InkMask = tuple[np.ndarray, tuple[float, float]]


@dataclass
class FastGroup:
    """One FAST crop: a group of one cluster's detect quads.

    `quad_idxs` index into `parse.py`'s page-wide `page_quads`. `crop_rect`
    is the padded union of the quads' pixel envelopes in cluster px, and
    `crop_page_bbox` the same rect in page space; `fast_input` is that crop white-padded to `FAST_CROP_MAX_ASPECT`, with
    the real crop at `input_offset` inside it. `heat` (crop-sized, set after
    FAST runs) is FAST's score map cut back to `crop_rect`."""

    cluster_vectors: list[Vector]
    dpi: int
    padding: float
    quad_idxs: list[int]
    crop_rect: PixelRect
    page_bbox: Bbox
    crop_page_bbox: Bbox
    fast_input: "np.ndarray | None"
    input_offset: tuple[int, int]
    heat: "np.ndarray | None" = None
    accepted: bool = False
    # Filled by `split_text_vectors`, per scored vector: (vector, fraction).
    scores: list[tuple[Vector, float]] = field(default_factory=list)


def _center(b: Bbox) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def group_quads_anchored(page_bboxes: list[Bbox], center_dist_pt: float) -> list[list[int]]:
    """Anchored (no-chaining) grouping of one cluster's detect quads: quads
    are taken in order; each joins the first existing group whose *seed*
    (first member) bbox it overlaps/touches AND whose seed center is closer
    than `center_dist_pt`, else it seeds a new group. Membership is only
    ever tested against the seed, so A~B and B~C never pulls C into A's
    group unless C is also close to A."""
    groups: list[list[int]] = []
    for i, bbox in enumerate(page_bboxes):
        cx, cy = _center(bbox)
        for members in groups:
            seed = page_bboxes[members[0]]
            sx, sy = _center(seed)
            if bboxes_intersect(bbox, seed) and math.hypot(cx - sx, cy - sy) < center_dist_pt:
                members.append(i)
                break
        else:
            groups.append([i])
    return groups


def quad_pixel_rect(quad: np.ndarray) -> tuple[float, float, float, float]:
    """Axis-aligned envelope of one detector quad (Nx2, cluster px)."""
    pts = np.asarray(quad, dtype=float)
    return (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))


def group_crop_rect(rects: list[tuple[float, float, float, float]], pad_px: int, img_w: int, img_h: int) -> PixelRect:
    """Union of `rects` (cluster px), expanded by `pad_px` on every side,
    rounded outwards to whole pixels and clamped to the `img_w`x`img_h`
    render. Always at least 1x1."""
    x0 = min(r[0] for r in rects) - pad_px
    y0 = min(r[1] for r in rects) - pad_px
    x1 = max(r[2] for r in rects) + pad_px
    y1 = max(r[3] for r in rects) + pad_px
    cx0 = min(max(int(math.floor(x0)), 0), img_w - 1)
    cy0 = min(max(int(math.floor(y0)), 0), img_h - 1)
    cx1 = max(min(int(math.ceil(x1)), img_w), cx0 + 1)
    cy1 = max(min(int(math.ceil(y1)), img_h), cy0 + 1)
    return (cx0, cy0, cx1, cy1)


def pad_to_aspect(rgb: np.ndarray, max_aspect: float) -> tuple[np.ndarray, tuple[int, int]]:
    """White-pad `rgb` (HxWx3) symmetrically along its short side so
    long/short <= `max_aspect`. Returns `(padded, (ox, oy))`, the original's
    top-left inside `padded`. Padding, not resizing: FAST rescales its input
    to a 640 px short side, so a thin one-line crop would otherwise be
    blown up far past the text scales FAST was trained on."""
    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        return rgb, (0, 0)
    if w > h * max_aspect:
        new_h, new_w = int(math.ceil(w / max_aspect)), w
    elif h > w * max_aspect:
        new_h, new_w = h, int(math.ceil(h / max_aspect))
    else:
        return rgb, (0, 0)
    ox, oy = (new_w - w) // 2, (new_h - h) // 2
    padded = np.full((new_h, new_w, 3), 255, dtype=np.uint8)
    padded[oy:oy + h, ox:ox + w] = rgb[:, :, :3]
    return padded, (ox, oy)


def build_groups(
    cluster_vectors: list[Vector], bgr: np.ndarray, dpi: int, padding: float,
    quads: list[np.ndarray], quad_bboxes: list[Bbox], first_quad_idx: int,
    *, center_dist_pt: float, pad_px: int, max_aspect: float,
) -> list[FastGroup]:
    """Every `FastGroup` for one cluster's render. `quads` are the
    detector's pixel quads (cluster px), `quad_bboxes` their page-space
    envelopes (same order), and `first_quad_idx` the page-wide `page_quads`
    index of `quads[0]`. The crops copy out of `bgr` (converted to RGB), so
    the full render can be freed right after."""
    h, w = bgr.shape[:2]
    out: list[FastGroup] = []
    for members in group_quads_anchored(quad_bboxes, center_dist_pt):
        rect = group_crop_rect([quad_pixel_rect(quads[i]) for i in members], pad_px, w, h)
        cx0, cy0, cx1, cy1 = rect
        crop_rgb = np.ascontiguousarray(bgr[cy0:cy1, cx0:cx1, :3][:, :, ::-1])
        fast_input, offset = pad_to_aspect(crop_rgb, max_aspect)
        out.append(FastGroup(
            cluster_vectors=cluster_vectors, dpi=dpi, padding=padding,
            quad_idxs=[first_quad_idx + i for i in members], crop_rect=rect,
            page_bbox=union_bbox([quad_bboxes[i] for i in members]),
            crop_page_bbox=pixel_to_page_bbox(
                cluster_vectors, dpi, [(cx0, cy0), (cx1, cy1)], padding,
            ),
            fast_input=fast_input, input_offset=offset,
        ))
    return out


def crop_heat(group: FastGroup, fast_mask: np.ndarray) -> np.ndarray:
    """FAST's score map over `group.fast_input`, cut back to the real crop
    (undoing `pad_to_aspect`)."""
    ox, oy = group.input_offset
    cx0, cy0, cx1, cy1 = group.crop_rect
    return np.asarray(fast_mask, dtype=np.float32)[oy:oy + (cy1 - cy0), ox:ox + (cx1 - cx0)]


def vector_ink_mask(v: Vector, dpi: int, gray_threshold: int) -> InkMask:
    """`v` rendered on its own at `dpi` (stroke-width margin so a thick
    stroke isn't clipped by the frame), thresholded to an ink mask. A
    vector that paints nothing visible yields an all-False mask."""
    pad = (v.width or 0.0) / 2.0 + 1.0
    image = render_vector_cluster([v], dpi, pad)
    gray = np.asarray(image.convert("L"))
    x0, y0 = v.bbox[0] - pad, v.bbox[1] - pad
    return gray < gray_threshold, (x0, y0)


def vector_ink_fraction(
    mask: np.ndarray, origin_px: tuple[float, float], crop_rect: PixelRect,
    heat: np.ndarray, heat_threshold: float,
) -> float:
    """Fraction of `mask`'s ink pixels (placed with pixel (0, 0) at
    `origin_px` in cluster px) that land inside `crop_rect` on a `heat`
    pixel >= `heat_threshold`. Ink outside the crop counts as cold, so a
    long line merely clipping a label scores low. No ink -> 0.0."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return 0.0
    X = xs + int(round(origin_px[0]))
    Y = ys + int(round(origin_px[1]))
    cx0, cy0, cx1, cy1 = crop_rect
    hh, hw = heat.shape[:2]
    inside = (X >= cx0) & (X < cx0 + hw) & (Y >= cy0) & (Y < cy0 + hh) & (X < cx1) & (Y < cy1)
    if not inside.any():
        return 0.0
    hot = heat[Y[inside] - cy0, X[inside] - cx0] >= heat_threshold
    return float(np.count_nonzero(hot)) / float(xs.size)


def split_text_vectors(
    groups: list[FastGroup], *, heat_threshold: float, ink_fraction: float, gray_threshold: int,
    ink_mask_fn: "Callable[[Vector, int, int], InkMask]" = vector_ink_mask,
) -> set[int]:
    """`id()`s of the vectors that stay text: under (bbox-intersecting the
    page bbox of) at least one `accepted` group whose heatmap highlights
    >= `ink_fraction` of their ink. Fills each accepted group's `scores`.
    A vector's mask is rendered once per dpi and reused across groups."""
    text_ids: set[int] = set()
    masks: dict[tuple[int, int], InkMask] = {}
    for group in groups:
        if not group.accepted or group.heat is None:
            continue
        for v in group.cluster_vectors:
            if not bboxes_intersect(v.bbox, group.page_bbox):
                continue
            key = (id(v), group.dpi)
            if key not in masks:
                masks[key] = ink_mask_fn(v, group.dpi, gray_threshold)
            mask, origin_page = masks[key]
            [origin_px] = page_points_to_pixel(group.cluster_vectors, group.dpi, [origin_page], group.padding)
            frac = vector_ink_fraction(mask, origin_px, group.crop_rect, group.heat, heat_threshold)
            group.scores.append((v, frac))
            if frac >= ink_fraction:
                text_ids.add(id(v))
    return text_ids


def downscale_heat(heat: np.ndarray, max_side: int) -> np.ndarray:
    """`heat` (float [0, 1]) as uint8, nearest-sampled so its longest side
    is <= `max_side` -- the small copy kept for the heatmap debug layer."""
    h, w = heat.shape[:2]
    step = max(1, int(math.ceil(max(h, w) / max_side)))
    return (np.clip(heat[::step, ::step], 0.0, 1.0) * 255).astype(np.uint8)
