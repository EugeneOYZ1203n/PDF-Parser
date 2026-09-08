"""Radon-transform text deskew + line/word segmentation.

Replaces the old axis-aligned ink-projection segmentation
(`OCR/Paddle_OCR/ink_segment.py`). A rendered vector-text cluster is
usually *roughly* upright but can carry a small skew (or a 90/180/270
turn, for rotated CAD text). Projecting the ink onto a swept set of
directions -- a Radon transform -- and scoring each direction by how
spiky its 1-D profile is (the classic projection-profile-variance /
Postl criterion) recovers the true baseline angle: the sharpest profile
comes from projecting *along* the text lines, where the gaps between
lines read as deep troughs.

Intuition (the "shine a light through the page" picture): each glyph is
a little wall. Rotate a light source around the page; the detector on the
far side sees the most light when the beam runs cleanly *between* the
lines of text. The beam angle that maximises that contrast is the text
angle.

Pipeline use: `pipelines/sub_pipelines/ocr.py::segment_for_ocr` calls
`segment_cluster` on each cluster render, before OCR recognition. The
0-vs-180 (and 90-vs-270) ambiguity Radon cannot resolve is left to
`RenderOCR.recognize_segmented`, which recognises both ways and keeps the
higher-confidence reading.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import ImageDraw
from skimage.transform import SimilarityTransform, radon, resize, warp

from rastervec.config import (
    RADON_ANGLE_STEP_DEG,
    RADON_LINE_BAND_MIN_FRAC,
    RADON_MAX_RENDER_SIDE_PX,
    RADON_SKEW_LIMIT_DEG,
)
from rastervec.renderer import render_vector_cluster

if TYPE_CHECKING:
    from rastervec.pipelines.result import PipelineResult
    from rastervec.renderer.notebook import RenderResult

# A pixel darker than this counts as glyph ink (0 = black, 255 = white).
# Kept identical to the old ink_segment.INK_LEVEL so ported gap rules are
# unchanged.
INK_LEVEL = 250


# --------------------------------------------------------------------------
# 1-D ink-run helpers (ported verbatim from the old ink_segment.py -- only
# the "assume axis-aligned" caller changed, not this gap math)
# --------------------------------------------------------------------------
def _ink_runs(has_ink: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous inclusive ``[start, end]`` index runs of True in a 1-D
    bool array."""
    idx = np.flatnonzero(has_ink)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [idx.size - 1]))
    return [(int(idx[s]), int(idx[e])) for s, e in zip(starts, ends)]


def _group_runs(runs: list[tuple[int, int]], gap_thresh: float) -> list[tuple[int, int]]:
    """Merge `runs` into spans, starting a new span whenever the gap
    between two consecutive runs exceeds `gap_thresh`."""
    if not runs:
        return []
    spans: list[tuple[int, int]] = []
    grp_start, grp_end = runs[0]
    for start, end in runs[1:]:
        if start - grp_end - 1 > gap_thresh:
            spans.append((grp_start, grp_end))
            grp_start = start
        grp_end = max(grp_end, end)
    spans.append((grp_start, grp_end))
    return spans


def _split_on_gaps(
    has_ink: np.ndarray, *, gap_factor: float, min_gap: float,
) -> list[tuple[int, int]]:
    """Group ``_ink_runs(has_ink)`` on any gap wider than
    ``max(min_gap, gap_factor * median(inter-run gaps))``. Fewer than two
    runs -> a single span over the whole ink extent (``[]`` if no ink)."""
    runs = _ink_runs(has_ink)
    if len(runs) < 2:
        return [(runs[0][0], runs[-1][1])] if runs else []
    gaps = [runs[i + 1][0] - runs[i][1] - 1 for i in range(len(runs) - 1)]
    thresh = max(min_gap, gap_factor * float(np.median(gaps)))
    return _group_runs(runs, thresh)


# --------------------------------------------------------------------------
# Radon primitives
# --------------------------------------------------------------------------
def to_gray(image) -> np.ndarray:
    """A PIL image or ndarray -> a 2-D uint8 grayscale array."""
    arr = np.asarray(image)
    if arr.ndim == 3:
        # luminosity; drop any alpha
        arr = arr[..., :3].astype(np.float64)
        arr = arr @ np.array([0.299, 0.587, 0.114])
    return arr.astype(np.uint8)


def to_ink(gray: np.ndarray) -> np.ndarray:
    """Boolean ink mask (True where darker than INK_LEVEL)."""
    return gray < INK_LEVEL


def _objective(sinogram: np.ndarray) -> np.ndarray:
    """Per-projection alignment score: sum of squared projection values
    (the Postl criterion). Radon roughly preserves total mass across
    angles, so this is maximal for the angle whose profile piles the ink
    into the fewest, sharpest bins -- i.e. projecting perpendicular to the
    text lines, where inter-line whitespace reads as true zeros."""
    return np.sum(sinogram ** 2, axis=0)


def best_theta(ink: np.ndarray, *, limit_deg: float, step_deg: float) -> float:
    """The projection angle (degrees, skimage convention, in [0, 180)) whose
    1-D profile is sharpest -- i.e. parallel to the text baseline. A coarse
    full sweep fixes the gross orientation (0/90/180 ~ horizontal vs
    vertical text), then a fine sweep around that peak refines it."""
    img = ink.astype(np.float64)
    coarse = np.arange(0.0, 180.0, 2.0)
    coarse_best = float(coarse[int(np.argmax(_objective(radon(img, theta=coarse, circle=False))))])
    fine = coarse_best + np.arange(-2.0, 2.0 + step_deg, step_deg)
    fine_best = float(fine[int(np.argmax(_objective(radon(img, theta=fine, circle=False))))])
    return fine_best % 180.0


def estimate_skew_from_mask(ink: np.ndarray) -> float:
    """`estimate_skew` for a pre-computed (possibly downscaled) ink mask."""
    if not ink.any():
        return 0.0
    theta = best_theta(ink, limit_deg=RADON_SKEW_LIMIT_DEG, step_deg=RADON_ANGLE_STEP_DEG)
    # skimage's `radon` theta=90 projects along image rows -> a horizontal
    # baseline. The counter-clockwise rotation that makes the baseline
    # horizontal is (90 - theta), mapped into (-90, 90].
    skew = 90.0 - theta
    if skew <= -90.0:
        skew += 180.0
    elif skew > 90.0:
        skew -= 180.0
    return skew


def estimate_skew(gray: np.ndarray) -> float:
    """Angle (degrees, counter-clockwise positive) to rotate `gray` so its
    text lines become horizontal. Near 0 for already-upright text; near
    +/-90 for vertical (rotated) text. The 0-vs-180 flip is *not* resolved
    here."""
    return estimate_skew_from_mask(to_ink(gray))


def row_profile(ink: np.ndarray) -> np.ndarray:
    """Ink count per row -- the vertical projection profile of an
    already-deskewed crop (peaks = text lines, troughs = inter-line gaps)."""
    return ink.sum(axis=1).astype(np.float64)


def line_bands(profile: np.ndarray, *, min_frac: float | None = None,
               pad: int = 1) -> list[tuple[int, int]]:
    """`(y0, y1)` inclusive row bands where `profile` exceeds
    `min_frac * profile.max()` -- one band per text line."""
    min_frac = RADON_LINE_BAND_MIN_FRAC if min_frac is None else min_frac
    if profile.size == 0 or profile.max() <= 0:
        return []
    has_ink = profile > min_frac * profile.max()
    n = profile.size
    return [
        (max(0, s - pad), min(n - 1, e + pad)) for s, e in _ink_runs(has_ink)
    ]


def line_spacing(profile: np.ndarray) -> float:
    """Dominant text-line pitch (pixels) from the profile's peak spacing.
    0.0 when fewer than two lines are visible."""
    bands = line_bands(profile)
    if len(bands) < 2:
        return 0.0
    centres = [(y0 + y1) / 2.0 for y0, y1 in bands]
    return float(np.median(np.diff(centres)))


def split_words(line_gray: np.ndarray, *, gap_factor: float = 1.9,
                min_gap: float = 2.0, pad: int = 1) -> list[tuple[int, int, int, int]]:
    """`(x0, y0, x1, y1)` pixel boxes, one per word, within one deskewed
    line crop. Both x- and y-extent are each word's own tight ink bbox:
    the column profile is split on the ported median-gap rule first, then
    each word's y-extent comes from ink within just that word's own
    column slice -- not the whole line -- so a short word doesn't inherit
    an ascender/descender that only exists in a different word on the
    same line."""
    if line_gray.ndim != 2 or line_gray.size == 0:
        return []
    h, w = line_gray.shape
    ink = to_ink(line_gray)
    if not ink.any():
        return []
    boxes: list[tuple[int, int, int, int]] = []
    for sx0, sx1 in _split_on_gaps(ink.any(axis=0), gap_factor=gap_factor, min_gap=min_gap):
        word_rows = ink[:, sx0:sx1 + 1].any(axis=1)
        if not word_rows.any():
            continue  # defensive; every column span has ink by construction
        ys = np.flatnonzero(word_rows)
        wy0, wy1 = int(ys[0]), int(ys[-1]) + 1
        boxes.append((
            max(0, sx0 - pad), max(0, wy0 - pad),
            min(w, sx1 + 1 + pad), min(h, wy1 + pad),
        ))
    return boxes


# --------------------------------------------------------------------------
# rotation geometry (own matrix so the corner map-back is exact)
# --------------------------------------------------------------------------
def _rotation(shape_hw: tuple[int, int], angle_deg: float):
    """Return `(out_shape, forward, inverse)` for a rotation of `angle_deg`
    about the input centre, onto a resized canvas that fits the whole
    rotated image (same construction skimage's own `rotate(resize=True)`
    uses, so the deskewed image and the mapped corners stay consistent).

    `forward(pts_xy)` maps input pixel coords -> output pixel coords;
    `inverse(pts_xy)` maps output -> input. Both take/return `(N, 2)`
    arrays of `(x, y)` (column, row). `forward` is also a skimage
    `SimilarityTransform` (so `skimage.transform.warp` accepts
    `forward.inverse` without tripping its bound-method check)."""
    h, w = shape_hw
    centre = np.array((w, h)) / 2.0 - 0.5
    # -angle so `forward` is a visual counter-clockwise rotation by
    # `angle_deg` (image y points down), matching skimage.transform.rotate.
    tf = (
        SimilarityTransform(translation=-centre)
        + SimilarityTransform(rotation=np.deg2rad(-angle_deg))
        + SimilarityTransform(translation=centre)
    )
    box = np.array([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)], dtype=np.float64)
    out_box = tf(box)
    lo = out_box.min(axis=0)
    hi = out_box.max(axis=0)
    ow = int(np.ceil(hi[0] - lo[0])) + 1
    oh = int(np.ceil(hi[1] - lo[1])) + 1
    forward = tf + SimilarityTransform(translation=-lo)

    def inverse(pts: np.ndarray) -> np.ndarray:
        return forward.inverse(np.asarray(pts, dtype=np.float64))

    return (oh, ow), forward, inverse


# --------------------------------------------------------------------------
# public entrypoint
# --------------------------------------------------------------------------
@dataclass
class ClusterSegmentation:
    """One cluster render's deskew + line/word split. `word_crops` are
    deskewed grayscale sub-images ready for `crop_normalize.normalize_line_crop`;
    `word_corners[i]` are the 4 corners of `word_crops[i]` in the *original*
    render's pixel space (so `renderer.pixel_to_page_bbox` can place them)."""

    skew_deg: float
    line_spacing_px: float
    word_crops: list[np.ndarray]
    word_corners: list[list[tuple[float, float]]]
    render_dpi: int = 300
    deskewed_gray: np.ndarray | None = None  # verbose-only
    profile: np.ndarray | None = None        # verbose-only


def _downscale_ink_for_radon(gray: np.ndarray) -> np.ndarray:
    """Ink mask, capped at RADON_MAX_RENDER_SIDE_PX on the long side --
    Radon cost is O(pixels * angles), so a huge merged cluster bbox would
    otherwise hang. Angle estimation is scale-invariant, so this only
    affects the estimate, never the returned crops/corners."""
    ink = to_ink(gray)
    long_side = max(ink.shape)
    if long_side <= RADON_MAX_RENDER_SIDE_PX:
        return ink
    scale = RADON_MAX_RENDER_SIDE_PX / long_side
    new_hw = (max(1, int(ink.shape[0] * scale)), max(1, int(ink.shape[1] * scale)))
    return resize(ink.astype(np.float64), new_hw, order=1) > 0.5


def segment_cluster(image, dpi_used: int = 300, *, verbose: bool = False) -> ClusterSegmentation:
    """Deskew `image` (a cluster render), split into line then word crops."""
    gray = to_gray(image)
    if gray.size == 0 or not to_ink(gray).any():
        return ClusterSegmentation(0.0, 0.0, [], [], render_dpi=dpi_used)

    skew = estimate_skew_from_mask(_downscale_ink_for_radon(gray))
    out_shape, forward, inverse = _rotation(gray.shape, skew)
    deskewed = warp(
        gray, forward.inverse, output_shape=out_shape,
        cval=255.0, order=1, preserve_range=True,
    ).astype(np.uint8)

    prof = row_profile(to_ink(deskewed))
    bands = line_bands(prof)

    crops: list[np.ndarray] = []
    corners: list[list[tuple[float, float]]] = []
    for by0, by1 in bands:
        line = deskewed[by0:by1 + 1, :]
        for wx0, wy0, wx1, wy1 in split_words(line):
            gy0, gy1 = by0 + wy0, by0 + wy1
            crops.append(deskewed[gy0:gy1, wx0:wx1])
            box = np.array([(wx0, gy0), (wx1, gy0), (wx1, gy1), (wx0, gy1)], dtype=np.float64)
            mapped = inverse(box)
            corners.append([(float(x), float(y)) for x, y in mapped])

    return ClusterSegmentation(
        skew_deg=float(skew),
        line_spacing_px=line_spacing(prof),
        word_crops=crops,
        word_corners=corners,
        render_dpi=dpi_used,
        deskewed_gray=deskewed if verbose else None,
        profile=prof if verbose else None,
    )


# --------------------------------------------------------------------------
# notebook visualization (pipeline_stage_visualization.ipynb's "Segment
# (Radon)" section) -- reads a PipelineResult, never called by the real
# pipeline.
# --------------------------------------------------------------------------
def render_radon(res: "PipelineResult") -> "RenderResult":
    """One row per segmented cluster: re-render that cluster's ORIGINAL
    (pre-deskew) image via renderer.render_vector_cluster and draw its
    real seg.word_corners polygons on top -- word_corners already live in
    exactly that image's own pixel space (ClusterSegmentation's own
    contract), so no extra transform is needed."""
    from rastervec.renderer.notebook import RenderResult

    clusters = res.regrouped_clusters or []
    segs = res.segmentations or []
    cor = res.cluster_ocr_results or []
    rows = []
    for cluster, seg, ocr in zip(clusters, segs, cor):
        if not seg.word_crops:
            continue
        base = render_vector_cluster(cluster, seg.render_dpi).convert("RGB")
        d = ImageDraw.Draw(base)
        for corners in seg.word_corners:
            d.polygon(corners, outline="#dc2626", width=1)
        caption = (
            f"skew={seg.skew_deg:+.1f}deg  spacing={seg.line_spacing_px:.0f}px  "
            f"{len(seg.word_crops)} word(s)  ->  {(ocr.resolved.text or '(blank)')[:40]}"
        )
        rows.append({"name": caption, "isolated": base, "overlay": base})

    return RenderResult(
        categories=rows[:8],
        note=f"{sum(1 for s in segs if s.word_crops)} segmented cluster(s) with word boxes",
    )


