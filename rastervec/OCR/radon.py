"""Radon-transform text deskew + word segmentation.

A rendered vector-text cluster is usually *roughly* upright but can carry a
small skew (or a 90/180/270 turn, for rotated CAD text). Projecting the ink
onto a swept set of directions -- a Radon transform -- and scoring each
direction by how spiky its 1-D profile is (the classic
projection-profile-variance / Postl criterion) recovers the true baseline
angle: the sharpest profile comes from projecting *along* the text lines,
where the gaps between lines read as deep troughs.

Intuition (the "shine a light through the page" picture): each glyph is
a little wall. Rotate a light source around the page; the detector on the
far side sees the most light when the beam runs cleanly *between* the
lines of text. The beam angle that maximises that contrast is the text
angle.

**Precision note (standing invariant -- read before touching this file):**
`Segment.angle` is Radon's raw, full-precision fine-sweep result
(`RADON_ANGLE_STEP_DEG` resolution, e.g. 0.25 deg) -- it must NEVER be
rounded or snapped to a multiple of 90 anywhere in this pipeline. Both the
post-segmentation similarity check (`pipelines/_steps.py`'s
`group_similar_segments`, which normalizes rotation using this exact value
instead of a page-wide search) and the final OCR `Text.direction` (which
combines this angle with PaddleOCR's cls-detected 180-degree flip) depend
on the full-precision value; rounding it here would silently degrade every
downstream angle to blocky 90-degree steps.

Pipeline use: this now runs directly on Vector_Classification's kept
clusters, *before* similarity grouping and FAST detection (moved earlier
in the pipeline vs. the pre-refactor design, where Radon ran right before
OCR on already-deduped/FAST-passed clusters) -- see
`pipelines/_common.py`. `segment_clusters` renders each cluster once
(transient -- the image itself is never returned or kept) purely to
estimate its skew and word boundaries, then maps each word's crop region
back onto the cluster's own `Vector`s (by bbox overlap) to build a flat
`list[Segment]`, one per word, at real page position. The 0-vs-180 (and
90-vs-270) ambiguity Radon cannot resolve is left to
`OCR/Paddle_OCR/ocr_backend.py`'s PaddleOCR `cls` pass, once a segment's
been deduped down to a `UniqueSegment` and is actually being OCR'd.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import ImageDraw
from skimage.transform import SimilarityTransform, radon, resize, warp

from rastervec.config import (
    MIN_RENDER_SIDE_PX,
    RADON_ANGLE_STEP_DEG,
    RADON_LINE_BAND_MIN_FRAC,
    RADON_MAX_RENDER_SIDE_PX,
    RADON_MIN_GAP_PX,
    RADON_SKEW_LIMIT_DEG,
)
from rastervec.helpers.geometry import (
    PDF_POINTS_PER_INCH,
    bbox_intersection_area,
    union_bbox,
)
from rastervec.models import Segment, Vector
from rastervec.renderer import cluster_frame_size, pixel_to_page_bbox, render_vector_cluster

if TYPE_CHECKING:
    from rastervec.pipelines.result import PipelineResult
    from rastervec.renderer.notebook import RenderResult

# A pixel darker than this counts as glyph ink (0 = black, 255 = white).
INK_LEVEL = 250


# --------------------------------------------------------------------------
# 1-D ink-run helpers
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


def _split_on_gaps(has_ink: np.ndarray, *, gap_threshold: float) -> list[tuple[int, int]]:
    """Group ``_ink_runs(has_ink)`` on any gap wider than `gap_threshold`.
    Fewer than two runs -> a single span over the whole ink extent (``[]``
    if no ink)."""
    runs = _ink_runs(has_ink)
    if len(runs) < 2:
        return [(runs[0][0], runs[-1][1])] if runs else []
    return _group_runs(runs, gap_threshold)


def _line_gaps(has_ink: np.ndarray) -> list[float]:
    """This line's own inter-run gaps (px), for pooling into the
    cluster-wide gap distribution. ``[]`` if fewer than two ink runs."""
    runs = _ink_runs(has_ink)
    if len(runs) < 2:
        return []
    return [float(runs[i + 1][0] - runs[i][1] - 1) for i in range(len(runs) - 1)]


def _cluster_gap_threshold(all_gaps: list[float], *, min_gap: float = RADON_MIN_GAP_PX) -> float:
    """One shared word-split threshold for the whole cluster: the median of
    every inter-run gap pooled across every line (not one line's own gaps)
    -- a gap wider than this starts a new word. `min_gap` floors the
    degenerate case (pooled median at/near zero, e.g. very tight kerning or
    mostly single-run lines), so splitting doesn't collapse to "every run
    is its own word."""
    if not all_gaps:
        return min_gap
    return max(min_gap, float(np.median(all_gaps)))


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
    vertical text), then a fine sweep around that peak refines it to full
    `step_deg` precision -- see this module's precision note."""
    img = ink.astype(np.float64)
    coarse = np.arange(0.0, 180.0, 2.0)
    coarse_best = float(coarse[int(np.argmax(_objective(radon(img, theta=coarse, circle=False))))])
    fine = coarse_best + np.arange(-2.0, 2.0 + step_deg, step_deg)
    fine_best = float(fine[int(np.argmax(_objective(radon(img, theta=fine, circle=False))))])
    return fine_best % 180.0


def estimate_skew_from_mask(ink: np.ndarray) -> float:
    """`estimate_skew` for a pre-computed (possibly downscaled) ink mask.
    Full precision -- see this module's precision note."""
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
    +/-90 for vertical (rotated) text. Full precision -- never rounded to a
    quarter turn. The 0-vs-180 flip is *not* resolved here."""
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


def split_words(line_gray: np.ndarray, *, gap_threshold: float,
                pad: int = 1) -> list[tuple[int, int, int, int]]:
    """`(x0, y0, x1, y1)` pixel boxes, one per word, within one deskewed
    line crop. Both x- and y-extent are each word's own tight ink bbox:
    the column profile is split on `gap_threshold` (the cluster-wide
    pooled-median gap, see `_cluster_gap_threshold`) first, then each
    word's y-extent comes from ink within just that word's own column
    slice -- not the whole line -- so a short word doesn't inherit an
    ascender/descender that only exists in a different word on the same
    line."""
    if line_gray.ndim != 2 or line_gray.size == 0:
        return []
    h, w = line_gray.shape
    ink = to_ink(line_gray)
    if not ink.any():
        return []
    boxes: list[tuple[int, int, int, int]] = []
    for sx0, sx1 in _split_on_gaps(ink.any(axis=0), gap_threshold=gap_threshold):
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


def _downscale_ink_for_radon(gray: np.ndarray) -> np.ndarray:
    """Ink mask, capped at RADON_MAX_RENDER_SIDE_PX on the long side --
    Radon cost is O(pixels * angles), so a huge merged cluster bbox would
    otherwise hang. Angle estimation is scale-invariant, so this only
    affects the estimate, never the returned word boxes."""
    ink = to_ink(gray)
    long_side = max(ink.shape)
    if long_side <= RADON_MAX_RENDER_SIDE_PX:
        return ink
    scale = RADON_MAX_RENDER_SIDE_PX / long_side
    new_hw = (max(1, int(ink.shape[0] * scale)), max(1, int(ink.shape[1] * scale)))
    return resize(ink.astype(np.float64), new_hw, order=1) > 0.5


def render_cluster_for_radon(vectors: list[Vector], dpi: int = 300) -> tuple["np.ndarray", int]:
    """Render `vectors` as Radon/OCR sees it: `dpi` bumped upward (never
    down) so the rendered image's shorter side is at least
    `MIN_RENDER_SIDE_PX`. Returns `(gray, dpi_used)`. Shared by
    `segment_clusters` (Phase D, one render per cluster) and OCR's own
    unique-segment render (Phase G, one render per unique segment) -- same
    "don't hand PaddleOCR a tiny crop" rationale either way."""
    width_pt, height_pt = cluster_frame_size(vectors)
    min_side_pt = min(width_pt, height_pt)
    if min_side_pt > 0:
        needed_dpi = math.ceil(MIN_RENDER_SIDE_PX * PDF_POINTS_PER_INCH / min_side_pt)
        dpi = max(dpi, needed_dpi)
    image = render_vector_cluster(vectors, dpi)
    return to_gray(image), dpi


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _assign_vectors_to_words(
    vectors: list[Vector], word_bboxes: list[tuple[float, float, float, float]],
) -> list[list[Vector]]:
    """Assigns every one of `vectors` to exactly one word bbox -- the one it
    overlaps most, or (no overlap at all) the one whose center it's
    nearest to -- so a cluster's Vectors are fully partitioned across its
    words with none lost and none duplicated. A Vector is never split
    across two words."""
    assignment: list[list[Vector]] = [[] for _ in word_bboxes]
    centers = [_bbox_center(b) for b in word_bboxes]
    for v in vectors:
        best_i, best_score = 0, -1.0
        for i, wb in enumerate(word_bboxes):
            score = bbox_intersection_area(v.bbox, wb)
            if score > best_score:
                best_score, best_i = score, i
        if best_score <= 0.0:
            vcx, vcy = _bbox_center(v.bbox)
            best_i = min(
                range(len(word_bboxes)),
                key=lambda i: math.hypot(centers[i][0] - vcx, centers[i][1] - vcy),
            )
        assignment[best_i].append(v)
    return assignment


def segment_clusters(clusters: list[list[Vector]], *, dpi: int = 300) -> list[Segment]:
    """Radon-segments every cluster into word-level `Segment`s at real page
    position. For each cluster: render (transient, not kept) -> estimate
    skew (full precision) -> deskew -> split into line/word crops ->
    map each word's crop region back onto the cluster's own `Vector`s (by
    bbox overlap, see `_assign_vectors_to_words`) -> one `Segment` per
    non-empty word. A cluster with no ink, or whose deskewed profile
    yields no word bands, is skipped (its Vectors are lost from this
    step's output -- callers should already know FAST/similarity only see
    clusters with real ink)."""
    segments: list[Segment] = []
    for cluster in clusters:
        if not cluster:
            continue
        gray, dpi_used = render_cluster_for_radon(cluster, dpi)
        if gray.size == 0 or not to_ink(gray).any():
            continue

        skew = estimate_skew_from_mask(_downscale_ink_for_radon(gray))
        out_shape, forward, inverse = _rotation(gray.shape, skew)
        deskewed = warp(
            gray, forward.inverse, output_shape=out_shape,
            cval=255.0, order=1, preserve_range=True,
        ).astype(np.uint8)

        prof = row_profile(to_ink(deskewed))
        bands = line_bands(prof)
        if not bands:
            continue

        line_cols = [to_ink(deskewed[by0:by1 + 1, :]).any(axis=0) for by0, by1 in bands]
        gap_threshold = _cluster_gap_threshold([g for cols in line_cols for g in _line_gaps(cols)])

        word_bboxes: list[tuple[float, float, float, float]] = []
        for by0, by1 in bands:
            line = deskewed[by0:by1 + 1, :]
            for wx0, wy0, wx1, wy1 in split_words(line, gap_threshold=gap_threshold):
                gy0, gy1 = by0 + wy0, by0 + wy1
                box = np.array([(wx0, gy0), (wx1, gy0), (wx1, gy1), (wx0, gy1)], dtype=np.float64)
                mapped = inverse(box)
                page_bbox = pixel_to_page_bbox(
                    cluster, dpi_used, [(float(x), float(y)) for x, y in mapped],
                )
                word_bboxes.append(page_bbox)

        if not word_bboxes:
            continue

        for word_vectors in _assign_vectors_to_words(cluster, word_bboxes):
            if word_vectors:
                segments.append(Segment(vectors=word_vectors, angle=float(skew)))

    return segments


# --------------------------------------------------------------------------
# notebook visualization (pipeline_stage_visualization.ipynb's "Segment
# (Radon)" section) -- reads a PipelineResult, never called by the real
# pipeline.
# --------------------------------------------------------------------------
def render_radon(res: "PipelineResult") -> "RenderResult":
    """One row per segmented cluster (grouped back by original cluster
    index via `Segment`'s own vectors): re-render each cluster and draw its
    words' page-space bboxes, mapped to this render's pixel space, on
    top."""
    from rastervec.renderer import page_points_to_pixel
    from rastervec.renderer.notebook import RenderResult

    segments = res.segments or []
    clusters = res.text_clusters or []

    def _flatten(entry) -> list[Vector]:
        if entry and isinstance(entry[0], list):
            return [v for g in entry for v in g]
        return entry

    rows = []
    for cluster_entry in clusters[:8]:
        cluster = _flatten(cluster_entry)
        cluster_ids = {id(v) for v in cluster}
        own_segments = [s for s in segments if any(id(v) in cluster_ids for v in s.vectors)]
        if not own_segments:
            continue
        base = render_vector_cluster(cluster, 300).convert("RGB")
        d = ImageDraw.Draw(base)
        for seg in own_segments:
            bbox = union_bbox([v.bbox for v in seg.vectors])
            pts = page_points_to_pixel(cluster, 300, [(bbox[0], bbox[1]), (bbox[2], bbox[3])])
            d.rectangle([pts[0], pts[1]], outline="#dc2626", width=1)
        caption = f"{len(own_segments)} word(s), angle={own_segments[0].angle:+.2f}deg"
        rows.append({"name": caption, "isolated": base, "overlay": base})

    return RenderResult(
        categories=rows,
        note=f"{len(segments)} segment(s) across {len(clusters)} cluster(s)",
    )
