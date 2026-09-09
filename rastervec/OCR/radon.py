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

Pipeline use: this runs directly after FAST detection (`pipelines/
_steps.py::detect_text_fast`) and before similarity grouping -- every
FAST-surviving classification cluster gets Radon-segmented here, not just a
deduped set of elected representatives, since dedup itself now happens
*after* this step, at word granularity, using this module's own precise
per-word `angle` instead of a coarser pre-Radon estimate (there is no such
estimate anymore -- FAST never needed a rotation at all, since it scores
each cluster independently). `_common.py` calls `segment_clusters` once
with every FAST-surviving cluster at once. `segment_clusters` renders each
input cluster once, estimates its skew and word boundaries, then maps each
word's crop region back onto the cluster's own `Vector`s (by bbox overlap)
to build a flat `list[Segment]`, one per word, in that cluster's own
frame -- and also captures each word's own deskewed pixel crop directly
into `Segment.image`, so OCR (`OCR/Paddle_OCR/ocr_backend.py::
recognize_segments`) never has to re-render from vectors a second time; an
elected similarity-group representative's canonicalized copy
(`pipelines/_steps.py::elect_unique_segments`) carries that same real image
over unchanged, so OCR never re-renders there either. The 0-vs-180 (and
90-vs-270) ambiguity Radon cannot resolve is left to PaddleOCR's `cls` pass
once a word is actually being OCR'd.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from skimage.transform import SimilarityTransform, radon, resize, warp

from rastervec.config import (
    MIN_RENDER_SIDE_PX,
    RADON_ANGLE_STEP_DEG,
    RADON_LINE_BAND_MIN_FRAC,
    RADON_MAX_RENDER_SIDE_PX,
    RADON_MIN_GAP_PX,
    RADON_MIN_WORD_CHARS,
    RADON_SKEW_LIMIT_DEG,
)
from rastervec.helpers.geometry import (
    PDF_POINTS_PER_INCH,
    bbox_intersection_area,
)
from rastervec.models import Segment, Vector
from rastervec.renderer import cluster_frame_size, pixel_to_page_bbox, render_vector_cluster
from rastervec.renderer.stages import render_radon  # noqa: F401 -- re-exported for callers

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


def _group_runs(runs: list[tuple[int, int]], gap_thresh: float) -> list[tuple[int, int, int]]:
    """Merge `runs` into `(start, end, run_count)` spans, starting a new
    span whenever the gap between two consecutive runs exceeds
    `gap_thresh`. `run_count` is how many of the original `runs` compose
    that span -- the pre-OCR character-count proxy `_enforce_min_run_count`
    merges short spans on."""
    if not runs:
        return []
    spans: list[tuple[int, int, int]] = []
    grp_start, grp_end = runs[0]
    grp_count = 1
    for start, end in runs[1:]:
        if start - grp_end - 1 > gap_thresh:
            spans.append((grp_start, grp_end, grp_count))
            grp_start, grp_count = start, 0
        grp_end = max(grp_end, end)
        grp_count += 1
    spans.append((grp_start, grp_end, grp_count))
    return spans


def _split_on_gaps(has_ink: np.ndarray, *, gap_threshold: float) -> list[tuple[int, int]]:
    """Group ``_ink_runs(has_ink)`` on any gap wider than `gap_threshold`.
    Fewer than two runs -> a single span over the whole ink extent (``[]``
    if no ink). Pure gap-based split, with no minimum-run-count enforcement
    -- see `_split_columns_into_words` for the word-splitting variant that
    adds that on top."""
    runs = _ink_runs(has_ink)
    if len(runs) < 2:
        return [(runs[0][0], runs[-1][1])] if runs else []
    return [(s, e) for s, e, _c in _group_runs(runs, gap_threshold)]


def _enforce_min_run_count(
    spans: list[tuple[int, int, int]], min_chars: int,
) -> list[tuple[int, int, int]]:
    """Repeatedly merges any span whose `run_count` is under `min_chars`
    into a neighboring span (summing counts, extending the interval) --
    the following span, or the previous one when it's the last span --
    until every remaining span clears `min_chars` or only one span is left
    (nothing further to merge into). This overrides the gap-based split:
    a merge happens regardless of how wide the gap originally separating
    the two spans was -- the 3-character minimum outranks the gap
    threshold, since a too-short word hurts PaddleOCR's ability to
    determine orientation more than an over-merged word does."""
    spans = list(spans)
    changed = True
    while changed and len(spans) > 1:
        changed = False
        for i, (s, e, c) in enumerate(spans):
            if c < min_chars:
                if i < len(spans) - 1:
                    ns, ne, nc = spans[i + 1]
                    spans[i:i + 2] = [(s, ne, c + nc)]
                else:
                    ps, pe, pc = spans[i - 1]
                    spans[i - 1:i + 1] = [(ps, e, pc + c)]
                changed = True
                break
    return spans


def _split_columns_into_words(
    has_ink: np.ndarray, *, gap_threshold: float, min_chars: int,
) -> list[tuple[int, int]]:
    """`_split_on_gaps`'s word-splitting counterpart: same gap-based
    grouping, then `_enforce_min_run_count` merges any resulting span with
    fewer than `min_chars` ink runs into a neighbor. Fewer than two runs
    behaves exactly like `_split_on_gaps` (a single span can't be merged
    further)."""
    runs = _ink_runs(has_ink)
    if len(runs) < 2:
        return [(runs[0][0], runs[-1][1])] if runs else []
    spans = _enforce_min_run_count(_group_runs(runs, gap_threshold), min_chars)
    return [(s, e) for s, e, _c in spans]


def _line_run_widths(has_ink: np.ndarray) -> list[float]:
    """This line's own ink-run widths (px) -- the character-width samples
    pooled into the cluster-wide word-split gap threshold (see
    `_cluster_gap_threshold`). ``[]`` if the line has no ink."""
    return [float(end - start + 1) for start, end in _ink_runs(has_ink)]


def _cluster_gap_threshold(all_run_widths: list[float], *, min_gap: float = RADON_MIN_GAP_PX) -> float:
    """One shared word-split threshold for the whole cluster: 15% of the
    widest ink run (character) found anywhere in the cluster, pooled
    across every line (not one line's own runs) -- a gap wider than this
    starts a new word. `min_gap` floors the degenerate case (no runs, or
    every run vanishingly thin), so splitting doesn't collapse to "every
    run is its own word."""
    if not all_run_widths:
        return min_gap
    return max(min_gap, 0.15 * max(all_run_widths))


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
                min_chars: int = RADON_MIN_WORD_CHARS,
                pad: int = 1) -> list[tuple[int, int, int, int]]:
    """`(x0, y0, x1, y1)` pixel boxes, one per word, within one deskewed
    line crop. Both x- and y-extent are each word's own tight ink bbox:
    the column profile is split on `gap_threshold` (the cluster-wide
    pooled 15%-of-widest-character gap, see `_cluster_gap_threshold`)
    first, then any resulting span with fewer than `min_chars` ink runs
    (the pre-OCR proxy for character count) is merged into a neighbor --
    overriding the gap split, since PaddleOCR reads a too-short word's
    orientation poorly -- via `_split_columns_into_words`. Each word's
    y-extent then comes from ink within just that word's own column slice
    -- not the whole line -- so a short word doesn't inherit an
    ascender/descender that only exists in a different word on the same
    line."""
    if line_gray.ndim != 2 or line_gray.size == 0:
        return []
    h, w = line_gray.shape
    ink = to_ink(line_gray)
    if not ink.any():
        return []
    boxes: list[tuple[int, int, int, int]] = []
    for sx0, sx1 in _split_columns_into_words(
        ink.any(axis=0), gap_threshold=gap_threshold, min_chars=min_chars,
    ):
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
    position. For each cluster: render (transient, not kept as a whole --
    but each word's own crop *is* kept, see below) -> estimate skew (full
    precision) -> deskew -> split into line/word crops -> map each word's
    crop region back onto the cluster's own `Vector`s (by bbox overlap, see
    `_assign_vectors_to_words`) -> one `Segment` per non-empty word, its
    `image` the word's own deskewed pixel crop (so OCR never has to
    re-render from vectors). A cluster with no ink, or whose deskewed
    profile yields no word bands, is skipped (its Vectors are lost from
    this step's output -- callers should already know FAST/similarity only
    see clusters with real ink)."""
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
        gap_threshold = _cluster_gap_threshold([w for cols in line_cols for w in _line_run_widths(cols)])

        word_bboxes: list[tuple[float, float, float, float]] = []
        word_images: list[np.ndarray] = []
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
                word_images.append(deskewed[gy0:gy1, wx0:wx1])

        if not word_bboxes:
            continue

        assignments = _assign_vectors_to_words(cluster, word_bboxes)
        for word_vectors, image in zip(assignments, word_images):
            if word_vectors:
                segments.append(Segment(vectors=word_vectors, angle=float(skew), image=image))

    return segments
