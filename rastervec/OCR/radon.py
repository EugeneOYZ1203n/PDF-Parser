"""Radon-transform text deskew + word segmentation.

A rendered vector-text cluster is usually *roughly* upright but can carry a
small skew (or a 90/180/270 turn, for rotated CAD text). Rotating the ink
mask through a swept set of angles, taking each rotation's row-projection
profile (`scipy.ndimage.rotate` then `sum(axis=1)` -- much faster per angle
than a full `skimage` Radon sinogram), and scoring each by the *quality of
the whitespace gaps* in that profile recovers the true baseline angle: when
the rows run cleanly between the text lines, the inter-line valleys drop to
(near) zero and there are only a handful of them; a misaligned angle smears
the lines together (shallow valleys) and a 90-degrees-off angle resolves
individual words/characters instead (many valleys).

**The objective (per candidate angle `theta`, minimised):**
`mean(gap_score) * sqrt(n_gaps)`.

  * A *gap* is a profile valley strictly between two detected peaks -- it
    must have a peak on the left *and* the right; a leading/trailing margin
    is not a gap.
  * `gap_score = 1 - ((L+R)/2 - M) / ((L+R)/2 + eps)` where `L`/`R` are the
    bounding peak heights and `M` the valley minimum. `g -> 0` is a clean
    (deep) gap, `g = 1` a bad (shallow) one, so a *lower* mean score is a
    *better* angle.
  * `sqrt(n_gaps)` penalises angles that fragment the ink into many gaps
    (word/character combs, not line gaps) and makes the single-line case
    fall out: with one text line the across-lines projection has no gaps at
    all (objective `inf`), so the sweep locks onto the along-lines comb
    instead -- and the 90-degrees-from-best check below turns that back into
    the real skew.

**Single line:** with only one text line there is no inter-line gap
structure for the objective to lock onto (its finite window spans almost
the whole sweep rather than a narrow basin), so `_across_lines_angle`
returns `None` and `_sweep_deskew` falls back to the classic Postl
`sum(profile**2)` criterion -- the rotation that concentrates the one line
into the sharpest single band. (This is the weakest part of the estimate:
Postl is reliable for bold text but can mis-call thin single-line text by a
quarter turn -- see the module's caller notes.)

**Precision note (standing invariant -- read before touching this file):**
`Segment.angle` is Radon's raw, full-precision sweep result
(`RADON_ANGLE_STEP_DEG` resolution, e.g. 0.25 deg) -- it must NEVER be
rounded or snapped to a multiple of 90 anywhere in this pipeline. Both the
post-segmentation similarity check (`pipelines/_steps.py`'s
`group_similar_segments`, which normalizes rotation using this exact value
instead of a page-wide search) and the final OCR `Text.direction` (which
combines this angle with PaddleOCR's cls-detected 180-degree flip) depend
on the full-precision value; rounding it here would silently degrade every
downstream angle to blocky 90-degree steps.

**Two padding mechanisms, both decided here.** `pad_image` is the pipeline's
pixel-space padding step: the OCR backend hands `Segment.image` to
PaddleOCR verbatim (there is no crop-normalization pass anymore), so the
breathing-room margin the OCR path needs comes from `segment_clusters`' two
explicit `pad_image` calls -- one around the whole cluster render, one
around each word crop. Separately, `render_cluster_for_radon` passes a
small page-space `padding` into `renderer.render_vector_cluster` itself
(sized from the cluster's own max stroke width, see
`RADON_RENDER_PADDING_EXTRA_PT`), so a stroke sitting at the exact edge of
the cluster's bbox isn't clipped by the render frame. Don't push either back
into the renderer or the backend; the point of the current shape is that
both paddings are readable at the call site.

Pipeline use: this runs directly after FAST detection (`pipelines/
_steps.py::detect_text_fast`) and before similarity grouping -- every
FAST-surviving classification cluster gets Radon-segmented here, not just a
deduped set of elected representatives, since dedup itself now happens
*after* this step, at word granularity, using this module's own precise
per-cluster `angle` instead of a coarser pre-Radon estimate. `_common.py`
calls `segment_clusters` once with every FAST-surviving cluster at once.
`segment_clusters` renders each input cluster once, estimates its skew,
splits it into line bands (on the *good* gaps) and then into
aspect-ratio-bounded word segments, then maps each segment's crop region
back onto the cluster's own `Vector`s (by bbox overlap) to build a flat
`list[Segment]`, one per segment. It also captures each segment's own
deskewed pixel crop -- grown to the union of its tight box and its assigned
`Vector`s' own bboxes, so clipped ascenders/descenders are recovered from
real geometry rather than an ink-pixel scan -- directly into
`Segment.image`, so OCR
(`OCR/Paddle_OCR/ocr_backend.py::recognize_segments`) never has to
re-render from vectors. The 0-vs-180 (and 90-vs-270) ambiguity Radon cannot
resolve is left to PaddleOCR's `cls` pass once a word is actually being
OCR'd.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.ndimage import rotate as _nd_rotate
from skimage.transform import SimilarityTransform, resize, warp

from rastervec.config import (
    MAX_RENDER_DPI,
    MIN_RENDER_SIDE_PX,
    RADON_ANGLE_STEP_DEG,
    RADON_COARSE_STEP_DEG,
    RADON_GAP_MEDIAN_MULTIPLIER,
    RADON_GAP_SCORE_EPS,
    RADON_GOOD_GAP_MAX,
    RADON_MAX_RENDER_SIDE_PX,
    RADON_MAX_SEGMENT_ASPECT,
    RADON_MIN_GAP_PX,
    RADON_MULTILINE_BASIN_MAX_DEG,
    RADON_MIN_WORD_CHARS,
    RADON_PAD_FRACTION,
    RADON_PEAK_MIN_FRAC,
    RADON_PROFILE_SMOOTH_PX,
    RADON_RENDER_PADDING_EXTRA_PT,
    RADON_SKEW_LIMIT_DEG,
)
from rastervec.helpers.geometry import (
    PDF_POINTS_PER_INCH,
    bbox_intersection_area,
    union_bbox,
)
from rastervec.models import Segment, Vector
from rastervec.renderer import (
    page_points_to_pixel,
    pixel_to_page_bbox,
    render_vector_cluster,
)
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


def _line_gaps(has_ink: np.ndarray) -> list[float]:
    """This line's own inter-run gaps (px) -- the samples pooled into the
    cluster-wide word-split gap threshold (see `_cluster_gap_threshold`).
    ``[]`` if the line has fewer than two ink runs."""
    runs = _ink_runs(has_ink)
    if len(runs) < 2:
        return []
    return [float(runs[i + 1][0] - runs[i][1] - 1) for i in range(len(runs) - 1)]


def _cluster_gap_threshold(all_gaps: list[float], *, min_gap: float = RADON_MIN_GAP_PX) -> float:
    """One shared word-split threshold for the whole cluster:
    `RADON_GAP_MEDIAN_MULTIPLIER` times the median inter-run gap, pooled
    across every line (not one line's own gaps) -- a gap wider than this
    starts a new word. Letter gaps outnumber word gaps, so that pooled
    median *is* the typical intra-word letter gap; sitting just above it
    separates words from letters. `min_gap` floors the degenerate case
    (no gaps at all, or a median at/near zero from very tight kerning or
    mostly single-run lines), so splitting doesn't collapse to "every run
    is its own word."""
    if not all_gaps:
        return min_gap
    return max(min_gap, RADON_GAP_MEDIAN_MULTIPLIER * float(np.median(all_gaps)))


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


# --------------------------------------------------------------------------
# gap-quality skew objective
# --------------------------------------------------------------------------
def _smooth(profile: np.ndarray, *, window: int | None = None) -> np.ndarray:
    """`profile` blurred with a moving average of `window`
    (`RADON_PROFILE_SMOOTH_PX`) bins -- so single-bin noise doesn't split a
    real peak or spike a valley."""
    window = RADON_PROFILE_SMOOTH_PX if window is None else window
    p = np.asarray(profile, dtype=np.float64)
    k = max(1, int(round(window)))
    if k <= 1 or p.size == 0:
        return p
    return np.convolve(p, np.ones(k) / k, mode="same")


def _profile_peaks(profile: np.ndarray, *, min_frac: float | None = None) -> list[tuple[int, int]]:
    """Inclusive `(start, end)` index runs of the smoothed `profile` above
    `min_frac` (`RADON_PEAK_MIN_FRAC`) of its peak -- one run per text-line
    band. A line projects to a plateau, not a spike, so a contiguous
    above-threshold run *is* the peak."""
    min_frac = RADON_PEAK_MIN_FRAC if min_frac is None else min_frac
    sm = _smooth(profile)
    if sm.size == 0 or sm.max() <= 0:
        return []
    return _ink_runs(sm > min_frac * sm.max())


def _gap_scores(profile: np.ndarray) -> list[float]:
    """One `gap_score` per valley strictly between two consecutive
    `_profile_peaks` (so every gap has both a left and a right peak):
    `1 - ((L+R)/2 - M) / ((L+R)/2 + eps)` with `L`/`R` the bounding peak
    heights and `M` the valley minimum. `0.0` = a clean, deep gap;
    `1.0` = a shallow one. `[]` for fewer than two peaks."""
    sm = _smooth(profile)
    peaks = _profile_peaks(profile)
    scores: list[float] = []
    for (s0, e0), (s1, e1) in zip(peaks, peaks[1:]):
        left = float(sm[s0:e0 + 1].max())
        right = float(sm[s1:e1 + 1].max())
        mid = (left + right) / 2.0
        between = sm[e0 + 1:s1]
        valley = float(between.min()) if between.size else min(left, right)
        g = 1.0 - (mid - valley) / (mid + RADON_GAP_SCORE_EPS)
        scores.append(float(np.clip(g, 0.0, 1.0)))
    return scores


def _skew_objective(profile: np.ndarray) -> float:
    """`mean(gap_score) * sqrt(n_gaps)` for one projection profile -- lower
    is a better-aligned angle. `inf` when the profile has fewer than two
    peaks (no gap to align on -- e.g. a single text line projected across
    its baseline), so an argmin sweep never *prefers* a zero-gap angle."""
    scores = _gap_scores(profile)
    if not scores:
        return math.inf
    return float(np.mean(scores) * math.sqrt(len(scores)))


# A projection function maps `(mask, angle_deg) -> 1-D row profile`. The
# default `_project` sums ink per row; an alternative (e.g. counting ink
# *runs* per row, which is stroke-weight-independent) can be threaded
# through the whole sweep via the `project=` parameter on
# `estimate_skew_from_mask` and friends -- see `notebooks/skew_method_
# comparison.ipynb`.
ProjectFn = Callable[[np.ndarray, float], np.ndarray]


def _project(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    """Row-projection profile of `mask` (a float ink mask) after rotating it
    `angle_deg` counter-clockwise -- `angle_deg = 0` is the raw
    `sum(axis=1)`. `scipy.ndimage.rotate` is ~7x faster per angle than
    `skimage.transform.radon`, which rebuilds a full sinogram."""
    if abs(angle_deg) < 1e-6:
        return mask.sum(axis=1).astype(np.float64)
    rot = _nd_rotate(mask, angle_deg, reshape=True, order=1, cval=0.0, prefilter=False)
    return rot.sum(axis=1).astype(np.float64)


def _objective_sweep(
    mask: np.ndarray, angles: np.ndarray, *, project: ProjectFn = _project,
) -> np.ndarray:
    return np.array([_skew_objective(project(mask, float(a))) for a in angles])


def _postl_deskew(mask: np.ndarray, *, project: ProjectFn = _project) -> float:
    """Deskew angle (degrees) by the classic Postl criterion -- the
    `sum(profile**2)`-maximising rotation, i.e. the one whose profile is
    most concentrated (the across-lines direction). Robust when the gap
    objective degenerates (a single glyph, or a single line seen from
    almost every angle), so it's the fallback there."""
    coarse = np.arange(-90.0, 90.0, RADON_COARSE_STEP_DEG)
    scores = np.array([float(np.sum(project(mask, float(a)) ** 2)) for a in coarse])
    c_best = float(coarse[int(np.argmax(scores))])
    fine = c_best + np.arange(
        -RADON_COARSE_STEP_DEG, RADON_COARSE_STEP_DEG + RADON_ANGLE_STEP_DEG, RADON_ANGLE_STEP_DEG,
    )
    scores = np.array([float(np.sum(project(mask, float(a)) ** 2)) for a in fine])
    return float(fine[int(np.argmax(scores))])


def _run_containing(runs: list[tuple[int, int]], idx: int) -> tuple[int, int] | None:
    for s, e in runs:
        if s <= idx <= e:
            return (s, e)
    return None


def _across_lines_angle(angles: np.ndarray, objs: np.ndarray) -> float | None:
    """The deskew angle from one `_skew_objective` sweep -- the centre of
    the multi-line line-gap basin -- or `None` when the sweep shows no such
    basin (a single line, or a single glyph), so the caller falls back to
    `_postl_deskew`.

    Rotating a *multi-line* block away from its baseline shears the line
    peaks until they merge and the objective jumps to `inf`, so the
    finite-objective window around the minimum is narrow and bounded by
    `inf` on both sides; its centre is a far more stable estimate than the
    near-flat interior argmin. A single line keeps >= 2 word/letter peaks
    at almost every angle, so its finite window is wide (or the whole
    sweep) -- not a basin -- and this returns `None`."""
    finite = np.isfinite(objs)
    if not finite.any() or finite.all():
        return None
    best = int(np.argmin(np.where(finite, objs, np.inf)))
    frun = _run_containing(_ink_runs(finite), best)
    if frun is None:
        return None
    lo, hi = frun
    bounded = lo > 0 and hi < len(objs) - 1
    if bounded and (angles[hi] - angles[lo]) <= RADON_MULTILINE_BASIN_MAX_DEG:
        return float((angles[lo] + angles[hi]) / 2.0)
    return None


def _sweep_deskew(mask: np.ndarray, *, project: ProjectFn = _project) -> float:
    """Deskew angle (degrees, counter-clockwise positive) that makes the
    text baseline horizontal. A coarse full `[-90, 90)` sweep at
    `RADON_COARSE_STEP_DEG` locates the regime and a rough angle; a fine
    sweep at `RADON_ANGLE_STEP_DEG` over `+/- RADON_SKEW_LIMIT_DEG` around
    it pins the edges. Falls back to the Postl criterion when the gap
    objective has no usable structure."""
    coarse = np.arange(-90.0, 90.0, RADON_COARSE_STEP_DEG)
    rough = _across_lines_angle(coarse, _objective_sweep(mask, coarse, project=project))
    if rough is None:
        return _postl_deskew(mask, project=project)

    fine = rough + np.arange(
        -RADON_SKEW_LIMIT_DEG, RADON_SKEW_LIMIT_DEG + RADON_ANGLE_STEP_DEG, RADON_ANGLE_STEP_DEG,
    )
    refined = _across_lines_angle(fine, _objective_sweep(mask, fine, project=project))
    return float(refined if refined is not None else rough)


def estimate_skew_from_mask(ink: np.ndarray, *, project: ProjectFn = _project) -> float:
    """`estimate_skew` for a pre-computed (possibly downscaled) ink mask.
    Full precision -- see this module's precision note. The 0-vs-180 flip is
    *not* resolved here.

    `project` swaps the row-projection function the whole sweep runs on
    (default `_project`, ink sum per row). Not used by the pipeline; it's
    the hook `notebooks/skew_method_comparison.ipynb` uses to try an
    ink-*run*-count projection without patching the module."""
    if not ink.any():
        return 0.0
    mask = ink.astype(np.float64)
    skew = _sweep_deskew(mask, project=project) % 180.0
    if skew > 90.0:
        skew -= 180.0
    return float(skew)


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


# --------------------------------------------------------------------------
# line / segment splitting
# --------------------------------------------------------------------------
def line_bands(profile: np.ndarray, *, pad: int = 1) -> list[tuple[int, int]]:
    """`(y0, y1)` inclusive row bands, one per text line, from the deskewed
    row `profile`. Peaks come from `_profile_peaks`; two consecutive peaks
    are a genuine line boundary only when the valley between them is a
    *good* gap (`_gap_scores` value below `RADON_GOOD_GAP_MAX`) -- peaks
    separated by a shallow valley (descenders bridging the gap, dotted
    rows) stay one band. Each band is padded `pad` rows on each side
    (clamped)."""
    peaks = _profile_peaks(profile)
    if not peaks:
        return []
    n = profile.size

    def _clamp(s: int, e: int) -> tuple[int, int]:
        return (max(0, s - pad), min(n - 1, e + pad))

    if len(peaks) == 1:
        return [_clamp(*peaks[0])]

    scores = _gap_scores(profile)  # exactly len(peaks) - 1 of them
    bands: list[tuple[int, int]] = []
    grp_s, grp_e = peaks[0]
    for (s1, e1), g in zip(peaks[1:], scores):
        if g < RADON_GOOD_GAP_MAX:
            bands.append((grp_s, grp_e))
            grp_s = s1
        grp_e = e1
    bands.append((grp_s, grp_e))
    return [_clamp(s, e) for s, e in bands]


def line_spacing(profile: np.ndarray) -> float:
    """Dominant text-line pitch (pixels) from the profile's peak spacing.
    0.0 when fewer than two lines are visible."""
    bands = line_bands(profile)
    if len(bands) < 2:
        return 0.0
    centres = [(y0 + y1) / 2.0 for y0, y1 in bands]
    return float(np.median(np.diff(centres)))


def group_by_aspect(
    line_gray: np.ndarray, *, gap_threshold: float,
    max_aspect: float = RADON_MAX_SEGMENT_ASPECT,
    min_chars: int = RADON_MIN_WORD_CHARS, pad: int = 1,
) -> list[tuple[int, int, int, int]]:
    """`(x0, y0, x1, y1)` pixel boxes, one per OCR segment, within one
    deskewed line crop.

    The line is first split into words exactly as before -- the column
    profile is split on `gap_threshold` (the cluster-wide pooled
    1.3x-median-gap, see `_cluster_gap_threshold`), then any span with
    fewer than `min_chars` ink runs is merged into a neighbor
    (`_split_columns_into_words`) -- then consecutive words are greedily
    grouped left to right into segments, closing a segment (and starting a
    new one) as soon as adding the next word would push its aspect ratio
    (segment width / the line's own ink height) to `max_aspect` or above.
    So "I love pineapples very much" becomes a few OCR-friendly chunks
    instead of one absurdly wide crop; a single word already wider than the
    limit is its own segment (words are never split).

    Each segment's y-extent is the tight ink bbox within just that
    segment's own column slice -- not the whole line -- so a short segment
    does not inherit a tall neighbour's ascender/descender."""
    if line_gray.ndim != 2 or line_gray.size == 0:
        return []
    h, w = line_gray.shape
    ink = to_ink(line_gray)
    if not ink.any():
        return []

    row_runs = _ink_runs(ink.any(axis=1))
    line_ink_h = max(1, row_runs[-1][1] - row_runs[0][0] + 1) if row_runs else 1

    word_spans = _split_columns_into_words(
        ink.any(axis=0), gap_threshold=gap_threshold, min_chars=min_chars,
    )
    if not word_spans:
        return []

    groups: list[tuple[int, int]] = []
    grp_s, grp_e = word_spans[0]
    for sx0, sx1 in word_spans[1:]:
        if (sx1 - grp_s) / line_ink_h < max_aspect:
            grp_e = sx1
        else:
            groups.append((grp_s, grp_e))
            grp_s, grp_e = sx0, sx1
    groups.append((grp_s, grp_e))

    boxes: list[tuple[int, int, int, int]] = []
    for sx0, sx1 in groups:
        seg_rows = ink[:, sx0:sx1 + 1].any(axis=1)
        if not seg_rows.any():
            continue  # defensive; every column span has ink by construction
        ys = np.flatnonzero(seg_rows)
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


def render_cluster_for_radon(
    vectors: list[Vector], dpi: int = 300, padding: float = 0.0,
) -> tuple["np.ndarray", int]:
    """Render `vectors` as Radon/OCR sees it: `dpi` bumped upward (never
    down) so the rendered image's shorter side is at least
    `MIN_RENDER_SIDE_PX` -- "don't hand PaddleOCR a tiny crop" -- and
    capped at `MAX_RENDER_DPI`, since the render frame is the bare bbox
    and a degenerate sub-point cluster would otherwise demand an unbounded
    dpi to reach that minimum. Returns `(gray, dpi_used)`. `padding` (PDF
    points, 0 by default) is forwarded to `renderer.render_vector_cluster`,
    expanding the render frame on every side -- `segment_clusters` sizes it
    from the cluster's own max stroke width so a thick stroke at the bbox
    edge isn't clipped; `segment_clusters` separately adds its own margin
    in pixel space afterward, via `pad_image`."""
    x0, y0, x1, y1 = union_bbox([v.bbox for v in vectors])
    min_side_pt = min(x1 - x0, y1 - y0) + 2 * padding
    if min_side_pt > 0:
        needed_dpi = math.ceil(MIN_RENDER_SIDE_PX * PDF_POINTS_PER_INCH / min_side_pt)
        dpi = min(max(dpi, needed_dpi), MAX_RENDER_DPI)
    image = render_vector_cluster(vectors, dpi, padding)
    return to_gray(image), dpi


def pad_image(
    img: np.ndarray, fraction: float = RADON_PAD_FRACTION,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Surround `img` with a white border of `fraction * width` px left and
    right and `fraction * height` px top and bottom, returning
    `(padded, (pad_x_px, pad_y_px))`.

    **This is the pipeline's only padding step.** Nothing upstream adds a
    margin -- `renderer.render_vector_cluster` renders a cluster's bare
    `union_bbox`, and PaddleOCR is handed `Segment.image` as-is -- so both
    the room Radon's deskew/line-band pass needs at the frame edge and the
    breathing room the recognizer wants around a word come from here. It is
    called twice in `segment_clusters` (whole cluster render, then each
    word crop) precisely so both are visible at the call site.

    `pad_x_px`/`pad_y_px` are what a caller must subtract to get back into
    the *unpadded* render's pixel space -- which is what
    `renderer.pixel_to_page_bbox` inverts. A zero-area image is returned
    unchanged with a `(0, 0)` offset."""
    if img.size == 0:
        return img, (0, 0)
    pad_y = int(round(img.shape[0] * fraction))
    pad_x = int(round(img.shape[1] * fraction))
    padded = np.pad(
        img, ((pad_y, pad_y), (pad_x, pad_x)), mode="constant", constant_values=255,
    )
    return padded, (pad_x, pad_y)


# --------------------------------------------------------------------------
# crop growth + vector assignment
# --------------------------------------------------------------------------
def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _assign_vectors_to_segments(
    vectors: list[Vector], segment_bboxes: list[tuple[float, float, float, float]],
) -> list[list[Vector]]:
    """Assigns every one of `vectors` to exactly one segment bbox -- the one
    it overlaps most, or (no overlap at all) the one whose center it's
    nearest to -- so a cluster's Vectors are fully partitioned across its
    segments with none lost and none duplicated. A Vector is never split
    across two segments."""
    assignment: list[list[Vector]] = [[] for _ in segment_bboxes]
    centers = [_bbox_center(b) for b in segment_bboxes]
    for v in vectors:
        best_i, best_score = 0, -1.0
        for i, sb in enumerate(segment_bboxes):
            score = bbox_intersection_area(v.bbox, sb)
            if score > best_score:
                best_score, best_i = score, i
        if best_score <= 0.0:
            vcx, vcy = _bbox_center(v.bbox)
            best_i = min(
                range(len(segment_bboxes)),
                key=lambda i: math.hypot(centers[i][0] - vcx, centers[i][1] - vcy),
            )
        assignment[best_i].append(v)
    return assignment


def _blank_segmentation_dbg(cluster: list[Vector]) -> dict:
    return {
        "cluster_bbox": union_bbox([v.bbox for v in cluster]),
        "line_gap_lines": [],
        "word_gap_lines": [],
        "segment_bboxes": [],
        "grown_segment_bboxes": [],
        "dropped_segment_bboxes": [],
        "assigned_vector_bboxes": [],
        "dropped_vector_bboxes": [],
    }


def segment_clusters(
    clusters: list[list[Vector]], *, dpi: int = 300, debug_out: "list | None" = None,
) -> list[Segment]:
    """Radon-segments every cluster into aspect-bounded word `Segment`s at
    real page position. For each cluster: render (transient, padded by half
    the cluster's own max stroke width so thick strokes at the bbox edge
    aren't clipped) -> **pad** -> estimate skew (full precision) -> deskew
    -> split into line bands on the good gaps -> split each band into
    aspect-ratio-bounded word segments (`group_by_aspect`) -> map each
    segment's *tight* crop region back onto the cluster's own `Vector`s (by
    bbox overlap, see `_assign_vectors_to_segments`) -> grow each kept
    segment's page-space bbox to the union of its tight box and its
    assigned `Vector`s' own bboxes (recovering ascenders/descenders the
    line band clipped, from real geometry rather than an ink-pixel scan) ->
    one `Segment` per non-empty segment, its `image` that grown region's
    own **padded** deskewed pixel crop (so OCR never has to re-render from
    vectors, and needs no normalization pass of its own).

    The two `pad_image` calls are the pipeline's pixel-space padding -- see
    that function's docstring. Both are inline here rather than hidden in
    the renderer or the OCR backend so the margins are visible where they
    happen; the price is the `- pad_x_px / - pad_y_px` correction on the
    way back out to page space (and its `+ pad_x_px / + pad_y_px` inverse
    on the way back in for the grown-box crop). Vector assignment
    (`_assign_vectors_to_segments`) always runs on the *tight* boxes, so
    growth can never steal a neighbour's vectors -- it can only enlarge a
    segment's own page footprint afterward.

    A cluster with no ink, or whose deskewed profile yields no line bands,
    is skipped (its Vectors are lost from this step's output -- callers
    should already know FAST/similarity only see clusters with real ink)."""
    segments: list[Segment] = []
    for cluster in clusters:
        if not cluster:
            continue
        stroke_padding = (
            max((v.width or 0.0) for v in cluster) / 2.0 + RADON_RENDER_PADDING_EXTRA_PT
        )
        gray, dpi_used = render_cluster_for_radon(cluster, dpi, padding=stroke_padding)
        if gray.size == 0 or not to_ink(gray).any():
            if debug_out is not None:
                d = _blank_segmentation_dbg(cluster)
                d["dropped_vector_bboxes"] = [v.bbox for v in cluster]
                debug_out.append(d)
            continue
        # Pad 1 of 2: the whole cluster render, so deskew's warp and
        # `line_bands`' band padding have room at the frame edge. Every
        # pixel coordinate below is in this padded space until it's mapped
        # back out by subtracting (pad_x_px, pad_y_px).
        gray, (pad_x_px, pad_y_px) = pad_image(gray)

        skew = estimate_skew_from_mask(_downscale_ink_for_radon(gray))
        out_shape, forward, inverse = _rotation(gray.shape, skew)
        deskewed = warp(
            gray, forward.inverse, output_shape=out_shape,
            cval=255.0, order=1, preserve_range=True,
        ).astype(np.uint8)

        deskew_ink = to_ink(deskewed)
        height, width = deskewed.shape
        prof = row_profile(deskew_ink)
        bands = line_bands(prof)
        if not bands:
            if debug_out is not None:
                d = _blank_segmentation_dbg(cluster)
                d["dropped_vector_bboxes"] = [v.bbox for v in cluster]
                debug_out.append(d)
            continue

        line_cols = [deskew_ink[by0:by1 + 1, :].any(axis=0) for by0, by1 in bands]
        gap_threshold = _cluster_gap_threshold(
            [g for cols in line_cols for g in _line_gaps(cols)]
        )

        def _px_to_page_bbox(pts_px):
            mapped = inverse(np.array(pts_px, dtype=np.float64))
            return pixel_to_page_bbox(
                cluster, dpi_used,
                [(float(x) - pad_x_px, float(y) - pad_y_px) for x, y in mapped],
                padding=stroke_padding,
            )

        dbg = None
        if debug_out is not None:
            line_gap_lines = []
            for by0, by1 in [(bands[0][0], bands[0][0])] + [
                ((bands[i - 1][1] + bands[i][0]) // 2,) * 2 for i in range(1, len(bands))
            ] + [(bands[-1][1], bands[-1][1])]:
                line_gap_lines.append(
                    _px_to_page_bbox([(0, by0), (width, by0), (width, by1), (0, by1)])
                )
            dbg = _blank_segmentation_dbg(cluster)
            dbg["line_gap_lines"] = line_gap_lines
            debug_out.append(dbg)

        seg_bboxes: list[tuple[float, float, float, float]] = []
        for by0, by1 in bands:
            line = deskewed[by0:by1 + 1, :]
            _prev_wx1 = None
            for wx0, wy0, wx1, wy1 in group_by_aspect(
                line, gap_threshold=gap_threshold, pad=0,
            ):
                gy0, gy1 = by0 + wy0, by0 + wy1
                if dbg is not None and _prev_wx1 is not None:
                    xm = (_prev_wx1 + wx0) / 2.0
                    dbg["word_gap_lines"].append(
                        _px_to_page_bbox([(xm, by0), (xm, by0), (xm, by1), (xm, by1)])
                    )
                _prev_wx1 = wx1
                # page bbox from the TIGHT box: `inverse` lands in the
                # *padded* render's pixel space; `pixel_to_page_bbox`
                # inverts the unpadded one.
                box = np.array(
                    [(wx0, gy0), (wx1, gy0), (wx1, gy1), (wx0, gy1)], dtype=np.float64,
                )
                mapped = inverse(box)
                page_bbox = pixel_to_page_bbox(
                    cluster, dpi_used,
                    [(float(x) - pad_x_px, float(y) - pad_y_px) for x, y in mapped],
                    padding=stroke_padding,
                )
                seg_bboxes.append(page_bbox)

        if not seg_bboxes:
            if dbg is not None:
                dbg["dropped_vector_bboxes"] = [v.bbox for v in cluster]
            continue

        assignments = _assign_vectors_to_segments(cluster, seg_bboxes)
        for seg_vectors, tight_bbox in zip(assignments, seg_bboxes):
            if not seg_vectors:
                if dbg is not None:
                    dbg["dropped_segment_bboxes"].append(tight_bbox)
                continue

            # Grow to the union of the tight box and its assigned Vectors'
            # own bboxes -- real geometry, not an ink-pixel scan -- then map
            # that page-space region back into this cluster's deskewed/
            # padded pixel space (the exact reverse of `_px_to_page_bbox`)
            # to build the OCR crop.
            grown_bbox = union_bbox([tight_bbox] + [v.bbox for v in seg_vectors])
            gpx0, gpy0, gpx1, gpy1 = grown_bbox
            corners_page = [(gpx0, gpy0), (gpx1, gpy0), (gpx1, gpy1), (gpx0, gpy1)]
            unpadded_px = page_points_to_pixel(
                cluster, dpi_used, corners_page, padding=stroke_padding,
            )
            padded_px = np.array(
                [(x + pad_x_px, y + pad_y_px) for x, y in unpadded_px], dtype=np.float64,
            )
            deskewed_px = forward(padded_px)
            gx0 = max(0, int(math.floor(float(deskewed_px[:, 0].min()))))
            gx1 = min(width, int(math.ceil(float(deskewed_px[:, 0].max()))))
            ggy0 = max(0, int(math.floor(float(deskewed_px[:, 1].min()))))
            ggy1 = min(height, int(math.ceil(float(deskewed_px[:, 1].max()))))
            # Pad 2 of 2: this segment's own crop, handed to PaddleOCR
            # verbatim as `Segment.image`.
            crop, _offset = pad_image(deskewed[ggy0:ggy1, gx0:gx1])

            segments.append(Segment(vectors=seg_vectors, angle=float(skew), image=crop))
            if dbg is not None:
                dbg["segment_bboxes"].append(tight_bbox)
                dbg["grown_segment_bboxes"].append(grown_bbox)
                dbg["assigned_vector_bboxes"].extend(v.bbox for v in seg_vectors)

    return segments
