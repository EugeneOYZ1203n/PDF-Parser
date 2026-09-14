"""Vector-geometry-only rotation refinement -- no rasterization anywhere in
this file.

Ported from `rastervec/notebooks/radon_vector_deskew_lab.ipynb`'s pure-vector
Radon-style sweep: `project(vectors, theta_deg)` counts, for a set of
horizontal rays swept across the cluster's own rotated bounding box, how many
`Vector`s each ray intersects -- a 1-D profile whose sharpness (scored by a
swappable value function, `VALUES`) peaks when `theta_deg` aligns the text
baseline to horizontal. `sweep_rotation` is the pipeline's own entry point:
given PaddleOCR's own coarse per-detection rotation estimate (from its quad +
`use_angle_cls`), it refines that estimate by scoring only a narrow range
around it -- there is no 90-degree flip ambiguity left to resolve here
(PaddleOCR's quad already localizes the line's rough orientation), unlike the
notebook's blind full 90-degree sweep with perpendicular pairing.

This step produces an angle and, for debugging, page-space line-gap
positions -- it does **not** split anything into words/characters (that
concept no longer exists in this pipeline: a whole PaddleOCR detection's
assigned vectors become one rotated crop, built by
`OCR/Paddle_OCR/ocr_backend.py::crop_rotated_detection`, which is where all
pixel/rasterization/padding work now lives instead).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rastervec.config import (
    VECTOR_RADON_LINE_SPACING_PT,
    VECTOR_RADON_MAX_LINES,
    VECTOR_RADON_SWEEP_RANGE_DEG,
    VECTOR_RADON_SWEEP_STEP_DEG,
)
from rastervec.commons.helpers.geometry import item_points, union_bbox
from rastervec.commons.models import Vector

# --------------------------------------------------------------------------
# Pure vector-geometry rotation + projection (no pixels)
# --------------------------------------------------------------------------
_CURVE_SAMPLES = 24


def rotate_pts(pts, theta_deg: float, centre) -> np.ndarray:
    """Rotate `(N, 2)` points by `-theta_deg` about `centre` (Radon-ray
    frame -- the same sign convention the sweep below expects)."""
    t = np.deg2rad(-theta_deg)
    c, s = np.cos(t), np.sin(t)
    m = np.array([[c, -s], [s, c]])
    return (np.asarray(pts, dtype=float) - centre) @ m.T + centre


def _bezier(pts, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in pts]
    u = np.linspace(0.0, 1.0, n + 1)[:, None]
    b = ((1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1
         + 3 * (1 - u) * u ** 2 * p2 + u ** 3 * p3)
    return list(zip(b[:-1], b[1:]))


def item_subsegments(item, curve_samples: int | None = None) -> list[tuple]:
    """Straight `(a, b)` pieces approximating one `Vector.items` entry:
    `'l'` -> 1 segment, `'c'` -> `curve_samples` Bernstein chords, `'re'`/
    `'qu'` -> the 4 closed border edges."""
    n = _CURVE_SAMPLES if curve_samples is None else curve_samples
    kind = item[0]
    if kind == "l":
        return [(item[1], item[2])]
    if kind == "c":
        return _bezier(item_points(item), n)
    if kind == "qu":
        corners = [tuple(p) for p in item_points(item)]
    elif kind == "re":
        x0, y0, x1, y1 = tuple(item[1])
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    else:
        return []
    return list(zip(corners, corners[1:] + corners[:1]))


def cluster_centre(vectors: list[Vector]) -> np.ndarray:
    x0, y0, x1, y1 = union_bbox([v.bbox for v in vectors])
    return np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])


def rotated_segments(
    vectors: list[Vector], theta_deg: float,
) -> "tuple[list[np.ndarray], tuple[float, float, float, float]]":
    """Rotate the whole cluster by `-theta_deg` about its own centre.
    Returns `(per_vector, (X0, X1, Y0, Y1))`: `per_vector[i]` is an
    `(M_i, 2, 2)` array of that `Vector`'s sub-segment endpoints in the
    rotated frame, and `(X0, X1, Y0, Y1)` is the cluster's rotated bbox
    (over every vector's own points, not the pre-rotation bbox)."""
    centre = cluster_centre(vectors)
    per_vector: list[np.ndarray] = []
    for v in vectors:
        segs = [ab for it in v.items for ab in item_subsegments(it)]
        if not segs:
            per_vector.append(np.empty((0, 2, 2)))
            continue
        flat = np.asarray(segs, dtype=float).reshape(-1, 2)
        rotated = rotate_pts(flat, theta_deg, centre).reshape(-1, 2, 2)
        per_vector.append(rotated)

    all_pts = np.concatenate(
        [pv.reshape(-1, 2) for pv in per_vector if pv.size], axis=0,
    ) if any(pv.size for pv in per_vector) else np.zeros((0, 2))
    if all_pts.size:
        (X0, Y0), (X1, Y1) = all_pts.min(axis=0), all_pts.max(axis=0)
    else:
        X0 = X1 = Y0 = Y1 = 0.0
    return per_vector, (float(X0), float(X1), float(Y0), float(Y1))


def n_lines_for(
    height: float, *, line_spacing: float = VECTOR_RADON_LINE_SPACING_PT,
    max_lines: int = VECTOR_RADON_MAX_LINES,
) -> int:
    """Division-line count for a rotated-bbox `height` -- scales with the
    bbox, clamped to `[2, max_lines]`."""
    return int(np.clip(round(height / line_spacing), 2, max_lines))


def project(vectors: list[Vector], theta_deg: float) -> "tuple[np.ndarray, np.ndarray]":
    """`(line_y[n], profile[n])` -- `n` horizontal rays `y=y_k` swept across
    the cluster rotated by `theta_deg`, `n` scaling with the rotated-bbox
    height; `profile[k]` = how many `Vector`s ray `k` intersects."""
    per_vector, (_X0, _X1, Y0, Y1) = rotated_segments(vectors, theta_deg)
    height = Y1 - Y0
    n = n_lines_for(height)
    k = np.arange(1, n + 1)
    line_y = Y0 + k * height / (n + 1)
    profile = np.zeros(n)
    if height < 1e-9:
        return line_y, profile
    for pv in per_vector:
        if not pv.size:
            continue
        ylo = pv[:, :, 1].min(axis=1)[:, None]
        yhi = pv[:, :, 1].max(axis=1)[:, None]
        hit = ((ylo <= line_y[None, :]) & (line_y[None, :] <= yhi)).any(axis=0)
        profile += hit
    return line_y, profile


# --------------------------------------------------------------------------
# Value functions: `fn(a, b) -> (score, a_is_correct)`, matching the
# notebook's pair-scoring seam. Only used here for a plain argopt over a
# narrow single-sided sweep (see `sweep_rotation`), so `a_is_correct` is
# unused in production -- kept for signature parity with the notebook so a
# value function can be dropped in either place unmodified.
# --------------------------------------------------------------------------
def val_postl_sq(a, b) -> "tuple[float, bool]":
    """Sum-of-squares (Postl criterion) -- the sharper/more concentrated
    profile wins. Matches the criterion the project's pixel-based deskew
    historically used, kept as the production default for consistency."""
    sa = float(np.sum(np.asarray(a, dtype=float) ** 2))
    sb = float(np.sum(np.asarray(b, dtype=float) ** 2))
    return sa, sa >= sb


def val_variance(a, b) -> "tuple[float, bool]":
    sa = float(np.var(np.asarray(a, dtype=float)))
    sb = float(np.var(np.asarray(b, dtype=float)))
    return sa, sa >= sb


VALUES = {"postl_sq": val_postl_sq, "variance": val_variance}
OPT = {"postl_sq": "max", "variance": "max"}


# --------------------------------------------------------------------------
# Production entry point
# --------------------------------------------------------------------------
@dataclass
class RotationSweepResult:
    theta_deg: float


def sweep_rotation(
    vectors: list[Vector],
    center_theta_deg: float,
    *,
    range_deg: float = VECTOR_RADON_SWEEP_RANGE_DEG,
    step_deg: float = VECTOR_RADON_SWEEP_STEP_DEG,
    value_fn=val_postl_sq,
) -> RotationSweepResult:
    """**Currently a no-op passthrough** -- returns `center_theta_deg`
    unchanged, without running the coarse/fine angle sweep below. Disabled
    at the user's request pending a future revisit of rotation correction;
    `range_deg`/`step_deg`/`value_fn` and the sweep logic (`project`,
    `val_postl_sq`/`val_variance`) are kept as-is, unused, so re-enabling is
    a one-line change back to the sweep loop this docstring used to
    describe."""
    return RotationSweepResult(theta_deg=center_theta_deg)
