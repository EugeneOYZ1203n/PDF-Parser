"""Padding + dynamic-dpi rendering -- the pixel-prep primitives shared by
every P3 backend's own OCR pipeline before a cluster reaches PaddleOCR.

`pad_image_uniform` is the one padding algorithm used everywhere in the
pipeline now (a second, axis-independent variant used to live in
`VectorClassification/radon.py`; it's retired, not ported here -- every
padding call site uses this uniform formula instead). `dpi_for_cluster` /
`render_cluster_with_dynamic_dpi` compose with `png.render_vector_cluster`
to render a cluster at a dpi bumped up (never down) so a small cluster
isn't handed to OCR at a few dozen px.

Rotation-related helpers (perspective rotate-crop, quad-angle estimation)
and BGR normalization are deliberately **not** here -- they stay duplicated
in whichever of `VectorClassification`/`LegacyRecreation`'s own
`paddle_engine.py` actually needs them, since this module is scoped to
padding and rendering
only.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from rastervec.commons.helpers.geometry import PDF_POINTS_PER_INCH, union_bbox
from rastervec.commons.models import Vector
from rastervec.commons.renderer.png import render_vector_cluster


def pad_image_uniform(
    img: np.ndarray, fraction: float = 0.1,
) -> "tuple[np.ndarray, tuple[int, int]]":
    """Surround `img` (2D grayscale or 3D H,W,C -- the same ndim-generic
    `np.pad` call handles either) with a white border of
    `fraction * max(width, height)` px on every side -- both axes padded
    off the *larger* dimension, so a wide/short or narrow/tall crop gets
    equal breathing room on all four sides instead of a border that scales
    independently per axis. `pad_x_px`/`pad_y_px` are what a caller must
    subtract to get back into the *unpadded* image's pixel space -- which
    is what `png.pixel_to_page_bbox` inverts. A zero-area image is returned
    unchanged with a `(0, 0)` offset."""
    if img.size == 0:
        return img, (0, 0)
    pad = int(round(max(img.shape[0], img.shape[1]) * fraction))
    pad_width = ((pad, pad), (pad, pad)) + ((0, 0),) * (img.ndim - 2)
    padded = np.pad(img, pad_width, mode="constant", constant_values=255)
    return padded, (pad, pad)


def dpi_for_cluster(
    vectors: list[Vector],
    dpi: int,
    padding: float,
    min_render_side_px: float,
    max_render_dpi: float,
) -> int:
    """Dynamic dpi-bump rule (never down, capped at `max_render_dpi`) so a
    small cluster's render still reaches at least `min_render_side_px` on
    its shorter side -- PaddleOCR's detector/recognizer read a tiny crop
    poorly. Same algorithm every P3 backend duplicated with its own
    module-scoped thresholds; those thresholds are now explicit params."""
    x0, y0, x1, y1 = union_bbox([v.bbox for v in vectors])
    min_side_pt = min(x1 - x0, y1 - y0) + 2 * padding
    if min_side_pt <= 0:
        return dpi
    needed_dpi = math.ceil(min_render_side_px * PDF_POINTS_PER_INCH / min_side_pt)
    return min(max(dpi, needed_dpi), max_render_dpi)


def render_cluster_with_dynamic_dpi(
    vectors: list[Vector],
    base_dpi: int,
    min_render_side_px: float,
    max_render_dpi: float,
    padding: float = 0.0,
) -> "tuple[Image.Image, int]":
    """Composes `dpi_for_cluster` + `png.render_vector_cluster` -- the
    "compute the dpi to render at, then render" glue every backend
    duplicated at its own cluster-render call site. Returns
    `(image, dpi_used)`."""
    dpi_used = dpi_for_cluster(vectors, base_dpi, padding, min_render_side_px, max_render_dpi)
    return render_vector_cluster(vectors, dpi_used, padding), dpi_used
