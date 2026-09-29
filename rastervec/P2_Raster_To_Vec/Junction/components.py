"""Step 9 of Junction's pre-tracing flow: split one color layer into
spatially separate components and hand them to the tracer one at a time.

Junction runs at full native resolution (no downscale), and its skeleton
graph / regularizer are pure Python, so tracing a whole multi-megapixel
layer in one call is slow and holds every intermediate mask for the whole
image at once. Instead each layer's ink is grouped into components -- ink
pixels closer than `COMPONENT_TOLERANCE_PT` (converted to px by the caller)
join, via one rectangular dilation + connected-component labelling -- and
`iter_layer_components` is a generator: it yields one tight crop at a time,
so only that crop and its own tracing intermediates are alive while it's
traced. Components at least the tolerance apart never share an endpoint
or junction, so tracing them separately loses nothing the regularizer
could have joined (its `snap_px` is far smaller).

The yielded crop is the *enhanced* grayscale where the pixel belongs to
this layer (dilated 1 px, so antialiased stroke edges stay part of the
stroke) and this component, and white (255) everywhere else -- the tracer
only ever sees its own layer's ink."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

# Margin (px) kept around a component's own ink extent in its crop, so
# binarization/skeletonization never touches the crop edge.
_CROP_MARGIN_PX = 3


@dataclass
class Component:
    x0: int
    y0: int
    gray: np.ndarray  # uint8 crop, 255 outside this component's ink
    n_pixels: int


def iter_layer_components(
    labels: np.ndarray, enhanced: np.ndarray, layer: int, tol_px: float,
) -> Iterator[Component]:
    """Yield `Component`s of `layer`, largest first -- see module docstring."""
    h, w = labels.shape[:2]
    layer_mask = (labels == layer).astype(np.uint8)
    if not layer_mask.any():
        return
    fringe = cv2.dilate(layer_mask, np.ones((3, 3), np.uint8))
    del layer_mask
    r = max(0, int(np.ceil(tol_px / 2.0)))
    grown = cv2.dilate(fringe, np.ones((2 * r + 1, 2 * r + 1), np.uint8)) if r else fringe
    n, cc, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=8, ltype=cv2.CV_32S)
    del grown

    order = sorted(range(1, n), key=lambda i: -int(stats[i, cv2.CC_STAT_AREA]))
    for i in order:
        bx, by, bw, bh = (int(v) for v in stats[i, :4])
        sub = (cc[by:by + bh, bx:bx + bw] == i) & (fringe[by:by + bh, bx:bx + bw] > 0)
        ys, xs = np.nonzero(sub)
        if xs.size == 0:
            continue
        x0 = max(0, bx + int(xs.min()) - _CROP_MARGIN_PX)
        y0 = max(0, by + int(ys.min()) - _CROP_MARGIN_PX)
        x1 = min(w, bx + int(xs.max()) + 1 + _CROP_MARGIN_PX)
        y1 = min(h, by + int(ys.max()) + 1 + _CROP_MARGIN_PX)
        mask = (cc[y0:y1, x0:x1] == i) & (fringe[y0:y1, x0:x1] > 0)
        gray = np.where(mask, enhanced[y0:y1, x0:x1], np.uint8(255)).astype(np.uint8)
        yield Component(x0=x0, y0=y0, gray=gray, n_pixels=int(xs.size))
