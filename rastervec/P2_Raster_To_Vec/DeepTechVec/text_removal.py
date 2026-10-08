"""Steps 6-7 of DeepTechVec's pre-vectorizing flow (copied from DeepVectoriser, originally Junction): identify each recognized text's
ink color and erase exactly that ink from the image.

The ink layer of a hit is the most frequent non-background color layer
inside its refined quad (background = the image-wide dominant layer, see
`color_separation.py`). Erasing removes only pixels of *that* layer inside
the quad's recognition region (the quad expanded by pad 2 --
`CROP_EXPAND_FRACTION` about its centroid plus `CROP_BORDER_PX`), dilated
`ERASE_DILATE_PX` to catch antialiased fringe, and paints them with the
background centroid color -- so a differently-colored line crossing a label
survives. Only hits whose recognition returned text are erased: a blank
detection is more likely a hatch or symbol PaddleOCR mis-fired on, which
the vectorizer should still trace. `rgb` and `labels` are mutated in place (no
extra full-size copy)."""
from __future__ import annotations

import cv2
import numpy as np

from rastervec.P2_Raster_To_Vec.DeepTechVec.color_separation import ColorLayers
from rastervec.P2_Raster_To_Vec.DeepTechVec.config import (
    CROP_BORDER_PX,
    CROP_EXPAND_FRACTION,
    ERASE_DILATE_PX,
)


def _region(quad: np.ndarray, shape: tuple[int, int], expand: bool):
    """`(x0, y0, local polygon mask)` for `quad` (optionally pad-2 expanded),
    clipped to an image of `shape`; `None` if it falls outside."""
    q = np.asarray(quad, dtype=np.float64)
    border = 0
    if expand:
        c = q.mean(axis=0)
        q = c + (q - c) * (1.0 + CROP_EXPAND_FRACTION)
        border = CROP_BORDER_PX
    h, w = shape
    x0 = max(0, int(np.floor(q[:, 0].min())) - border - ERASE_DILATE_PX)
    y0 = max(0, int(np.floor(q[:, 1].min())) - border - ERASE_DILATE_PX)
    x1 = min(w, int(np.ceil(q[:, 0].max())) + border + ERASE_DILATE_PX + 1)
    y1 = min(h, int(np.ceil(q[:, 1].max())) + border + ERASE_DILATE_PX + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(q - [x0, y0]).astype(np.int32)], 1)
    if border:
        mask = cv2.dilate(mask, np.ones((2 * border + 1, 2 * border + 1), np.uint8))
    return x0, y0, mask.astype(bool)


def ink_label(labels: np.ndarray, quad: np.ndarray, background: int) -> "int | None":
    """Most frequent non-background layer inside `quad`, or `None`."""
    region = _region(quad, labels.shape[:2], expand=False)
    if region is None:
        return None
    x0, y0, mask = region
    local = labels[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]][mask]
    local = local[local != background]
    if local.size == 0:
        return None
    values, counts = np.unique(local, return_counts=True)
    return int(values[int(np.argmax(counts))])


def erase_text(rgb: np.ndarray, layers: ColorLayers, hits) -> list["int | None"]:
    """Erase every non-blank hit's ink (see module docstring). Returns each
    hit's ink layer (`None` for a blank hit or one with no ink found), in
    `hits` order -- used for `Text.color`."""
    bg = layers.background
    bg_rgb = layers.centroids_rgb[bg] if layers.n_layers else np.array([255, 255, 255], np.uint8)
    kernel = np.ones((2 * ERASE_DILATE_PX + 1, 2 * ERASE_DILATE_PX + 1), np.uint8)
    inks: list["int | None"] = []
    for hit in hits:
        ink = ink_label(layers.labels, hit.quad, bg) if hit.text else None
        inks.append(ink)
        if ink is None:
            continue
        region = _region(hit.quad, layers.labels.shape[:2], expand=True)
        if region is None:
            continue
        x0, y0, mask = region
        ys = slice(y0, y0 + mask.shape[0])
        xs = slice(x0, x0 + mask.shape[1])
        erase = (layers.labels[ys, xs] == ink) & mask
        if ERASE_DILATE_PX > 0:
            erase = cv2.dilate(erase.astype(np.uint8), kernel).astype(bool)
        rgb[ys, xs][erase] = bg_rgb
        layers.labels[ys, xs][erase] = bg
    return inks
