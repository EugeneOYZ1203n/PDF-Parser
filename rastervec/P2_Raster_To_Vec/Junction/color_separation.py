"""Step 1 of Junction's pre-tracing flow: split a raster into ink-color
layers with DBSCAN in HSV space.

DBSCAN never sees individual pixels (O(n^2) neighbourhood queries on a
multi-megapixel scan are infeasible). Instead every pixel is quantized to
`COLOR_QUANT_LEVELS` per RGB channel, the distinct quantized colors are
counted (`np.bincount` -- no sort, no per-pixel int64 inverse), and DBSCAN
clusters those few thousand distinct colors with `sample_weight` = their
pixel count, so a color family needs `DBSCAN_MIN_PIXELS` actual pixels to
seed a cluster. Each pixel's layer is then one lookup-table read.

Feature space is the HSV *cone*, `(S*V*cos H, S*V*sin H, V)`: hue wraps
correctly (2 deg and 358 deg are neighbours), and dark or unsaturated
pixels -- whose hue is meaningless noise -- collapse onto the gray axis, so
black / gray / white separate by value alone with one `eps`.

DBSCAN's density chaining can still fuse ink and paper on a blurred scan
(the antialiased gray ramp between them is dense enough to be all core
points), so any cluster spanning more than `VALUE_SPLIT_RANGE` of Value is
split at a pixel-weighted Otsu threshold on V (`_split_value_ranges`).

DBSCAN-noise colors (antialiased edges, speckle) are assigned to the
nearest cluster centroid in that same space, so every pixel belongs to
some layer. The background layer is the cluster with the most pixels."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rastervec.P2_Raster_To_Vec.Junction.config import (
    COLOR_QUANT_LEVELS,
    DBSCAN_EPS,
    DBSCAN_MIN_PIXELS,
    VALUE_SPLIT_MAX_DEPTH,
    VALUE_SPLIT_RANGE,
)

# Rows processed per band when quantizing / labelling, to keep per-band
# temporaries small on very large images.
_BAND_ROWS = 1024


@dataclass
class ColorLayers:
    """`labels[y, x]` is the layer index of that pixel (uint8, or uint16 past
    255 layers); `centroids_rgb[k]` is layer k's pixel-weighted mean color;
    `counts[k]` its pixel count; `background` the index of the layer with
    the most pixels."""

    labels: np.ndarray
    centroids_rgb: np.ndarray
    counts: np.ndarray
    background: int

    @property
    def n_layers(self) -> int:
        return len(self.counts)

    def ink_layers(self) -> list[int]:
        """Every non-background layer index, largest first."""
        order = np.argsort(-self.counts)
        return [int(k) for k in order if int(k) != self.background and self.counts[k] > 0]


def _bits(levels: int) -> int:
    bits = int(levels).bit_length() - 1
    if levels != 1 << bits or not 1 <= bits <= 8:
        raise ValueError(f"COLOR_QUANT_LEVELS must be a power of two in [2, 256], got {levels}")
    return bits


def _as_rgb(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2).astype(np.uint8)
    return arr[:, :, :3].astype(np.uint8, copy=False)


def _pack(band: np.ndarray, bits: int) -> np.ndarray:
    shift = 8 - bits
    q = (band >> shift).astype(np.uint16)
    return (q[..., 0] << (2 * bits)) | (q[..., 1] << bits) | q[..., 2]


def _unpack_centers(codes: np.ndarray, bits: int) -> np.ndarray:
    """Packed codes -> the RGB center of each quantization bin, float 0..255."""
    mask = (1 << bits) - 1
    shift = 8 - bits
    r = (codes >> (2 * bits)) & mask
    g = (codes >> bits) & mask
    b = codes & mask
    q = np.stack([r, g, b], axis=1).astype(np.float64)
    return q * (1 << shift) + (1 << shift) / 2.0


def cone_coords(rgb: np.ndarray) -> np.ndarray:
    """`(N, 3)` RGB (0..255) -> `(N, 3)` HSV-cone coordinates."""
    from skimage.color import rgb2hsv

    hsv = rgb2hsv(np.clip(rgb, 0, 255).reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    h = hsv[:, 0] * 2.0 * np.pi
    sv = hsv[:, 1] * hsv[:, 2]
    return np.stack([sv * np.cos(h), sv * np.sin(h), hsv[:, 2]], axis=1)


def _dbscan(points: np.ndarray, weights: np.ndarray, eps: float, min_pixels: float) -> np.ndarray:
    from sklearn.cluster import DBSCAN

    return DBSCAN(eps=eps, min_samples=min_pixels).fit(points, sample_weight=weights).labels_


def _weighted_otsu(values: np.ndarray, weights: np.ndarray) -> float:
    """Threshold maximizing between-class variance of weighted `values`."""
    order = np.argsort(values)
    v, wt = values[order], weights[order]
    cw = np.cumsum(wt)
    cm = np.cumsum(wt * v)
    total_w, total_m = cw[-1], cm[-1]
    w0 = cw[:-1]
    w1 = total_w - w0
    valid = (w0 > 0) & (w1 > 0)
    m0 = np.where(valid, cm[:-1] / np.where(w0 > 0, w0, 1), 0)
    m1 = np.where(valid, (total_m - cm[:-1]) / np.where(w1 > 0, w1, 1), 0)
    between = np.where(valid, w0 * w1 * (m0 - m1) ** 2, -1)
    i = int(np.argmax(between))
    return float((v[i] + v[i + 1]) / 2.0)


def _split_value_ranges(labels: np.ndarray, v: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Split every cluster spanning > `VALUE_SPLIT_RANGE` of V in two (see
    module docstring). Noise (-1) is left alone."""
    labels = labels.copy()
    for _depth in range(VALUE_SPLIT_MAX_DEPTH):
        next_id = int(labels.max()) + 1 if labels.size else 0
        changed = False
        for c in sorted(set(labels.tolist()) - {-1}):
            m = labels == c
            if m.sum() < 2 or np.ptp(v[m]) <= VALUE_SPLIT_RANGE:
                continue
            thr = _weighted_otsu(v[m], weights[m])
            idx = np.nonzero(m)[0]
            labels[idx[v[idx] > thr]] = next_id
            next_id += 1
            changed = True
        if not changed:
            break
    return labels


def separate_colors(
    array: np.ndarray, *, levels: int = COLOR_QUANT_LEVELS, eps: float = DBSCAN_EPS,
    min_pixels: float = DBSCAN_MIN_PIXELS,
) -> ColorLayers:
    """Cluster `array` (gray or RGB uint8) into color layers -- see the
    module docstring."""
    rgb = _as_rgb(array)
    h, w = rgb.shape[:2]
    bits = _bits(levels)
    n_codes = 1 << (3 * bits)

    counts_per_code = np.zeros(n_codes, dtype=np.int64)
    for y0 in range(0, h, _BAND_ROWS):
        counts_per_code += np.bincount(
            _pack(rgb[y0:y0 + _BAND_ROWS], bits).ravel(), minlength=n_codes,
        )
    codes = np.nonzero(counts_per_code)[0]
    weights = counts_per_code[codes].astype(np.float64)
    colors = _unpack_centers(codes, bits)
    points = cone_coords(colors)

    raw = _dbscan(points, weights, eps, min_pixels) if len(codes) else np.zeros(0, int)
    if len(codes) and (raw < 0).all():
        # Too few pixels for any core point at this min_pixels (a tiny
        # image): fall back to eps-connected color families.
        raw = _dbscan(points, weights, eps, 1.0)
    if len(codes):
        raw = _split_value_ranges(np.asarray(raw), points[:, 2], weights)

    cluster_ids = sorted(int(c) for c in set(raw.tolist()) if c >= 0)
    remap = {c: i for i, c in enumerate(cluster_ids)}
    code_labels = np.array([remap.get(int(c), -1) for c in raw], dtype=np.int64)
    k = len(cluster_ids)

    centroid_cone = np.zeros((k, 3))
    centroid_rgb = np.zeros((k, 3))
    for i in range(k):
        m = code_labels == i
        wsum = weights[m].sum()
        centroid_cone[i] = (points[m] * weights[m, None]).sum(axis=0) / wsum
        centroid_rgb[i] = (colors[m] * weights[m, None]).sum(axis=0) / wsum

    noise = code_labels < 0
    if noise.any():
        d = np.linalg.norm(points[noise, None, :] - centroid_cone[None, :, :], axis=2)
        code_labels[noise] = np.argmin(d, axis=1)
        # Fold the reassigned noise into the final RGB centroids too.
        for i in range(k):
            m = code_labels == i
            centroid_rgb[i] = (colors[m] * weights[m, None]).sum(axis=0) / weights[m].sum()

    label_dtype = np.uint8 if k <= 255 else np.uint16
    lut = np.zeros(n_codes, dtype=label_dtype)
    lut[codes] = code_labels.astype(label_dtype)
    labels = np.empty((h, w), dtype=label_dtype)
    for y0 in range(0, h, _BAND_ROWS):
        labels[y0:y0 + _BAND_ROWS] = lut[_pack(rgb[y0:y0 + _BAND_ROWS], bits)]

    counts = np.array([weights[code_labels == i].sum() for i in range(k)], dtype=np.int64)
    background = int(np.argmax(counts)) if k else 0
    return ColorLayers(
        labels=labels, centroids_rgb=np.clip(np.round(centroid_rgb), 0, 255).astype(np.uint8),
        counts=counts, background=background,
    )


def recolor(layers: ColorLayers) -> np.ndarray:
    """Every pixel painted with its layer's centroid color -- the
    `color_separation / clusters` debug image."""
    return layers.centroids_rgb[layers.labels]
