"""Simple colour layers for the shared P2 raster prelude (replaces each
backend's former DBSCAN `color_separation.py`). Runs on the CLAHE'd LAB
image (`enhance.clahe_lab`):

1. ink -- `L` at or below its Otsu threshold (only when the two Otsu classes are
   at least `OTSU_MIN_SEPARATION` apart, so a blank or near-uniform image
   doesn't get half its paper called ink), **or** chroma
   `hypot(a - 128, b - 128)` above `INK_CHROMA_MIN` (keeps pale coloured
   lines whose lightness alone reads as paper).
2. ink at or below `ACHROMATIC_CHROMA_MAX` chroma -> one achromatic
   (black / gray) layer.
3. the rest: a chroma-weighted LAB hue-angle histogram (`HUE_HIST_BINS`
   bins over 360 deg), circularly Gaussian-smoothed; every local maximum
   holding at least `HUE_PEAK_MIN_FRAC` of the highest is a peak, and each
   hue bin goes to its circularly nearest peak -- one layer per peak, a
   contiguous hue arc each.
4. a layer under `LAYER_MIN_PIXELS` is folded into its nearest layer
   (nearest hue peak; the achromatic layer when no other hue layer exists;
   a small achromatic layer into the largest hue layer), smallest first,
   until every layer is big enough or only one is left.

Layers are numbered by pixel count, largest first; `labels` is -1 on
paper. A layer's colour is the median *original* (pre-CLAHE) RGB of its
pixels, so output vectors keep the drawing's real colours."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from rastervec.P2_Raster_To_Vec.RasterText.config import (
    ACHROMATIC_CHROMA_MAX,
    HUE_HIST_BINS,
    HUE_PEAK_MIN_FRAC,
    HUE_SMOOTH_SIGMA_DEG,
    INK_CHROMA_MIN,
    LAYER_MIN_PIXELS,
)

PAPER = -1
# Otsu on L only counts when the dark class's mean is at least this far
# below the light class's (8-bit L units).
OTSU_MIN_SEPARATION = 40.0
# Pixels sampled per layer for its median colour.
_COLOR_SAMPLE = 100_000


@dataclass
class ColorLayers:
    """`labels` (HxW int32, `PAPER` = -1 or a layer index), and per layer:
    `colors_rgb` (n x 3 uint8), `kinds` ("achromatic" | "hue"),
    `hue_ranges` (`(start_deg, end_deg)` arc, counter-clockwise, `None` for
    achromatic), `peaks_deg` (`None` for achromatic), `pixel_counts`.
    Diagnostics: `otsu_l` (the L threshold, `None` when unused), `hist` /
    `hist_smooth` (the chromatic hue histogram, `HUE_HIST_BINS` long)."""

    labels: np.ndarray
    colors_rgb: np.ndarray
    kinds: list[str] = field(default_factory=list)
    hue_ranges: list = field(default_factory=list)
    peaks_deg: list = field(default_factory=list)
    pixel_counts: list[int] = field(default_factory=list)
    otsu_l: "float | None" = None
    hist: "np.ndarray | None" = None
    hist_smooth: "np.ndarray | None" = None

    @property
    def n_layers(self) -> int:
        return len(self.kinds)

    def ink_layers(self) -> range:
        return range(self.n_layers)


def chroma_of(lab: np.ndarray) -> np.ndarray:
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    return np.hypot(a, b)


def hue_deg_of(lab: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """LAB hue angle (0..360) of the pixels under `mask`, flat."""
    a = lab[:, :, 1][mask].astype(np.float32) - 128.0
    b = lab[:, :, 2][mask].astype(np.float32) - 128.0
    return np.degrees(np.arctan2(b, a)) % 360.0


def ink_mask(lab: np.ndarray, chroma: "np.ndarray | None" = None) -> tuple[np.ndarray, "float | None"]:
    """Step 1: `(ink bool HxW, Otsu L threshold or None)`."""
    L = np.ascontiguousarray(lab[:, :, 0])
    chroma = chroma_of(lab) if chroma is None else chroma
    ink = chroma > INK_CHROMA_MIN
    t, _ = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = L <= t  # OpenCV Otsu: > t is the light class
    if dark.any() and not dark.all():
        separation = float(L[~dark].mean()) - float(L[dark].mean())
        if separation >= OTSU_MIN_SEPARATION:
            return ink | dark, float(t)
    return ink, None


def _smooth_circular(hist: np.ndarray, sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return hist.astype(np.float64)
    r = int(np.ceil(3 * sigma_bins))
    x = np.arange(-r, r + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    n = len(hist)
    padded = np.concatenate([hist[-r:], hist, hist[:r]]) if r < n else np.tile(hist, 3)
    out = np.convolve(padded, kernel, mode="same")
    return out[r:r + n] if r < n else out[n:2 * n]


def hue_peaks(hist_smooth: np.ndarray, min_frac: float = HUE_PEAK_MIN_FRAC) -> list[int]:
    """Bins that are circular local maxima (> left, >= right) of at least
    `min_frac` of the highest bin."""
    top = float(hist_smooth.max()) if hist_smooth.size else 0.0
    if top <= 0:
        return []
    left = np.roll(hist_smooth, 1)
    right = np.roll(hist_smooth, -1)
    is_peak = (hist_smooth > left) & (hist_smooth >= right) & (hist_smooth >= min_frac * top)
    peaks = [int(i) for i in np.nonzero(is_peak)[0]]
    if not peaks:  # flat plateau: one peak at its maximum
        peaks = [int(np.argmax(hist_smooth))]
    return peaks


def _circ_dist(a, b, period: float) -> np.ndarray:
    d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) % period
    return np.minimum(d, period - d)


def _nearest_peak_lut(peaks: list[int], n_bins: int) -> np.ndarray:
    bins = np.arange(n_bins)
    dists = np.stack([_circ_dist(bins, p, n_bins) for p in peaks])
    return np.argmin(dists, axis=0)


def _arc(lut: np.ndarray, k: int, n_bins: int) -> tuple[float, float]:
    """The contiguous circular arc of bins owned by peak `k`, in degrees."""
    own = lut == k
    if own.all():
        return (0.0, 360.0)
    starts = np.nonzero(own & ~np.roll(own, 1))[0]
    ends = np.nonzero(own & ~np.roll(own, -1))[0]
    start = int(starts[0])
    end = int(ends[ends >= start][0]) if (ends >= start).any() else int(ends[0])
    deg = 360.0 / n_bins
    return (start * deg, ((end + 1) * deg) % 360.0 or 360.0)


def split_layers(lab: np.ndarray, rgb: np.ndarray) -> ColorLayers:
    """Steps 1-4 (module docstring). `lab` = the CLAHE'd LAB image, `rgb` =
    the original image (layer colours)."""
    h, w = lab.shape[:2]
    chroma = chroma_of(lab)
    ink, otsu_l = ink_mask(lab, chroma)
    labels = np.full((h, w), PAPER, dtype=np.int32)

    # provisional layers: id 0 = achromatic, 1.. = hue peaks
    kinds: list[str] = []
    peaks_deg: list = []
    hue_ranges: list = []
    achrom = ink & (chroma <= ACHROMATIC_CHROMA_MAX)
    chromatic = ink & ~achrom
    hist = np.zeros(HUE_HIST_BINS, dtype=np.float64)
    hist_smooth = hist
    if achrom.any():
        labels[achrom] = 0
    kinds.append("achromatic")
    peaks_deg.append(None)
    hue_ranges.append(None)
    if chromatic.any():
        hue = hue_deg_of(lab, chromatic)
        bins = np.minimum((hue * HUE_HIST_BINS / 360.0).astype(np.int64), HUE_HIST_BINS - 1)
        hist = np.bincount(bins, weights=chroma[chromatic], minlength=HUE_HIST_BINS).astype(np.float64)
        hist_smooth = _smooth_circular(hist, HUE_SMOOTH_SIGMA_DEG * HUE_HIST_BINS / 360.0)
        peaks = hue_peaks(hist_smooth)
        lut = _nearest_peak_lut(peaks, HUE_HIST_BINS)
        labels[chromatic] = 1 + lut[bins]
        for k, p in enumerate(peaks):
            kinds.append("hue")
            peaks_deg.append((p + 0.5) * 360.0 / HUE_HIST_BINS)
            hue_ranges.append(_arc(lut, k, HUE_HIST_BINS))

    counts = np.bincount(labels[labels >= 0].ravel(), minlength=len(kinds)).astype(np.int64)
    alive = [i for i in range(len(kinds)) if counts[i] > 0]
    _fold_small(labels, counts, alive, kinds, peaks_deg)
    return _finalize(labels, alive, kinds, peaks_deg, hue_ranges, rgb, otsu_l, hist, hist_smooth)


def _fold_small(labels: np.ndarray, counts: np.ndarray, alive: list[int], kinds: list[str],
                peaks_deg: list) -> None:
    """Step 4, in place on `labels` / `counts` / `alive`."""
    while len(alive) > 1:
        small = [i for i in alive if counts[i] < LAYER_MIN_PIXELS]
        if not small:
            return
        i = min(small, key=lambda j: counts[j])
        others = [j for j in alive if j != i]
        hue_others = [j for j in others if kinds[j] == "hue"]
        if kinds[i] == "hue" and hue_others:
            target = min(hue_others, key=lambda j: float(_circ_dist(peaks_deg[i], peaks_deg[j], 360.0)))
        elif kinds[i] == "hue":
            target = others[0]  # only the achromatic layer is left
        else:
            target = max(others, key=lambda j: counts[j])
        labels[labels == i] = target
        counts[target] += counts[i]
        counts[i] = 0
        alive.remove(i)


def _finalize(labels, alive, kinds, peaks_deg, hue_ranges, rgb, otsu_l, hist, hist_smooth) -> ColorLayers:
    counts = {i: int(np.count_nonzero(labels == i)) for i in alive}
    order = sorted(alive, key=lambda i: -counts[i])
    remap = np.full(max(len(kinds), 1) + 1, PAPER, dtype=np.int32)  # index -1 -> last slot = PAPER
    for new, old in enumerate(order):
        remap[old] = new
    out = remap[labels]
    flat_rgb = np.asarray(rgb)[:, :, :3].reshape(-1, 3)
    flat_lab = out.ravel()
    colors = []
    for new in range(len(order)):
        idx = np.flatnonzero(flat_lab == new)
        if idx.size > _COLOR_SAMPLE:
            idx = idx[:: idx.size // _COLOR_SAMPLE + 1]
        colors.append(np.median(flat_rgb[idx], axis=0) if idx.size else np.zeros(3))
    return ColorLayers(
        labels=out,
        colors_rgb=np.asarray(colors, dtype=np.uint8).reshape(-1, 3),
        kinds=[kinds[i] for i in order],
        hue_ranges=[hue_ranges[i] for i in order],
        peaks_deg=[peaks_deg[i] for i in order],
        pixel_counts=[counts[i] for i in order],
        otsu_l=otsu_l, hist=hist, hist_smooth=hist_smooth,
    )


def recolor(layers: ColorLayers) -> np.ndarray:
    """RGBA debug image: every ink pixel in its layer's colour, paper
    transparent."""
    h, w = layers.labels.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    ink = layers.labels >= 0
    if layers.n_layers:
        out[..., :3][ink] = layers.colors_rgb[layers.labels[ink]]
    out[..., 3][ink] = 255
    return out
