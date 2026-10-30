"""Data types of the raster prelude (`commons/raster_text/`): colour layers, tiled-OCR hits, prepared
images and the prelude's debug output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from raster_parser.models.image import Image
from raster_parser.models.text import Text

# pixel (x, y) -> page (x, y)
Mapper = Callable[[tuple[float, float]], tuple[float, float]]
BBox = tuple[float, float, float, float]


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


@dataclass
class TileBox:
    bbox: BBox
    tile: int
    # The detector's own quad (4x2, whole-image px) -- `bbox` is its
    # axis-aligned envelope; PaddleOCR's DB detector returns rotated quads.
    quad: "np.ndarray | None" = None


@dataclass
class OcrHit:
    """One refined detection: `quad` (4x2, image px), `padded_box` (the
    region text removal may erase in), recognized `text` ("" if blank after
    every retry), `confidence`, `angle_deg` (the reading-direction rotation,
    `None` if blank), and `retry_count` (0 = first pass, 1-3 = retry, `None`
    if blank)."""

    quad: np.ndarray
    padded_box: BBox
    text: str = ""
    confidence: float = 0.0
    angle_deg: "float | None" = None
    retry_count: "int | None" = None


@dataclass
class OcrResult:
    tiles: list[BBox] = field(default_factory=list)
    tile_boxes: list[TileBox] = field(default_factory=list)
    merged: list[BBox] = field(default_factory=list)
    hits: list[OcrHit] = field(default_factory=list)


@dataclass
class PreparedImage:
    """One embedded image after the prelude: `enhanced_rgb` (CLAHE'd, text
    erased, HxWx3 uint8, native resolution) and its `layers` (`labels` -1 =
    paper), plus the pixel -> page map and pixels per page point."""

    image: Image
    to_page: Mapper
    px_per_pt: float
    enhanced_rgb: np.ndarray
    layers: ColorLayers


@dataclass
class PreludeRaster:
    stage: str
    label: str
    hexcolor: str
    array: np.ndarray
    image: Image


@dataclass
class PreludeDebug:
    """Page-space OCR geometry, kept raster debug items (`keep_debug`), and
    one `(name, png bytes)` hue/lightness histogram per image."""

    boxes: dict = field(
        default_factory=lambda: {
            "tile": [],
            "tile_detect": [],
            "merged": [],
            "refined_passed": [],
            "refined_failed": [],
        }
    )
    quads: dict = field(
        default_factory=lambda: {
            "tile_detect": [],
            "refined_passed": [],
            "refined_failed": [],
        }
    )
    rasters: list[PreludeRaster] = field(default_factory=list)
    histograms: list[tuple[str, bytes]] = field(default_factory=list)


@dataclass
class RasterPrelude:
    texts: list[Text] = field(default_factory=list)
    images: list[PreparedImage] = field(default_factory=list)
    debug: PreludeDebug = field(default_factory=PreludeDebug)

    def consume_images(self):
        """Yield each `PreparedImage` while dropping the prelude's own
        reference to it -- a backend's loop then frees every image's
        full-resolution arrays (enhanced RGB, int32 labels) as soon as it is
        vectorized, and a prelude kept in `debug_out` (batch
        `prelude_debug_layers`, which never reads `images`) pins none."""
        while self.images:
            yield self.images.pop(0)


@dataclass
class RasterItem:
    stage: str
    label: str
    hexcolor: str
    array: np.ndarray
    image: Image
    shape: tuple[int, int]
