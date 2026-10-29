"""Intermediate results of the v1_raster_to_vec backend, one per step."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from raster_parser.models.text import Text
from raster_parser.models.vector import Vector


@dataclass
class OcrParseResult:
    drawing: np.ndarray  # uint8 grayscale: text erased -> white, lines/drawing kept
    text: np.ndarray  # uint8 mask, text ink = 255 on 0 background
    text_gray: np.ndarray  # uint8 grayscale: original ink where text, white elsewhere
    lines: np.ndarray  # uint8 mask, line ink = 255 on 0 background
    binary: np.ndarray  # uint8 mask, all ink = 255 on 0 background
    boxes: list  # OCR boxes [(x0, y0, x1, y1), ...]

    @property
    def n_boxes(self) -> int:
        return len(self.boxes)

    @property
    def text_ink_px(self) -> int:
        return int((self.text == 255).sum())


@dataclass
class RegionOcrResult:
    texts: list[Text]  # recognized words, bbox in PDF points
    kept_mask: np.ndarray  # uint8 mask: ink of boxes that passed the confidence gate
    n_boxes: int  # detection boxes fed in
    n_text: int  # boxes that recognized as confident text


@dataclass
class GlyphCc:
    label: int
    x0: int
    y0: int
    x1: int
    y1: int
    area: int
    cx: float
    cy: float

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0


@dataclass
class ColourGroup:
    """One colour cluster (centroid RGB + pixel count; optional black/white kind)."""

    label: int
    rgb: tuple[int, int, int]
    n_pixels: int
    kind: str | None = None  # "black" | "white" | None

    @property
    def tag(self) -> str:
        """Stable folder/file tag: ``c{label:02d}_{rrggbb}``."""
        r, g, b = self.rgb
        return f"c{self.label:02d}_{r:02x}{g:02x}{b:02x}"


@dataclass
class ColourFillTraceResult:
    """What ``colour_fill_autotrace`` found: the colour groups and the stamped vectors of all of
    them, in paint order. The QA artefacts are files under ``{stem}_colours/``."""

    groups: list[ColourGroup]
    combined_vectors: list[Vector]
