"""The draw specs `commons/renderer/draw.py` builds and draws: frozen, PyMuPDF-free descriptions of
one mark each (text, path, dot, vector, raster image) or a batch of polylines, exactly the numbers PyMuPDF needs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from raster_parser.models.vector import Vector

Point = tuple[float, float]
Rgb = tuple[float, float, float]
Bbox = tuple[float, float, float, float]

# (text, page-space bbox, rotation degrees[, (r, g, b) 0..1]) -- a plain
# text box with no `Text` behind it (ground-truth labels, overlays).
TextBox = Union[tuple[str, Bbox, float], tuple[str, Bbox, float, Rgb]]

# "helv" is PyMuPDF's name for the built-in (base-14) Helvetica font.
FONTNAME = "helv"


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TextSpec:
    text: str
    point: Point
    fontsize: float
    color: Rgb
    morph_fixpoint: Point
    morph_matrix: tuple[float, float, float, float, float, float]
    fontname: str = FONTNAME


@dataclass(frozen=True)
class PathSpec:
    """A bbox (`kind="rect"`), closed polygon or open polyline."""

    points: tuple[Point, ...]
    closed: bool
    color: Rgb | None
    fill: Rgb | None = None
    width: float = 1.0
    dashes: str | None = None
    fill_opacity: float | None = None
    kind: str = "poly"  # "rect" | "poly"


@dataclass(frozen=True)
class DotSpec:
    center: Point
    radius: float
    fill: Rgb


@dataclass(frozen=True, eq=False)
class VectorSpec:
    vector: "Vector"


@dataclass(frozen=True, eq=False)
class ImageSpec:
    rect: Bbox
    width: int
    height: int
    samples: bytes
    n: int  # colour channels: 1 (gray) or 3 (RGB)
    alpha: bool


@dataclass(frozen=True, eq=False)
class StrokeBatchSpec:
    """Many open polylines in one colour/width, round caps + joins; a
    one-point polyline draws as a round dot `width` across."""

    polylines: tuple[tuple[Point, ...], ...]
    color: Rgb
    width: float = 1.0
