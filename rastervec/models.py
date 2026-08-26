"""Shared dataclasses that flow between pipeline stages.

Coordinate convention
----------------------
All geometry fields here are in PDF page **unrotated MediaBox space** — the
same space PyMuPDF's `get_text`/`get_drawings`/`get_image_info`/`annots()`
return. This stays the pipeline's canonical space throughout every stage;
only `Renderer`/reconstruction converts to rotated *display* space at the
very end (matching `page.get_pixmap()`/`page.rect`). Violating this rule
(e.g. applying a page-rotation matrix to already-canonical geometry, or
treating an axis-aligned bbox's width/height as along/normal text extents
for rotated text) is exactly the bug class that caused misaligned overlays
in the `inspector` tool before it was fixed — do not reintroduce it here.

Every dataclass below uses only plain tuples/primitives (no `fitz.Rect`/
`fitz.Quad`/`fitz.Point` in field types) so a stage's *output* is testable
without a live PyMuPDF document. `Page` is the one exception, since Reader
must hand later stages a live `fitz.Page` to extract from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pymupdf as fitz


@dataclass
class PageMeta:
    index: int
    number: int
    mediabox: tuple[float, float, float, float]
    rotation: int
    width: float
    height: float


@dataclass
class Page:
    doc_path: str
    meta: PageMeta
    fitz_page: "fitz.Page" = field(compare=False, repr=False)


@dataclass
class TextWord:
    text: str
    bbox: tuple[float, float, float, float]
    quad: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    angle: float
    direction: tuple[float, float]
    font: str
    font_size: float
    color: int | None
    flags: int
    origin: tuple[float, float] | None
    ascender: float | None
    descender: float | None
    orientation_source: str  # "text-span" | "fallback"
    page_index: int
    seq: int


@dataclass
class TextRun:
    """Line/span-level grouping of TextWords. Not produced by Phase 1 yet."""

    words: list[TextWord]
    bbox: tuple[float, float, float, float]
    quad: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    font: str
    font_size: float
    color: int | None
    page_index: int


# ----------------------------------------------------------------------
# Vector stage (implemented) and Raster/Renderer phases (interface only,
# those fields may still change once those stages are actually implemented).
# ----------------------------------------------------------------------


@dataclass
class VectorPath:
    """One drawing path *item* -- a single line/rect/quad/curve primitive
    from a get_drawings() drawing's "items" list. A drawing with several
    items (e.g. a glyph outline made of several curves) becomes several
    VectorPaths sharing the same `seq`."""

    seq: int  # drawing-level seqno (content-stream draw order)
    item_index: int  # index of this item within its drawing's items list
    kind: str  # "l" | "re" | "qu" | "c" (item primitive type)
    fill_rule: str  # "f" | "s" | "fs" (drawing-level fill/stroke classification)
    points: list[tuple[float, float]]
    bbox: tuple[float, float, float, float]
    stroke_color: tuple[float, ...] | None
    fill_color: tuple[float, ...] | None
    stroke_opacity: float | None
    fill_opacity: float | None
    stroke_width: float | None
    dashes: str | None
    closed: bool | None
    layer: str | None
    page_index: int


@dataclass
class DrawingVector:
    paths: list[VectorPath]
    bbox: tuple[float, float, float, float]
    stroke_color: tuple[float, ...] | None
    fill_color: tuple[float, ...] | None
    stroke_width: float | None
    dashed: bool
    page_index: int


@dataclass
class OcrWord:
    """One detected text box from an OcrBackend (helpers/ocr_backend.py),
    mapped back into PDF page space -- word-level for a backend like
    Tesseract, line/region-level for one like Paddle. Used by
    Renderer.render_reconstructed_page to place/scale each entry into its
    own bbox instead of stretching one string across a whole cluster's
    bbox."""

    text: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass
class TextVectorResult:
    """One OCR reading -- of a whole cluster, or of one of its composing
    groups (see Glossary.md) -- from `RenderOCR.ocr_cluster`. `bbox` is the
    vector-geometry bbox of `paths`; `ocr_bbox` is the backend-detected
    text region for this same reading, mapped back into PDF page space via
    `Renderer.pixel_to_page_bbox` (`None` if the backend detected nothing).
    `words` is one `OcrWord` per individually detected box (`None` when
    nothing was detected, or the input wasn't a vector-path cluster)."""

    paths: list[VectorPath]
    text: str
    confidence: float
    bbox: tuple[float, float, float, float]
    ocr_bbox: tuple[float, float, float, float] | None
    rotation_used: int
    page_index: int
    words: list[OcrWord] | None = None


@dataclass
class ClusterOcrResult:
    """One spatial_regroup cluster's OCR resolution (see pipeline.py's
    _run_ocr_compare): `cluster` is that cluster's own member VectorPaths,
    `resolved` is the direct RenderOCR.ocr_cluster reading over the whole
    cluster -- no fallback tiers, no similarity-group reuse, one OCR call
    per cluster. `ocr_seconds` is that call's wall-clock time, surfaced by
    the debug app as a timing readout."""

    cluster: list[VectorPath]
    resolved: TextVectorResult
    ocr_seconds: float


@dataclass
class RasterImage:
    xref: int
    bbox: tuple[float, float, float, float]
    width_px: int
    height_px: int
    image_bytes: bytes
    page_index: int


@dataclass
class JunctionPoint:
    xy: tuple[float, float]
    directions: list[float]
    page_index: int


@dataclass
class LineVector:
    start: tuple[float, float]
    end: tuple[float, float]
    width: float
    color: tuple[float, ...] | None
    page_index: int


@dataclass
class ReconstructedPage:
    page_index: int
    text_items: list  # TextWord | TextVectorResult
    line_items: list[LineVector]
    remainder_image: bytes | None
