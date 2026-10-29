"""Reader's own output: a page and its geometry snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pymupdf


@dataclass
class PageMeta:
    index: int
    number: int
    mediabox: tuple[float, float, float, float]
    rotation: int
    # `width`/`height` are the *CropBox* size (unrotated): PyMuPDF reports
    # every extracted coordinate relative to the CropBox's top-left corner,
    # so that -- not the MediaBox -- is the frame text/vectors live in and
    # every rendered page must match.
    width: float
    height: float
    cropbox: tuple[float, float, float, float] | None = None


@dataclass
class Page:
    doc_path: str
    meta: PageMeta
    pymupdf_page: "pymupdf.Page" = field(compare=False, repr=False)
