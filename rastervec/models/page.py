"""Reader's own output -- moved verbatim out of the old flat `models.py`."""
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
