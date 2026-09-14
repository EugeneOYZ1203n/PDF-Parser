"""Phase 1: file reading + native extraction, shared by every pipeline run
regardless of which Phase 2 / Phase 3 backend is selected.

Opens the PDF, hands back the page plus everything native extraction can
get for free without any classification: native text words, raw
(unclassified) vector paths, and raster images (whole-page render +
embedded placements) for Phase 2 to consume.
"""
from __future__ import annotations

from dataclasses import dataclass

from rastervec.commons.models import Image, Page, Text, Vector
from rastervec.P1_Reading_Native.image_extract import extract_images
from rastervec.P1_Reading_Native.native_text import extract_native_text
from rastervec.P1_Reading_Native.reader import Reader
from rastervec.P1_Reading_Native.vector_extract import extract_vectors


@dataclass
class Phase1Result:
    page: Page
    texts: list[Text]
    images: list[Image]
    vectors: list[Vector]


def read_and_extract(pdf_path: str, page_index: int, *, image_dpi: float | None = None) -> Phase1Result:
    """Open `pdf_path`, read page `page_index`, and run every Phase-1-owned
    extractor. Callers that want the `fitz.Page` kept open should hold the
    `Reader` themselves and call the private `_extract` step instead --
    `PipelineResult.open_page()` (core/result.py) is how the rest of the
    pipeline gets it back after this closes the Reader."""
    with Reader(pdf_path) as reader:
        page = reader.get_page(page_index)
        return _extract(page, image_dpi=image_dpi)


def _extract(page: Page, *, image_dpi: float | None = None) -> Phase1Result:
    texts = extract_native_text(page)
    vectors = extract_vectors(page)
    kwargs = {"dpi": image_dpi} if image_dpi is not None else {}
    images = extract_images(page, **kwargs)
    return Phase1Result(page=page, texts=texts, images=images, vectors=vectors)
