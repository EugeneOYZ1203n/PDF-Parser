"""native_read: file reading + native extraction, shared by every pipeline run
regardless of which raster_vectorize / vector_classify backend is selected.

Opens the PDF, hands back the page plus everything native extraction can
get for free without any classification: native text words, raw
(unclassified) vector paths, and embedded raster images for raster_vectorize to
consume.
"""

from __future__ import annotations


from raster_parser.models.page import Page
from raster_parser.models.native_read import NativeReadResult
from raster_parser.phase_native_read.image_extract import extract_images
from raster_parser.phase_native_read.native_text import extract_native_text
from raster_parser.phase_native_read.reader import Reader
from raster_parser.phase_native_read.vector_extract import extract_vectors


def read_and_extract(pdf_path: str, page_index: int) -> NativeReadResult:
    """Open `pdf_path`, read page `page_index`, and run every native_read-owned
    extractor. The PDF is closed again on return, so `page.pymupdf_page` is
    unusable afterwards; reopen it with `Reader(page.doc_path)` if needed."""
    with Reader(pdf_path) as reader:
        page = reader.get_page(page_index)
        return _extract(page)


def _extract(page: Page) -> NativeReadResult:
    texts = extract_native_text(page)
    vectors = extract_vectors(page)
    images = extract_images(page)
    return NativeReadResult(page=page, texts=texts, images=images, vectors=vectors)
