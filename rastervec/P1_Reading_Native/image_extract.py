"""Raster images for a page -- Phase 1's third output, consumed by Phase 2.

Only real embedded raster image placements (via
`page.get_image_info(xrefs=True)`, the same API
`Evaluation/Labelling/raster_label.py::embedded_images_for_page` and
`Evaluation/inspector/pdf_model.py::extract_image_items` already use). There
is deliberately no whole-page render: Phase 2 turns *raster content* into
vectors, and re-rasterising a page's own native text/vector drawing would
only make a Phase 2 backend trace content Phase 1/3 already own exactly
(and, on a benchmark `rasterised.pdf` page -- one full-page embedded image,
see `scripts/rasterize_pdf.py` -- trace the same page twice). A pure vector
PDF therefore hands Phase 2 no images at all.
"""
from __future__ import annotations

import numpy as np

from rastervec.commons.logging_setup import get_logger
from rastervec.commons.models import Image, Page

_LOG = get_logger("P1.image_extract")

# Fallback for an `Image.dpi` whose placement has a degenerate bbox width.
_FALLBACK_DPI = 150.0


def extract_images(page: Page) -> list[Image]:
    """One `Image` per embedded raster placement on `page`."""
    return _extract_embedded_images(page)


def _extract_embedded_images(page: Page) -> list[Image]:
    try:
        infos = page.fitz_page.get_image_info(xrefs=True)
    except Exception:  # noqa: BLE001 -- malformed image stream shouldn't crash extraction
        _LOG.warning("get_image_info failed on page %d", page.meta.index)
        return []

    images: list[Image] = []
    for info in infos:
        xref = info.get("xref", 0)
        if not xref:
            continue
        bbox = tuple(info.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        width, height = info.get("width", 0), info.get("height", 0)
        if width <= 0 or height <= 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        array = _extract_xref_pixels(page, xref)
        if array is None:
            continue
        dpi = width / ((bbox[2] - bbox[0]) / 72.0) if bbox[2] > bbox[0] else _FALLBACK_DPI
        transform = info.get("transform")
        images.append(Image(
            array=array, bbox=bbox, dpi=dpi, source="embedded", xref=xref,
            transform=tuple(float(v) for v in transform) if transform is not None else None,
        ))
    return images


def _extract_xref_pixels(page: Page, xref: int) -> np.ndarray | None:
    import pymupdf as fitz

    try:
        pix = fitz.Pixmap(page.fitz_page.parent, xref)
    except Exception:  # noqa: BLE001
        return None
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)  # drop alpha -- n would otherwise be 2/4
    if pix.colorspace and pix.colorspace.n not in (1, 3):
        pix = fitz.Pixmap(fitz.csRGB, pix)
    array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3:
        array = array[:, :, :3]
    else:
        array = array[:, :, 0]
    return array
