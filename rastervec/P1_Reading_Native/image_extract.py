"""Raster images for a page -- Phase 1's third output, consumed by Phase 2.

Two sources, both returned: one whole-page render (a Phase 2 backend that
treats the page like a scan -- e.g. a classical raster-to-vector line-tracer
-- wants this), plus every real embedded raster image placement (via
`page.get_image_info(xrefs=True)`, the same API
`Evaluation/Labelling/raster_label.py::embedded_images_for_page` and
`Evaluation/inspector/pdf_model.py::extract_image_items` already use).
"""
from __future__ import annotations

import numpy as np

from rastervec.commons.logging_setup import get_logger
from rastervec.commons.models import Image, Page

_LOG = get_logger("P1.image_extract")

DEFAULT_PAGE_RENDER_DPI = 150.0


def extract_images(page: Page, *, dpi: float = DEFAULT_PAGE_RENDER_DPI) -> list[Image]:
    """`[whole-page render] + [one entry per embedded raster placement]`."""
    images = [_render_whole_page(page, dpi=dpi)]
    images.extend(_extract_embedded_images(page))
    return images


def _render_whole_page(page: Page, *, dpi: float) -> Image:
    zoom = dpi / 72.0
    import pymupdf as fitz

    # get_pixmap() always bakes the page's own `/Rotate` into what it
    # renders (regardless of the `matrix` passed), so a 90/270 page comes
    # back in *rotated* display space with dimensions swapped relative to
    # `page.meta.width`/`height`. Composing `derotation_matrix` first
    # cancels that baked-in rotation, landing back in the unrotated
    # MediaBox space every other Phase-1 output uses -- and that the
    # `bbox` below already assumes. Confirmed empirically (not just by
    # matrix algebra): `derotation_matrix * zoom` keeps pixel content
    # anchored to the unrotated page corners at every `/Rotate` value.
    fitz_page = page.fitz_page
    pix = fitz_page.get_pixmap(
        matrix=fitz_page.derotation_matrix * fitz.Matrix(zoom, zoom), alpha=False,
    )
    array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3:
        array = array[:, :, :3]
    else:
        array = array[:, :, 0]
    return Image(
        array=array,
        bbox=(0.0, 0.0, page.meta.width, page.meta.height),
        dpi=dpi,
        source="page",
    )


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
        dpi = width / ((bbox[2] - bbox[0]) / 72.0) if bbox[2] > bbox[0] else DEFAULT_PAGE_RENDER_DPI
        images.append(Image(array=array, bbox=bbox, dpi=dpi, source="embedded", xref=xref))
    return images


def _extract_xref_pixels(page: Page, xref: int) -> np.ndarray | None:
    import pymupdf as fitz

    try:
        pix = fitz.Pixmap(page.fitz_page.parent, xref)
    except Exception:  # noqa: BLE001
        return None
    if pix.colorspace and pix.colorspace.n not in (1, 3):
        pix = fitz.Pixmap(fitz.csRGB, pix)
    array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3:
        array = array[:, :, :3]
    else:
        array = array[:, :, 0]
    return array
