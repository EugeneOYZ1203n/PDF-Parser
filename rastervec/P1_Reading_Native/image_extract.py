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

# What a masked pixel composites onto -- the page's paper color, so a
# fully-transparent region doesn't leak whatever undefined color PyMuPDF
# stored underneath its mask into the extracted array.
_COMPOSITE_BACKGROUND = 255.0


def extract_images(page: Page) -> list[Image]:
    """One `Image` per embedded raster placement on `page`."""
    return _extract_embedded_images(page)


def _extract_embedded_images(page: Page) -> list[Image]:
    try:
        infos = page.fitz_page.get_image_info(xrefs=True)
    except Exception:  # noqa: BLE001 -- malformed image stream shouldn't crash extraction
        _LOG.warning("get_image_info failed on page %d", page.meta.index)
        return []

    smask_of, mask_owned_xrefs = _collect_mask_info(page)

    images: list[Image] = []
    for info in infos:
        xref = info.get("xref", 0)
        if not xref:
            continue
        if xref in mask_owned_xrefs:
            # This xref is another image's /SMask or stencil /Mask --
            # transparency data, not standalone page content.
            continue
        bbox = tuple(info.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        width, height = info.get("width", 0), info.get("height", 0)
        if width <= 0 or height <= 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        array = _extract_xref_pixels(page, xref, smask_of.get(xref, 0))
        if array is None:
            continue
        dpi = width / ((bbox[2] - bbox[0]) / 72.0) if bbox[2] > bbox[0] else _FALLBACK_DPI
        transform = info.get("transform")
        images.append(Image(
            array=array, bbox=bbox, dpi=dpi, source="embedded", xref=xref,
            transform=tuple(float(v) for v in transform) if transform is not None else None,
        ))
    return images


def _collect_mask_info(page: Page) -> tuple[dict[int, int], set[int]]:
    """`(smask_of, mask_owned_xrefs)` for every image XObject `page`
    references. `smask_of[xref]` is that image's own `/SMask` xref (absent
    if it has none) -- PyMuPDF does *not* auto-composite this into
    `fitz.Pixmap(doc, xref).alpha`, so it has to be fetched and blended in
    separately (see `_extract_xref_pixels`). `mask_owned_xrefs` is every
    xref that exists only as another image's `/SMask` or stencil `/Mask` --
    transparency data for a different image, never independent page
    content, even on the rare page where `get_image_info` also enumerates
    it as its own placement (a stencil `/Mask` is excluded here but, unlike
    `/SMask`, not composited -- see `_extract_xref_pixels`)."""
    doc = page.fitz_page.parent
    try:
        full_images = page.fitz_page.get_images(full=True)
    except Exception:  # noqa: BLE001
        full_images = []

    smask_of: dict[int, int] = {}
    owned: set[int] = set()
    for entry in full_images:
        xref, smask = entry[0], entry[1]
        if smask:
            smask_of[xref] = smask
            owned.add(smask)
        try:
            mask_key = doc.xref_get_key(xref, "Mask")
        except Exception:  # noqa: BLE001
            continue
        if mask_key and mask_key[0] == "xref":
            # e.g. ("xref", "123 0 R") -- an indirect reference to a
            # stencil-mask XObject, as opposed to a color-key array.
            try:
                owned.add(int(mask_key[1].split()[0]))
            except (ValueError, IndexError):
                pass
    return smask_of, owned


def _extract_xref_pixels(page: Page, xref: int, smask_xref: int = 0) -> np.ndarray | None:
    import pymupdf as fitz

    try:
        pix = fitz.Pixmap(page.fitz_page.parent, xref)
    except Exception:  # noqa: BLE001
        _LOG.warning("failed to open pixmap for xref %d on page %d", xref, page.meta.index)
        return None

    if pix.colorspace is None:
        # A standalone stencil mask (`/ImageMask true`) placed directly on
        # the page and not owned by any other image per `_collect_mask_info`
        # -- its pixel data is a paint/don't-paint mask, not color, so
        # there's nothing to extract as raster content here.
        _LOG.warning("skipping colorless mask/stencil xref %d on page %d", xref, page.meta.index)
        return None

    try:
        if smask_xref:
            pix = _composite_smask(page.fitz_page.parent, pix, smask_xref)
        elif pix.alpha:
            # Rare: an image whose own pixel data carries alpha directly
            # (no separate /SMask xref) -- composite the same way.
            pix = _composite_over_background(pix)
        if pix.colorspace.n not in (1, 3):
            pix = fitz.Pixmap(fitz.csRGB, pix)
        array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    except Exception:  # noqa: BLE001 -- malformed image stream
        _LOG.warning("failed to extract pixels for xref %d on page %d", xref, page.meta.index)
        return None

    if pix.n >= 3:
        array = array[:, :, :3]
    else:
        array = array[:, :, 0]
    return array


def _composite_smask(doc, pix, smask_xref: int):
    """Blend `pix`'s color channels over `_COMPOSITE_BACKGROUND` using its
    `/SMask` (`smask_xref`, its own standalone grayscale pixmap: 0 =
    fully transparent, 255 = fully opaque), instead of leaving whatever
    undefined color the base image stores under fully-transparent pixels."""
    import pymupdf as fitz

    try:
        mask_pix = fitz.Pixmap(doc, smask_xref)
    except Exception:  # noqa: BLE001
        return pix  # no mask data to apply -- fall back to raw color
    if (mask_pix.width, mask_pix.height) != (pix.width, pix.height):
        mask_pix = fitz.Pixmap(mask_pix, pix.width, pix.height)

    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    ).astype(np.float32)
    mask_arr = np.frombuffer(mask_pix.samples, dtype=np.uint8).reshape(
        mask_pix.height, mask_pix.width, mask_pix.n
    )
    alpha = mask_arr[:, :, :1].astype(np.float32) / 255.0
    composited = rgb * alpha + _COMPOSITE_BACKGROUND * (1.0 - alpha)
    samples = composited.astype(np.uint8).tobytes()
    return fitz.Pixmap(pix.colorspace, pix.width, pix.height, samples, 0)


def _composite_over_background(pix):
    """Blend `pix`'s color channels over `_COMPOSITE_BACKGROUND` using its
    own, already-embedded alpha channel."""
    import pymupdf as fitz

    array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    rgb = array[:, :, :-1].astype(np.float32)
    alpha = array[:, :, -1:].astype(np.float32) / 255.0
    composited = rgb * alpha + _COMPOSITE_BACKGROUND * (1.0 - alpha)
    samples = composited.astype(np.uint8).tobytes()
    return fitz.Pixmap(pix.colorspace, pix.width, pix.height, samples, 0)
