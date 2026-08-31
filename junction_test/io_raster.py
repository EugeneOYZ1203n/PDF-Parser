"""Load a raster to analyse: from an image file, or from a PDF (embedded
raster image, or a whole-page render as fallback)."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class LoadedRaster:
    gray: np.ndarray            # uint8 HxW
    source: str                 # human-readable provenance
    page_transform: tuple[float, float, float] | None = None
    # (scale, off_x, off_y): page_pt = pixel * scale + (off_x, off_y). None for plain images.


def load_image(path: str) -> LoadedRaster:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return LoadedRaster(gray=img, source=f"file:{path}")


def load_pdf_page_images(pdf_path: str, page_index: int, min_px: int = 64) -> list[LoadedRaster]:
    """Every embedded raster image on the page, largest first, as grayscale.

    page_transform maps image-pixel coords back to PDF page points using the
    image's placement rectangle.
    """
    import pymupdf as fitz

    out: list[LoadedRaster] = []
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        for img in page.get_images(full=True):
            xref = img[0]
            rects = page.get_image_rects(xref)
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha >= 3:
                pix = fitz.Pixmap(fitz.csGRAY, pix)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            gray = arr[:, :, 0]
            if gray.shape[0] < min_px or gray.shape[1] < min_px:
                continue
            transform = None
            if rects:
                r = rects[0]
                sx = (r.x1 - r.x0) / pix.width
                sy = (r.y1 - r.y0) / pix.height
                transform = (float((sx + sy) / 2), float(r.x0), float(r.y0))
            out.append(
                LoadedRaster(
                    gray=gray.copy(),
                    source=f"pdf:{pdf_path}#p{page_index} xref={xref} {pix.width}x{pix.height}",
                    page_transform=transform,
                )
            )
        out.sort(key=lambda lr: -lr.gray.size)
        return out
    finally:
        doc.close()


def render_pdf_page(pdf_path: str, page_index: int, dpi: int = 200) -> LoadedRaster:
    """Whole-page raster (fallback for native-vector PDFs)."""
    import pymupdf as fitz

    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False)
        gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        return LoadedRaster(
            gray=gray.copy(),
            source=f"pdf-render:{pdf_path}#p{page_index}@{dpi}dpi",
            page_transform=(72.0 / dpi, 0.0, 0.0),
        )
    finally:
        doc.close()
