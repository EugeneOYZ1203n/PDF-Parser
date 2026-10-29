"""Step 2 -- render the cleared page to BGR at `dpi`, plus the pixel -> PDF-point transform.

Copied from `parsing/extract_raster_page.py` (`page_to_bgr`).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pymupdf

from raster_parser.commons.helpers.pixmap import pixmap_to_array, zoom_matrix


def page_to_bgr(pdf_path: str | Path, page_index: int, dpi: int):
    """Render one PDF page to BGR and the pixel->PDF-point transform."""
    doc = pymupdf.open(str(pdf_path))
    try:
        page = doc[page_index]
        to_points = ~(page.rotation_matrix * zoom_matrix(dpi))
        pix = page.get_pixmap(dpi=dpi)
    finally:
        doc.close()
    # pix.n is the number of channels per pixel (1 gray, 3 RGB, 4 RGBA).
    img = pixmap_to_array(pix)
    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR), to_points
    if pix.n == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), to_points
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), to_points
