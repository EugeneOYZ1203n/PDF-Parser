"""Step 5 -- the `rotate` an OCR word reports, from its word group's orientation (horizontal
target 0, vertical target 90) and the page's `/Rotate`.

Copied from `rendering/pdf_render/reconstruct.py`.
"""

from __future__ import annotations


def _glyph_pdf_rotate(ocr_rotate: int, page_rotation: int) -> int:
    r = (int(ocr_rotate) - int(page_rotation)) % 360
    if int(page_rotation) % 360:
        r = (r + 180) % 360
    return r


def ocr_rotate_for_target(target_rotate: int, page_rotation: int) -> int:
    for cand in (0, 90, 180, 270):
        if _glyph_pdf_rotate(cand, page_rotation) == int(target_rotate) % 360:
            return cand
    return 0
