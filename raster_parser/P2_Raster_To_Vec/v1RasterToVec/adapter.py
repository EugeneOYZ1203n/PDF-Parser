"""v1RasterToVec -- the original raster_parser "Type-4" raster pass as a Phase 2 backend.

1. LibreOffice's DeleteDrawings macro removes the page's native text and vector shapes from a
   one-page copy (`clear_page`); embedded images stay.
2. If anything is left (`page_is_nonempty`), `raster_page.extract_raster_page` renders the
   residual page at `settings.ocr_dpi` (s2), splits text from drawing (s3), recognises the text
   boxes (s4) and colour-fill autotraces the drawing layer (s5).

It works on the whole page (from `page.doc_path`), so it ignores Phase 1's `images`. The steps
return their own records (`s4_region_ocr.OcrWord`, `s5_colour_autotrace.TraceVector`); they are
converted to the shared models here. Intermediates go under `settings.dump_dir`.
"""
from __future__ import annotations

from raster_parser.commons.models import Image, Page, Text, Vector
from raster_parser.core.settings import RunSettings


def extract(images: list[Image], page: Page, settings: RunSettings) -> tuple[list[Vector], list[Text]]:
    from raster_parser.P2_Raster_To_Vec.v1RasterToVec.raster_page import extract_raster_page
    from raster_parser.P2_Raster_To_Vec.v1RasterToVec.s1_clear_page import clear_page, page_is_nonempty
    from raster_parser.P2_Raster_To_Vec.v1RasterToVec.s5_colour_autotrace import trace_vector_to_vector

    work_dir, cleared = clear_page(page.doc_path, page.meta.index, dump_dir=settings.dump_dir)
    if not page_is_nonempty(cleared, 0):
        return [], []
    words, traced = extract_raster_page(
        cleared, 0, dpi=settings.ocr_dpi, workers=settings.workers, output_dump_dir=work_dir,
    )
    index = page.meta.index
    vectors = [trace_vector_to_vector(v, page_index=index) for v in traced]
    texts = [_ocr_text(w, index) for w in words]
    return vectors, texts


def _ocr_text(word, page_index: int) -> Text:
    """The shared `Text` for one OCR word: direction from its `rotate`, no font or colour."""
    from raster_parser.commons.helpers.geometry import compute_origin
    from raster_parser.commons.helpers.text_orientation import rotate_to_direction

    bbox = (float(word.x0), float(word.y0), float(word.x1), float(word.y1))
    direction = rotate_to_direction(word.rotate)
    return Text(
        text=word.word, bbox=bbox, direction=direction, origin=compute_origin(bbox, direction),
        font="", font_size=0.0, color=None, flags=0, ascender=None, descender=None, wmode=0,
        block_no=word.block_no, line_no=word.line_no, word_no=word.word_no,
        page_index=page_index, seqno=0, source="ocr", orientation_source="dto",
    )
