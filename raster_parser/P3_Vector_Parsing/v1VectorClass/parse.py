"""v1VectorClass -- the original raster_parser "Type-2" vector parsing as a Phase 3 backend.

1. `s1_fill_filter`: fill-glyph candidates -- filled (`"f"`) vectors that aren't white and
   aren't layout boxes or rules.
2. `s2_word_grouping`: group them into words by content-stream order + proximity, each with
   an orientation.
3. `s3_glyph_render` + `s4_word_ocr`: render each group alone and read it with PaddleOCR
   (`s4_paddle_ocr`).
4. `s5_orientation`: each word's `rotate` from its group's orientation and the page rotation.
5. Every vector that isn't a glyph candidate is returned as drawing, unchanged.

The steps run on the shared `Vector`; OCR words are `s4_word_ocr.OcrWord` records, converted to
the shared `Text` here (`_ocr_text`).
"""
from __future__ import annotations

from raster_parser.commons.models import Page, Text, Vector
from raster_parser.core.settings import RunSettings


def parse(vectors: list[Vector], page: Page, settings: RunSettings) -> tuple[list[Vector], list[Text]]:
    import pymupdf

    from raster_parser.P3_Vector_Parsing.v1VectorClass.s1_fill_filter import filter_text_vectors
    from raster_parser.P3_Vector_Parsing.v1VectorClass.s2_word_grouping import (
        Orientation,
        cluster_by_seqno,
        convert_vectors_to_glyphs,
        get_vectors,
    )
    from raster_parser.P3_Vector_Parsing.v1VectorClass.s4_word_ocr import ocr_word_groups
    from raster_parser.P3_Vector_Parsing.v1VectorClass.s5_orientation import ocr_rotate_for_target

    fill_ids = {v.seqno for v in filter_text_vectors(vectors)}
    drawings = [v for v in vectors if v.seqno not in fill_ids]

    page_rotation = int(page.meta.rotation or 0)
    glyphs = convert_vectors_to_glyphs([v for v in vectors if v.seqno in fill_ids])
    word_groups = cluster_by_seqno(glyphs, page_rotation)

    with pymupdf.open(str(page.doc_path)) as doc:
        pairs = ocr_word_groups(
            doc[page.meta.index],
            [get_vectors(wg) for wg in word_groups],
            [(wg.bbox.x0, wg.bbox.y0, wg.bbox.x1, wg.bbox.y1) for wg in word_groups],
            dpi=settings.ocr_dpi,
            use_angle_cls=settings.use_angle_cls,
        )

    texts: list[Text] = []
    for word, _ in pairs:
        wg = word_groups[word.block_no]
        target = 90 if wg.orientation == Orientation.VERTICAL else 0
        texts.append(_ocr_text(word, ocr_rotate_for_target(target, page_rotation), page.meta.index))
    texts.sort(key=lambda t: t.block_no)
    return drawings, texts


def _ocr_text(word, rotate: int, page_index: int) -> Text:
    """The shared `Text` for one OCR word: direction from `rotate`, no font or colour."""
    from raster_parser.commons.helpers.geometry import compute_origin
    from raster_parser.commons.helpers.text_orientation import rotate_to_direction

    bbox = (float(word.x0), float(word.y0), float(word.x1), float(word.y1))
    direction = rotate_to_direction(rotate)
    return Text(
        text=word.word, bbox=bbox, direction=direction, origin=compute_origin(bbox, direction),
        font="", font_size=0.0, color=None, flags=0, ascender=None, descender=None, wmode=0,
        block_no=word.block_no, line_no=word.line_no, word_no=word.word_no,
        page_index=page_index, seqno=0, source="ocr", orientation_source="dto",
    )
