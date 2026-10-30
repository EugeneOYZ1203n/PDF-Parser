"""v1_vector_class -- the original raster_parser "Type-2" vector parsing as a vector_classify backend.

1. `s1_fill_filter`: fill-glyph candidates -- filled (`"f"`) vectors that aren't white and
   aren't layout boxes or rules.
2. `s2_word_grouping`: group them into words by content-stream order + proximity, each with
   an orientation.
3. `s3_glyph_render` + `s4_word_ocr`: render each group alone and read it with PaddleOCR
   (`s4_paddle_ocr`).
4. `s5_orientation`: each word's `rotate` from its group's orientation and the page rotation.
5. Every vector that isn't a glyph candidate -- and every candidate in a group OCR could not read --
   is returned as drawing, unchanged.

The steps run on the shared `Vector`; each word read becomes a shared `Text` here
(`commons.v1_commons.ocr.ocr_text`) with its group's box and the group index as `block_no`.
"""

from __future__ import annotations

from raster_parser.models.source import Source
from raster_parser.models.page import Page
from raster_parser.models.text import Text
from raster_parser.models.vector import Vector
from raster_parser.models.settings import RunSettings
from raster_parser.models.vector_classify import Orientation


def parse(
    vectors: list[Vector], page: Page, settings: RunSettings
) -> tuple[list[Vector], list[Text]]:
    import pymupdf

    from raster_parser.commons.v1_commons.ocr import ocr_text
    from raster_parser.phase_vector_classify.v1_vector_class.s1_fill_filter import (
        filter_text_vectors,
    )
    from raster_parser.phase_vector_classify.v1_vector_class.s2_word_grouping import (
        cluster_by_seqno,
    )
    from raster_parser.phase_vector_classify.v1_vector_class.s4_word_ocr import ocr_word_groups
    from raster_parser.phase_vector_classify.v1_vector_class.s5_orientation import (
        ocr_rotate_for_target,
    )

    candidates = filter_text_vectors(vectors)
    candidate_ids = {id(v) for v in candidates}

    page_rotation = int(page.meta.rotation or 0)
    word_groups = cluster_by_seqno(candidates, page_rotation)

    with pymupdf.open(str(page.doc_path)) as doc:
        words = ocr_word_groups(
            doc[page.meta.index],
            [wg.vectors for wg in word_groups],
            [wg.bbox for wg in word_groups],
            dpi=settings.ocr_dpi,
            use_angle_cls=settings.use_angle_cls,
        )

    read = {index for index, _, _ in words}
    unread_ids = {id(v) for i, wg in enumerate(word_groups) if i not in read for v in wg.vectors}
    drawings = [v for v in vectors if id(v) not in candidate_ids or id(v) in unread_ids]

    texts: list[Text] = []
    for index, word, confidence in words:
        wg = word_groups[index]
        target = 90 if wg.orientation == Orientation.VERTICAL else 0
        rotate = ocr_rotate_for_target(target, page_rotation)
        texts.append(
            ocr_text(
                word,
                wg.bbox,
                rotate,
                page_index=page.meta.index,
                block_no=index,
                confidence=confidence,
                source=Source.NATIVE_VECTOR,
            )
        )
    texts.sort(key=lambda t: t.block_no)
    return drawings, texts
