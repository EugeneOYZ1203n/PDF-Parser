"""LegacyRecreation's Phase3Backend entrypoint -- a ported (not wrapped)
recreation of archive/raster_parser's own Type-2 algorithm: classify filled
vectors into glyph-ink vs box/panel/rule, group by seqno-adjacency into word
groups, render + OCR each group, and merge with everything else as drawing
content. Fully self-contained -- own filters.py/wordgrouping.py/
paddle_engine.py/config.py, imports nothing from VectorClassification or
FastIntoPaddle. Unlike `Evaluation/Evaluate/legacy_adapter.py` (which
shells out to the real archive/raster_parser codebase unmodified, kept for
benchmark-baseline comparison), this is a genuine from-scratch port onto
commons.models types, selectable as a normal P3 backend.
"""
from __future__ import annotations

from rastervec.commons.models import Page, Text, Vector
from rastervec.commons.renderer import render_vector_cluster
from rastervec.commons.helpers.geometry import compute_origin, transform_direction, union_bbox
from rastervec.P3_Vector_Parsing.LegacyRecreation.config import OCR_DPI
from rastervec.P3_Vector_Parsing.LegacyRecreation.filters import filter_text_vectors, ocr_rotate_for_target
from rastervec.P3_Vector_Parsing.LegacyRecreation.paddle_engine import PaddleRecBackend
from rastervec.P3_Vector_Parsing.LegacyRecreation.wordgrouping import (
    cluster_by_seqno,
    convert_vectors_to_glyphs,
    get_vectors,
)

STEP_NAMES = ["filter_fill", "group_words", "ocr", "drawing"]


def parse(
    vectors_p1: list[Vector], vectors_p2: list[Vector], page: Page,
    *, verbose: bool = False, compute=None, progress_counter=None,
) -> tuple[list[Vector], list[Text]]:
    """Combines Phase 1's raw native vectors and Phase 2's raster-derived
    vectors, classifies them into Type-2 glyph-ink candidates vs everything
    else (drawing), groups the glyph candidates by seqno-adjacency into word
    groups, OCRs each group's own render, and returns
    (drawing_vectors, ocr_texts)."""
    all_vectors = list(vectors_p1) + list(vectors_p2)

    fill_vectors = filter_text_vectors(all_vectors)
    fill_ids = {id(v) for v in fill_vectors}
    drawing_vectors = [v for v in all_vectors if id(v) not in fill_ids]

    page_rotation = int(page.meta.rotation or 0)
    glyphs = convert_vectors_to_glyphs(fill_vectors)
    word_groups = cluster_by_seqno(glyphs, page_rotation)

    backend = PaddleRecBackend()
    texts: list[Text] = []
    for wg in word_groups:
        group_vectors = get_vectors(wg)
        if not group_vectors:
            continue
        try:
            image = render_vector_cluster(group_vectors, OCR_DPI)
        except ValueError:
            continue
        import numpy as np

        crop = np.asarray(image)
        boxes = backend.recognize_crops([crop])
        if not boxes or not boxes[0].text:
            continue
        box = boxes[0]
        bbox = union_bbox([v.bbox for v in group_vectors])
        target = 90 if wg.orientation == "vertical" else 0
        rotate_deg = ocr_rotate_for_target(target, page_rotation) + box.flip_deg
        direction = transform_direction((1.0, 0.0), rotate_deg)
        texts.append(Text(
            text=box.text, bbox=bbox, direction=direction,
            origin=compute_origin(bbox, direction),
            font="", font_size=0.0, color=None, flags=0,
            ascender=None, descender=None, wmode=0,
            block_no=0, line_no=0, word_no=0,
            page_index=page.meta.index, seqno=group_vectors[0].seqno,
            confidence=box.confidence, source="ocr", orientation_source="ocr",
        ))

    return drawing_vectors, texts
