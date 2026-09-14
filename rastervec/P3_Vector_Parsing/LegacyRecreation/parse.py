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
    debug_out: "dict | None" = None,
) -> tuple[list[Vector], list[Text]]:
    """Combines Phase 1's raw native vectors and Phase 2's raster-derived
    vectors, classifies them into Type-2 glyph-ink candidates vs everything
    else (drawing), groups the glyph candidates by seqno-adjacency into word
    groups, OCRs each group's own render, and returns
    (drawing_vectors, ocr_texts). When `debug_out` is given (a plain dict),
    this stage's own intermediate objects are stashed into it verbatim --
    see `render_debug` below."""
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

    if debug_out is not None:
        debug_out["fill_vectors"] = fill_vectors
        debug_out["drawing_vectors"] = drawing_vectors
        debug_out["word_groups"] = word_groups
        debug_out["texts"] = texts

    return drawing_vectors, texts


# ---------------------------------------------------------------------------
# Debug rendering -- this backend's own render function over its own
# `debug_out` shape, built only from the three generic primitives in
# `commons/renderer`. Nothing here is shared with VectorClassification/
# FastIntoPaddle/Junction.
# ---------------------------------------------------------------------------
_C_FILL = "#059669"
_C_DRAWING = "#111827"
_C_GROUP = "#7c3aed"
_C_OCR = "#16a34a"


def _hex_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def render_debug(debug_out: "dict | None", page_meta) -> "list[tuple[str, str, str, bytes]]":
    """One (stage, label, hex, pdf_bytes) tuple per debug layer, built
    straight from `parse()`'s own `debug_out` stash."""
    from rastervec.commons.renderer import render_boxes_pdf, render_text_pdf, render_vectors_pdf

    out: "list[tuple[str, str, str, bytes]]" = []
    if not debug_out:
        return out

    fill_vectors = debug_out.get("fill_vectors") or []
    out.append(("filter_fill", "fill vectors", _C_FILL, render_vectors_pdf(
        page_meta, fill_vectors, color_of=lambda _v: _hex_rgb(_C_FILL),
    )))

    word_groups = debug_out.get("word_groups") or []
    group_boxes = [wg.bbox for wg in word_groups]
    out.append(("group_words", "word group bbox", _C_GROUP, render_boxes_pdf(
        page_meta, [(b, _hex_rgb(_C_GROUP)) for b in group_boxes],
    )))

    texts = debug_out.get("texts") or []
    out.append(("ocr", "recognized text", _C_OCR, render_text_pdf(
        page_meta, texts, color_of=lambda _t: _hex_rgb(_C_OCR),
    )))

    drawing_vectors = debug_out.get("drawing_vectors") or []
    out.append(("drawing", "drawing vectors", _C_DRAWING, render_vectors_pdf(
        page_meta, drawing_vectors, color_of=lambda _v: _hex_rgb(_C_DRAWING),
    )))

    return out
