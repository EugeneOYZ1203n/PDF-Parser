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

import numpy as np

from rastervec.commons.models import Page, Text, Vector
from rastervec.commons.renderer import ocr_prep, pixel_to_page_bbox
from rastervec.commons.helpers.geometry import compute_origin, transform_direction
from rastervec.P3_Vector_Parsing.LegacyRecreation.config import (
    MAX_RENDER_DPI,
    MIN_RENDER_SIDE_PX,
    OCR_DPI,
    RECOGNITION_PAD_FRACTION,
    RENDER_PADDING_EXTRA_PT,
)
from rastervec.P3_Vector_Parsing.LegacyRecreation.filters import filter_text_vectors
from rastervec.P3_Vector_Parsing.LegacyRecreation.paddle_engine import (
    PaddleDetectBackend,
    PaddleRecBackend,
    _normalize_bgr,
    _normalize_rotation,
    _quad_rotation_deg,
    _rotate_crop,
)
from rastervec.P3_Vector_Parsing.LegacyRecreation.wordgrouping import (
    cluster_by_seqno,
    convert_vectors_to_glyphs,
    get_vectors,
)

STEP_NAMES = ["filter_fill", "group_words", "ocr", "drawing"]

DebugLayer = "tuple[str, str, str, bytes]"
OnDebugLayer = "Callable[[str, str, str, bytes], None]"


def _cluster_render_padding(vectors: list[Vector]) -> float:
    """Page-space PDF-point margin for a word group's own OCR render frame --
    half its own max stroke width (so a stroke at the bbox edge isn't
    clipped) plus `RENDER_PADDING_EXTRA_PT`, matching FastIntoPaddle/
    steps.py's own `max(v.width)/2 + <extra>` convention."""
    return max((v.width or 0.0) for v in vectors) / 2.0 + RENDER_PADDING_EXTRA_PT


def parse(
    vectors_p1: list[Vector], vectors_p2: list[Vector], page: Page,
    *, verbose: bool = False, compute=None, progress_counter=None,
    debug_out: "dict | None" = None, on_debug_layer: "OnDebugLayer | None" = None,
) -> tuple[list[Vector], list[Text]]:
    """Combines Phase 1's raw native vectors and Phase 2's raster-derived
    vectors, classifies them into Type-2 glyph-ink candidates vs everything
    else (drawing), groups the glyph candidates by seqno-adjacency into word
    groups, OCRs each group's own render, and returns
    (drawing_vectors, ocr_texts).

    Two independent, optional debug outlets (see `FastIntoPaddle/parse.py`
    for the shared convention): `debug_out` stashes every stage's own
    intermediate object verbatim for `render_debug` to render as a
    post-hoc batch later; `on_debug_layer` renders and emits each stage's
    layers immediately, right after that stage runs."""
    all_vectors = list(vectors_p1) + list(vectors_p2)
    page_meta = page.meta

    def _emit(layers: "list[DebugLayer]") -> None:
        if on_debug_layer is not None:
            for layer in layers:
                on_debug_layer(*layer)

    fill_vectors = filter_text_vectors(all_vectors)
    fill_ids = {id(v) for v in fill_vectors}
    drawing_vectors = [v for v in all_vectors if id(v) not in fill_ids]
    _emit(_render_filter_fill_layers(page_meta, fill_vectors))

    page_rotation = int(page.meta.rotation or 0)
    glyphs = convert_vectors_to_glyphs(fill_vectors)
    word_groups = cluster_by_seqno(glyphs, page_rotation)
    _emit(_render_group_words_layers(page_meta, word_groups))

    rec_backend = PaddleRecBackend()
    det_backend = PaddleDetectBackend()
    texts: list[Text] = []
    ocr_crops: list[tuple[np.ndarray, str]] = []
    for wg in word_groups:
        group_vectors = get_vectors(wg)
        if not group_vectors:
            continue
        padding = _cluster_render_padding(group_vectors)
        try:
            image, dpi_used = ocr_prep.render_cluster_with_dynamic_dpi(
                group_vectors, OCR_DPI, MIN_RENDER_SIDE_PX, MAX_RENDER_DPI, padding,
            )
        except ValueError:
            continue

        bgr = _normalize_bgr(np.asarray(image))
        bgr, (pad_x_px, pad_y_px) = ocr_prep.pad_image_uniform(bgr, RECOGNITION_PAD_FRACTION)
        quads = det_backend.detect(bgr)
        if not quads:
            continue

        # _rotate_crop's output is cropped straight out of `bgr`, so it's
        # already BGR -- reverse channels back before recognize_crops, which
        # does its own RGB->BGR flip internally (same gotcha as
        # scripts/verify_ocr.py::_run_paddle_full; failing to reverse first
        # double-flips the channels).
        crops = [_rotate_crop(bgr, quad)[:, :, ::-1] for quad in quads]
        boxes = rec_backend.recognize_crops(crops)

        for quad, crop, box in zip(quads, crops, boxes):
            if not box.text:
                continue
            unpadded_quad = (quad - np.array([pad_x_px, pad_y_px])).tolist()
            bbox = pixel_to_page_bbox(group_vectors, dpi_used, unpadded_quad, padding)
            rotate_deg = _normalize_rotation(_quad_rotation_deg(quad) + box.flip_deg)
            direction = transform_direction((1.0, 0.0), rotate_deg)
            ocr_crops.append((crop, box.text))
            texts.append(Text(
                text=box.text, bbox=bbox, direction=direction,
                origin=compute_origin(bbox, direction),
                font="", font_size=0.0, color=None, flags=0,
                ascender=None, descender=None, wmode=0,
                block_no=0, line_no=0, word_no=0,
                page_index=page.meta.index, seqno=group_vectors[0].seqno,
                confidence=box.confidence, source="ocr", orientation_source="ocr",
            ))
    _emit(_render_ocr_layers(page_meta, texts))
    _emit(_render_drawing_layers(page_meta, drawing_vectors))

    if debug_out is not None:
        debug_out["fill_vectors"] = fill_vectors
        debug_out["drawing_vectors"] = drawing_vectors
        debug_out["word_groups"] = word_groups
        debug_out["texts"] = texts
        debug_out["ocr_crops"] = ocr_crops

    return drawing_vectors, texts


# ---------------------------------------------------------------------------
# Debug rendering -- one small `_render_<stage>_layers` helper per pipeline
# stage (same convention as `FastIntoPaddle/parse.py`), built only from the
# three generic primitives in `commons/renderer`. Each is called two ways:
# inline from `parse()` (streaming) and from `render_debug` below (batch,
# reading the same data back out of `debug_out`). Nothing here is shared
# with VectorClassification/FastIntoPaddle/Junction.
# ---------------------------------------------------------------------------
_C_FILL = "#059669"
_C_DRAWING = "#111827"
_C_GROUP = "#7c3aed"
_C_OCR = "#16a34a"


def _hex_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def _render_filter_fill_layers(page_meta, fill_vectors) -> "list[DebugLayer]":
    from rastervec.commons.renderer import render_vectors_pdf

    return [("filter_fill", "fill vectors", _C_FILL, render_vectors_pdf(
        page_meta, fill_vectors or [], color_of=lambda _v: _hex_rgb(_C_FILL),
    ))]


def _render_group_words_layers(page_meta, word_groups) -> "list[DebugLayer]":
    from rastervec.commons.renderer import render_boxes_pdf

    group_boxes = [wg.bbox for wg in (word_groups or [])]
    return [("group_words", "word group bbox", _C_GROUP, render_boxes_pdf(
        page_meta, [(b, _hex_rgb(_C_GROUP)) for b in group_boxes],
    ))]


def _render_ocr_layers(page_meta, texts) -> "list[DebugLayer]":
    from rastervec.commons.renderer import render_text_pdf

    return [("ocr", "recognized text", _C_OCR, render_text_pdf(
        page_meta, texts or [], color_of=lambda _t: _hex_rgb(_C_OCR),
    ))]


def _render_drawing_layers(page_meta, drawing_vectors) -> "list[DebugLayer]":
    from rastervec.commons.renderer import render_vectors_pdf

    return [("drawing", "drawing vectors", _C_DRAWING, render_vectors_pdf(
        page_meta, drawing_vectors or [], color_of=lambda _v: _hex_rgb(_C_DRAWING),
    ))]


def render_debug(debug_out: "dict | None", page_meta) -> "list[DebugLayer]":
    """Batch/standalone counterpart to the `on_debug_layer` streaming path
    above: one (stage, label, hex, pdf_bytes) tuple per debug layer, built
    from a fully-populated `debug_out`."""
    if not debug_out:
        return []
    out: "list[DebugLayer]" = []
    out += _render_filter_fill_layers(page_meta, debug_out.get("fill_vectors"))
    out += _render_group_words_layers(page_meta, debug_out.get("word_groups"))
    out += _render_ocr_layers(page_meta, debug_out.get("texts"))
    out += _render_drawing_layers(page_meta, debug_out.get("drawing_vectors"))
    return out
