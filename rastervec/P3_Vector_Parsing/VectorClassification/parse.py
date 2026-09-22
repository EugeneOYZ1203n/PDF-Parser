"""VectorClassification's Phase3Backend entrypoint -- the 12-step
Vector_Classification chain + FAST filtering, then a merge-across-buckets +
seqno-clustering + full PaddleOCR detect/recognize pass, matching
archive/raster_parser/scripts/type2_dump_extraction_pipeline.py::
run_ocr_extraction's pattern (own `wordgrouping.py::cluster_by_seqno` +
`paddle_engine.py::PaddleDetectBackend`/`PaddleRecBackend`, the same
detect-then-recognize pair `P3_Vector_Parsing/LegacyRecreation/parse.py`
already ports independently). Fully self-contained (own fast_detect.py/
paddle_engine.py/layer_color_separation.py/wordgrouping.py/config.py) --
imports nothing from P3_Vector_Parsing/FastIntoPaddle,
P3_Vector_Parsing/LegacyRecreation, or P2_Raster_To_Vec.
"""
from __future__ import annotations

import numpy as np

from rastervec.commons.helpers.geometry import compute_origin, transform_direction
from rastervec.commons.models import Page, Vector, Text
from rastervec.commons.renderer import ocr_prep, pixel_to_page_bbox
from rastervec.P3_Vector_Parsing.VectorClassification.classify_vectors import classify_vectors
from rastervec.P3_Vector_Parsing.VectorClassification.config import (
    MAX_RENDER_DPI,
    MIN_RENDER_SIDE_PX,
    OCR_DPI,
    RECOGNITION_PAD_FRACTION,
    RENDER_PADDING_EXTRA_PT,
)
from rastervec.P3_Vector_Parsing.VectorClassification.fast_filter import detect_text_fast
from rastervec.P3_Vector_Parsing.VectorClassification.paddle_engine import (
    PaddleDetectBackend,
    PaddleRecBackend,
    _normalize_bgr,
    _normalize_rotation,
    _quad_rotation_deg,
    _rotate_crop,
)
from rastervec.P3_Vector_Parsing.VectorClassification.wordgrouping import (
    cluster_by_seqno,
    convert_vectors_to_glyphs,
    get_vectors,
)

STEP_NAMES = ["classify", "fast", "group_words", "ocr", "drawing"]

DebugLayer = "tuple[str, str, str, bytes]"
OnDebugLayer = "Callable[[str, str, str, bytes], None]"


def _cluster_render_padding(vectors: list[Vector]) -> float:
    """Page-space PDF-point margin for a seqno-cluster's own OCR render
    frame -- half its own max stroke width (so a stroke at the bbox edge
    isn't clipped) plus `RENDER_PADDING_EXTRA_PT`, matching
    `LegacyRecreation/parse.py`'s own identical helper."""
    return max((v.width or 0.0) for v in vectors) / 2.0 + RENDER_PADDING_EXTRA_PT


def parse(
    vectors_p1: list[Vector], vectors_p2: list[Vector], page: Page,
    *, enable_fast: bool = True, verbose: bool = False, compute=None, progress_counter=None,
    debug_out: "dict | None" = None, on_debug_layer: "OnDebugLayer | None" = None,
) -> tuple[list[Vector], list[Text]]:
    """Combines Phase 1's raw native vectors and Phase 2's raster-derived
    vectors into one flat pool, then: classify (12-step chain, per
    `(layer,color)` bucket) -> FAST filter (still per classification
    cluster) -> merge every FAST-surviving cluster's vectors across every
    bucket into one flat pool -> cluster by content-stream draw order
    (`cluster_by_seqno`) -> per seqno-cluster, render + PaddleOCR's full
    detect+recognize pass -> merge every dropped Vector as drawing content.

    Two independent, optional debug outlets (see `FastIntoPaddle/parse.py`
    for the shared convention): `debug_out` stashes every stage's own
    intermediate object verbatim for `render_debug` to render as a
    post-hoc batch later; `on_debug_layer` renders and emits each stage's
    layers immediately, right after that stage runs. The 12-step
    classification chain itself (`classify_vectors`) is one atomic call
    either way -- its own per-step kept/dropped breakdown is rendered as
    soon as it returns, still well before the later (heavier) fast/ocr
    stages run."""
    all_vectors = list(vectors_p1) + list(vectors_p2)
    page_meta = page.meta

    def _emit(layers: "list[DebugLayer]") -> None:
        if on_debug_layer is not None:
            for layer in layers:
                on_debug_layer(*layer)

    cls = classify_vectors(all_vectors, page, verbose=verbose)
    _emit(_render_classification_layers(page_meta, cls))

    flat_clusters = [
        [v for group in cluster for v in group] for cluster in cls.text_clusters
    ]
    fast = detect_text_fast(
        flat_clusters, page, enable_fast=enable_fast, verbose=verbose,
        compute=compute, progress_counter=progress_counter,
    )
    _emit(_render_fast_layers(page_meta, fast.passed, fast.dropped_vectors))

    merged_vectors = [v for cluster in fast.passed for v in cluster]
    page_rotation = int(page_meta.rotation or 0)
    glyphs = convert_vectors_to_glyphs(merged_vectors)
    word_groups = cluster_by_seqno(glyphs, page_rotation)
    _emit(_render_group_words_layers(page_meta, word_groups))

    rec_backend = PaddleRecBackend()
    det_backend = PaddleDetectBackend()
    texts: list[Text] = []
    ocr_crops: list[tuple[np.ndarray, str]] = []
    cluster_detections: list[tuple[np.ndarray, list]] = []
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
        cluster_detections.append((bgr, quads))
        if not quads:
            continue

        # _rotate_crop's output is cropped straight out of `bgr`, so it's
        # already BGR -- reverse channels back before recognize_crops, which
        # does its own RGB->BGR flip internally (same gotcha LegacyRecreation's
        # own identical loop works around).
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
                page_index=page_meta.index, seqno=group_vectors[0].seqno,
                confidence=box.confidence, source="ocr", orientation_source="ocr",
            ))
    _emit(_render_ocr_layers(page_meta, texts))

    drawing = list(cls.drawing_vectors) + list(fast.dropped_vectors)
    _emit(_render_drawing_layers(page_meta, drawing))

    if debug_out is not None:
        debug_out["classification"] = cls
        debug_out["fast_passed"] = fast.passed
        debug_out["fast_dropped"] = fast.dropped_vectors
        debug_out["fast_result"] = fast.page_result
        debug_out["word_groups"] = word_groups
        debug_out["texts"] = texts
        debug_out["ocr_crops"] = ocr_crops
        debug_out["cluster_detections"] = cluster_detections
        debug_out["drawing"] = drawing

    return drawing, texts


# ---------------------------------------------------------------------------
# Debug rendering -- one small `_render_<stage>_layers` helper per pipeline
# stage (same convention as `FastIntoPaddle/parse.py`), built only from the
# three generic primitives in `commons/renderer`
# (render_boxes_pdf/render_text_pdf/render_vectors_pdf). Each is called two
# ways: inline from `parse()` (streaming) and from `render_debug` below
# (batch, reading the same data back out of `debug_out`). Nothing here is
# shared with FastIntoPaddle/LegacyRecreation/Junction.
# ---------------------------------------------------------------------------
_C_KEPT = "#059669"
_C_DROPPED = "#dc2626"
_C_FAST_PASS = "#059669"
_C_FAST_DROP = "#dc2626"
_C_GROUP = "#7c3aed"
_C_OCR = "#16a34a"
_C_DRAWING = "#111827"

# Pass-through annotation steps that never drop anything -- not worth a
# debug layer of their own.
_SKIP_STEP_LABELS = {"Vector signatures", "Group stats"}


def _hex_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def _slug(label: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in label.lower())
    while "__" in keep:
        keep = keep.replace("__", "_")
    return keep.strip("_") or "step"


def _flatten_entries(entries: list) -> list[Vector]:
    """A step category's `groups` is `list[list[Vector]]` (steps 1-5) or
    `list[list[list[Vector]]]` (steps 6-12, tiered clusters) -- flatten
    either down to a plain `list[Vector]` (same pattern as
    `classify_vectors._collect_dropped`)."""
    out: list[Vector] = []
    for entry in entries:
        if entry and isinstance(entry[0], list):
            for g in entry:
                out.extend(g)
        else:
            out.extend(entry)
    return out


def _entry_bbox(entry):
    """Bbox of one group (`list[Vector]`) or one tiered cluster
    (`list[list[Vector]]`) -- reuses `_flatten_entries`'s per-entry shape
    dispatch by wrapping the single entry in a one-item list."""
    from rastervec.commons.helpers.geometry import union_bbox

    vectors = _flatten_entries([entry])
    return union_bbox([v.bbox for v in vectors]) if vectors else None


def _render_classification_layers(page_meta, cls) -> "list[DebugLayer]":
    from rastervec.commons.renderer import render_boxes_pdf, render_vectors_pdf

    out: "list[DebugLayer]" = []
    if cls is None or not cls.clustering:
        return out
    steps_per_bucket = [stage.steps for stage in cls.clustering.values() if stage.steps]
    if not steps_per_bucket:
        return out
    n_steps = len(steps_per_bucket[0])
    for i in range(n_steps):
        label = steps_per_bucket[0][i].label
        if label in _SKIP_STEP_LABELS:
            continue
        kept_groups: list = []
        dropped_groups: list = []
        for steps in steps_per_bucket:
            if i >= len(steps):
                continue
            for cat in steps[i].categories.values():
                if cat.role == "kept":
                    kept_groups.extend(cat.groups)
                elif cat.role == "dropped":
                    dropped_groups.extend(cat.groups)
        stage = f"classify_{i + 1:02d}_{_slug(label)}"
        out.append((stage, "kept", _C_KEPT, render_vectors_pdf(
            page_meta, _flatten_entries(kept_groups), color_of=lambda _v: _hex_rgb(_C_KEPT),
        )))
        out.append((stage, "dropped", _C_DROPPED, render_vectors_pdf(
            page_meta, _flatten_entries(dropped_groups), color_of=lambda _v: _hex_rgb(_C_DROPPED),
        )))
        kept_boxes = [b for b in (_entry_bbox(g) for g in kept_groups if g) if b is not None]
        dropped_boxes = [b for b in (_entry_bbox(g) for g in dropped_groups if g) if b is not None]
        out.append((stage, f"kept bbox ({len(kept_boxes)})", _C_KEPT, render_boxes_pdf(
            page_meta, [(b, _hex_rgb(_C_KEPT)) for b in kept_boxes],
        )))
        out.append((stage, f"dropped bbox ({len(dropped_boxes)})", _C_DROPPED, render_boxes_pdf(
            page_meta, [(b, _hex_rgb(_C_DROPPED)) for b in dropped_boxes],
        )))
    return out


def _render_fast_layers(page_meta, fast_passed, fast_dropped) -> "list[DebugLayer]":
    from rastervec.commons.helpers.geometry import union_bbox
    from rastervec.commons.renderer import render_boxes_pdf

    passed_boxes = [union_bbox([v.bbox for v in c]) for c in (fast_passed or []) if c]
    dropped_boxes = [v.bbox for v in (fast_dropped or [])]
    return [
        ("fast", "passed", _C_FAST_PASS, render_boxes_pdf(
            page_meta, [(b, _hex_rgb(_C_FAST_PASS)) for b in passed_boxes],
        )),
        ("fast", "dropped", _C_FAST_DROP, render_boxes_pdf(
            page_meta, [(b, _hex_rgb(_C_FAST_DROP)) for b in dropped_boxes],
        )),
    ]


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


def _render_drawing_layers(page_meta, drawing) -> "list[DebugLayer]":
    from rastervec.commons.renderer import render_vectors_pdf

    return [("drawing", "drawing vectors", _C_DRAWING, render_vectors_pdf(
        page_meta, drawing or [], color_of=lambda _v: _hex_rgb(_C_DRAWING),
    ))]


def render_debug(debug_out: "dict | None", page_meta) -> "list[DebugLayer]":
    """Batch/standalone counterpart to the `on_debug_layer` streaming path
    above: one (stage, label, hex, pdf_bytes) tuple per debug layer, built
    from a fully-populated `debug_out`. Called by
    `generate_pipeline_report.py` after a `verbose=True` run without an
    `on_debug_layer` callback."""
    if not debug_out:
        return []
    out: "list[DebugLayer]" = []
    out += _render_classification_layers(page_meta, debug_out.get("classification"))
    out += _render_fast_layers(page_meta, debug_out.get("fast_passed"), debug_out.get("fast_dropped"))
    out += _render_group_words_layers(page_meta, debug_out.get("word_groups"))
    out += _render_ocr_layers(page_meta, debug_out.get("texts"))
    out += _render_drawing_layers(page_meta, debug_out.get("drawing"))
    return out
