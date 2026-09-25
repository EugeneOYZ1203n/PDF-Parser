"""VectorClassification's Phase3Backend entrypoint -- the reduced 2-step
Vector_Classification chain (seqno-overlap merge + spatial clustering) +
FAST filtering, then full PaddleOCR detect/recognize directly over each
FAST-surviving cluster (no re-clustering step in between -- a cluster
already *is* a `list[Vector]`, matching
archive/raster_parser/scripts/type2_dump_extraction_pipeline.py::
run_ocr_extraction's `paddle_engine.py::PaddleDetectBackend`/
`PaddleRecBackend` detect-then-recognize pair, the same pair
`P3_Vector_Parsing/LegacyRecreation/parse.py` already ports independently).
Fully self-contained (own fast_detect.py/paddle_engine.py/
layer_color_separation.py/config.py) -- imports nothing from
P3_Vector_Parsing/FastIntoPaddle, P3_Vector_Parsing/LegacyRecreation, or
P2_Raster_To_Vec.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from rastervec.commons.helpers.geometry import compute_origin, transform_direction
from rastervec.commons.models import Page, Vector, Text
from rastervec.commons.renderer import ocr_prep, pixel_to_page_bbox
from rastervec.commons.step_timing import StepClock
from rastervec.P3_Vector_Parsing.VectorClassification.classify_vectors import classify_vectors
from rastervec.P3_Vector_Parsing.VectorClassification.config import (
    MAX_RENDER_DPI,
    MIN_RENDER_SIDE_PX,
    OCR_DPI,
    RENDER_PADDING_EXTRA_PT,
)
from rastervec.P3_Vector_Parsing.VectorClassification.fast_filter import detect_text_fast
from rastervec.P3_Vector_Parsing.VectorClassification.paddle_engine import (
    PaddleDetectBackend,
    PaddleRecBackend,
    _normalize_bgr,
    _normalize_rotation,
    hough_deskew,
)

STEP_NAMES = ["classify", "fast", "ocr", "drawing"]

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
    step_durations: "dict | None" = None,
) -> tuple[list[Vector], list[Text]]:
    """Combines Phase 1's raw native vectors and Phase 2's raster-derived
    vectors into one flat pool, then: classify (reduced 2-step chain, per
    `(layer,color)` bucket) -> FAST filter (still per classification
    cluster) -> per FAST-surviving cluster (a plain `list[Vector]`, no
    re-clustering step in between), render + PaddleOCR's full
    detect+recognize pass -> merge every dropped Vector as drawing content
    (classify's own drops, always empty now that neither remaining step
    drops anything, plus FAST's drops).

    Two independent, optional debug outlets (see `FastIntoPaddle/parse.py`
    for the shared convention): `debug_out` stashes every stage's own
    intermediate object verbatim for `render_debug` to render as a
    post-hoc batch later; `on_debug_layer` renders and emits each stage's
    layers immediately, right after that stage runs. The classification
    chain itself (`classify_vectors`) is one atomic call either way -- its
    own per-step kept/dropped breakdown is rendered as soon as it returns,
    still well before the later (heavier) fast/ocr stages run.

    `step_durations`, when given, receives wall-clock seconds per step
    (`commons.step_timing.StepClock`; debug rendering excluded) --
    `classify`, `fast`, then the per-cluster OCR loop split into
    `ocr_render`/`ocr_detect`/`ocr_recognize` (summed over clusters), and
    `drawing`."""
    all_vectors = list(vectors_p1) + list(vectors_p2)
    page_meta = page.meta
    clock = StepClock(step_durations)

    def _emit(render: "Callable[[], list[DebugLayer]]") -> None:
        # Lazy: nothing is rendered unless a caller is listening. Render +
        # hand-off time is its own `debug_render` step, kept out of every
        # algorithmic step's timing.
        if on_debug_layer is None:
            return
        with clock("debug_render"):
            for layer in render():
                on_debug_layer(*layer)

    with clock("classify"):
        cls = classify_vectors(all_vectors, page, verbose=verbose)
    _emit(lambda: _render_classification_layers(page_meta, cls))

    with clock("fast"):
        flat_clusters = [
            [v for group in cluster for v in group] for cluster in cls.text_clusters
        ]
        fast = detect_text_fast(
            flat_clusters, page, enable_fast=enable_fast, verbose=verbose,
            compute=compute, progress_counter=progress_counter,
        )
    _emit(lambda: _render_fast_layers(
        page_meta, fast.passed, fast.dropped_vectors, fast.page_result.page_mask,
    ))

    rec_backend = PaddleRecBackend()
    det_backend = PaddleDetectBackend()
    texts: list[Text] = []
    ocr_crops: list[tuple[np.ndarray, str]] = []
    classifier_crops: list[tuple[np.ndarray, np.ndarray]] = []
    cluster_detections: list[tuple[np.ndarray, list]] = []
    blank_boxes: list[tuple] = []
    detect_boxes: list[tuple] = []
    rotation_entries: list[dict] = []
    for group_vectors in fast.passed:
        if not group_vectors:
            continue
        padding = _cluster_render_padding(group_vectors)
        try:
            with clock("ocr_render"):
                image, dpi_used = ocr_prep.render_cluster_with_dynamic_dpi(
                    group_vectors, OCR_DPI, MIN_RENDER_SIDE_PX, MAX_RENDER_DPI, padding,
                )
        except ValueError:
            continue

        with clock("ocr_render"):
            bgr = _normalize_bgr(np.asarray(image))
        with clock("ocr_detect"):
            quads = det_backend.detect(bgr)
        cluster_detections.append((bgr, quads))
        detect_boxes.extend(
            pixel_to_page_bbox(group_vectors, dpi_used, quad.tolist(), padding) for quad in quads
        )
        if not quads:
            continue

        # hough_deskew's crop is cropped straight out of `bgr`, so it's
        # already BGR -- reverse channels back before recognize_crops, which
        # does its own RGB->BGR flip internally (same gotcha LegacyRecreation's
        # own identical loop works around).
        with clock("ocr_recognize"):
            deskewed = [hough_deskew(bgr, quad) for quad in quads]
            crops = [c[:, :, ::-1] for c, _rd in deskewed]
            rotation_debugs = [rd for _c, rd in deskewed]
            boxes = rec_backend.recognize_crops(crops)

        for quad, crop, rd, box in zip(quads, crops, rotation_debugs, boxes):
            # box.flip_deg is the 0/180 decision recognize_crops' own angle
            # classifier made for this crop -- recognition actually ran on
            # the rotated (upright) pixels, not `crop` as-is, so mirror that
            # same rotation here for the stashed debug image too.
            recog_crop = np.rot90(crop, 2) if box.flip_deg else crop
            ocr_crops.append((recog_crop, box.text))
            classifier_crops.append((crop, recog_crop))
            bbox = pixel_to_page_bbox(group_vectors, dpi_used, quad.tolist(), padding)
            rotation_entries.append({
                "bbox": bbox,
                "quad_angle_deg": rd.quad_angle_deg,
                "hough_angle_deg": rd.hough_angle_deg,
                "combined_angle_deg": rd.combined_angle_deg,
                "base_crop": rd.base_crop,
                "dilated_ink_mask": rd.dilated_ink_mask,
            })
            if not box.text:
                blank_boxes.append(bbox)
                continue
            rotate_deg = _normalize_rotation(rd.combined_angle_deg + box.flip_deg)
            direction = transform_direction((1.0, 0.0), rotate_deg)
            texts.append(Text(
                text=box.text, bbox=bbox, direction=direction,
                origin=compute_origin(bbox, direction),
                font="", font_size=0.0, color=None, flags=0,
                ascender=None, descender=None, wmode=0,
                block_no=0, line_no=0, word_no=0,
                page_index=page_meta.index, seqno=min(v.seqno for v in group_vectors),
                confidence=box.confidence, source="ocr", orientation_source="ocr",
            ))
    _emit(lambda: _render_ocr_layers(page_meta, texts, blank_boxes, detect_boxes))
    _emit(lambda: _render_rotation_layers(page_meta, rotation_entries))

    with clock("drawing"):
        drawing = list(cls.drawing_vectors) + list(fast.dropped_vectors)
    _emit(lambda: _render_drawing_layers(page_meta, drawing))

    if debug_out is not None:
        debug_out["classification"] = cls
        debug_out["fast_passed"] = fast.passed
        debug_out["fast_dropped"] = fast.dropped_vectors
        debug_out["fast_result"] = fast.page_result
        debug_out["texts"] = texts
        debug_out["ocr_crops"] = ocr_crops
        debug_out["classifier_crops"] = classifier_crops
        debug_out["cluster_detections"] = cluster_detections
        debug_out["ocr_blank_boxes"] = blank_boxes
        debug_out["ocr_detect_boxes"] = detect_boxes
        debug_out["rotation"] = rotation_entries
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
_C_FAST_PASS = "#059669"
_C_FAST_DROP = "#dc2626"
_C_FAST_HEATMAP = "#f97316"
_C_OCR = "#16a34a"
_C_OCR_BLANK = "#9333ea"
_C_OCR_DETECT = "#2563eb"
_C_DRAWING = "#111827"
_C_ANGLE_QUAD = "#0891b2"
_C_ANGLE_HOUGH = "#ea580c"
_C_ANGLE_FINAL = "#65a30d"


def _hex_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def _slug(label: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in label.lower())
    while "__" in keep:
        keep = keep.replace("__", "_")
    return keep.strip("_") or "step"


def _flatten_entries(entries: list) -> list[Vector]:
    """A step category's `groups` is `list[list[Vector]]` ("Seq overlap
    merge") or `list[list[list[Vector]]]` ("Spatial cluster", tiered
    clusters) -- flatten either down to a plain `list[Vector]` (same
    pattern as `classify_vectors._collect_dropped`)."""
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
    """One `kept bbox` layer per classification step (the kept vectors
    themselves are visible in the inspector), and only for a step whose
    kept boxes differ from the previous step's -- a step that kept
    everything unchanged adds no layer."""
    from rastervec.commons.renderer import render_boxes_pdf

    out: "list[DebugLayer]" = []
    if cls is None or not cls.clustering:
        return out
    steps_per_bucket = [stage.steps for stage in cls.clustering.values() if stage.steps]
    if not steps_per_bucket:
        return out
    n_steps = len(steps_per_bucket[0])
    prev_boxes = None
    for i in range(n_steps):
        label = steps_per_bucket[0][i].label
        kept_groups: list = []
        for steps in steps_per_bucket:
            if i >= len(steps):
                continue
            for cat in steps[i].categories.values():
                if cat.role == "kept":
                    kept_groups.extend(cat.groups)
        kept_boxes = sorted(
            tuple(b) for b in (_entry_bbox(g) for g in kept_groups if g) if b is not None
        )
        if kept_boxes == prev_boxes:
            continue
        prev_boxes = kept_boxes
        stage = f"classify_{i + 1:02d}_{_slug(label)}"
        out.append((stage, "kept bbox", _C_KEPT, render_boxes_pdf(
            page_meta, [(b, _hex_rgb(_C_KEPT)) for b in kept_boxes],
        )))
    return out


def _render_fast_heatmap_pdf(page_meta, mask: "np.ndarray") -> bytes:
    """Full-page raster of FAST's stitched text-probability mask (the same
    array `fast_filter.FastPageResult.page_mask` samples per-cluster) as a
    white(0)->red(1) heat ramp, embedded as one full-page image -- lets a
    viewer see exactly what FAST scored across the page, not just which
    clusters passed/failed. Mirrors the `insert_image` pattern
    `commons/renderer/stages.py::_compose` already uses to embed a raster
    onto a page-sized PDF."""
    import io

    import pymupdf as fitz
    from PIL import Image

    clipped = np.clip(mask, 0.0, 1.0)
    rgb = np.empty((*clipped.shape, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    rgb[..., 1] = ((1.0 - clipped) * 255).astype(np.uint8)
    rgb[..., 2] = rgb[..., 1]
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")

    doc = fitz.open()
    try:
        page = doc.new_page(width=page_meta.width, height=page_meta.height)
        page.set_rotation(page_meta.rotation)
        page.insert_image(
            fitz.Rect(0, 0, page_meta.width, page_meta.height),
            stream=buf.getvalue(), keep_proportion=False,
        )
        return doc.tobytes()
    finally:
        doc.close()


def _render_fast_layers(page_meta, fast_passed, fast_dropped, page_mask=None) -> "list[DebugLayer]":
    from rastervec.commons.helpers.geometry import union_bbox
    from rastervec.commons.renderer import render_boxes_pdf

    passed_boxes = [union_bbox([v.bbox for v in c]) for c in (fast_passed or []) if c]
    dropped_boxes = [v.bbox for v in (fast_dropped or [])]
    layers: "list[DebugLayer]" = [
        ("fast", "passed", _C_FAST_PASS, render_boxes_pdf(
            page_meta, [(b, _hex_rgb(_C_FAST_PASS)) for b in passed_boxes],
        )),
        ("fast", "dropped", _C_FAST_DROP, render_boxes_pdf(
            page_meta, [(b, _hex_rgb(_C_FAST_DROP)) for b in dropped_boxes],
        )),
    ]
    if page_mask is not None:
        layers.append(("fast", "heatmap", _C_FAST_HEATMAP,
                        _render_fast_heatmap_pdf(page_meta, page_mask)))
    return layers


def _render_ocr_layers(page_meta, texts, blank_boxes=None, detect_boxes=None) -> "list[DebugLayer]":
    from rastervec.commons.renderer import render_boxes_pdf, render_text_pdf

    texts = texts or []
    passed_boxes = [t.bbox for t in texts]
    return [
        ("ocr", "detect bbox", _C_OCR_DETECT, render_boxes_pdf(
            page_meta, [(b, _hex_rgb(_C_OCR_DETECT)) for b in (detect_boxes or [])],
        )),
        ("ocr", "passed bbox", _C_OCR, render_boxes_pdf(
            page_meta, [(b, _hex_rgb(_C_OCR)) for b in passed_boxes],
        )),
        ("ocr", "failed bbox", _C_OCR_BLANK, render_boxes_pdf(
            page_meta, [(b, _hex_rgb(_C_OCR_BLANK)) for b in (blank_boxes or [])],
        )),
        ("ocr", "passed text", _C_OCR, render_text_pdf(
            page_meta, texts, color_of=lambda _t: _hex_rgb(_C_OCR),
        )),
    ]


def _render_angle_arrows_pdf(page_meta, entries: list[dict], angle_key: str, hexcolor: str) -> bytes:
    """A fresh page with one direction arrow per `entries` item whose
    `entries[i][angle_key]` isn't `None` -- centered on that entry's own
    `"bbox"`, pointing along the angle (same `transform_direction`
    convention `parse.py` already uses for real `Text.direction`), similar
    in spirit to `scripts/label/vector_label.py`'s rotation-arrow overlay.
    Visualizes `hough_deskew`'s three angle sources (quad/hough/final) side
    by side as toggleable layers -- see `_render_rotation_layers`."""
    import pymupdf as fitz

    color = _hex_rgb(hexcolor)
    doc = fitz.open()
    try:
        page = doc.new_page(width=page_meta.width, height=page_meta.height)
        page.set_rotation(page_meta.rotation)
        for entry in entries:
            angle = entry.get(angle_key)
            if angle is None:
                continue
            x0, y0, x1, y1 = entry["bbox"]
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            length = max(10.0, 0.4 * max(x1 - x0, y1 - y0))
            dx, dy = transform_direction((1.0, 0.0), angle)
            tail = (cx - dx * length / 2.0, cy - dy * length / 2.0)
            tip = (cx + dx * length / 2.0, cy + dy * length / 2.0)
            page.draw_line(tail, tip, color=color, width=1.5)
            head_len = length * 0.3
            for sign in (1.0, -1.0):
                wing = _rotate_vec(-dx, -dy, sign * 25.0)
                page.draw_line(
                    tip, (tip[0] + wing[0] * head_len, tip[1] + wing[1] * head_len),
                    color=color, width=1.5,
                )
        return doc.tobytes()
    finally:
        doc.close()


def _rotate_vec(dx: float, dy: float, deg: float) -> "tuple[float, float]":
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return dx * c - dy * s, dx * s + dy * c


def _render_rotation_layers(page_meta, rotation_entries: "list[dict] | None") -> "list[DebugLayer]":
    """Three arrow layers, one per angle source `hough_deskew` computes per
    detection (`parse.py`'s `debug_out["rotation"]`) -- quad's own
    dominant-edge angle, Hough's raw line-angle reading, and the combined,
    10-degree-snapped angle actually applied."""
    entries = rotation_entries or []
    return [
        ("rotation", "quad angle", _C_ANGLE_QUAD,
         _render_angle_arrows_pdf(page_meta, entries, "quad_angle_deg", _C_ANGLE_QUAD)),
        ("rotation", "hough angle", _C_ANGLE_HOUGH,
         _render_angle_arrows_pdf(page_meta, entries, "hough_angle_deg", _C_ANGLE_HOUGH)),
        ("rotation", "final angle", _C_ANGLE_FINAL,
         _render_angle_arrows_pdf(page_meta, entries, "combined_angle_deg", _C_ANGLE_FINAL)),
    ]


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
    fast_result = debug_out.get("fast_result")
    out: "list[DebugLayer]" = []
    out += _render_classification_layers(page_meta, debug_out.get("classification"))
    out += _render_fast_layers(
        page_meta, debug_out.get("fast_passed"), debug_out.get("fast_dropped"),
        fast_result.page_mask if fast_result is not None else None,
    )
    out += _render_ocr_layers(
        page_meta, debug_out.get("texts"), debug_out.get("ocr_blank_boxes"),
        debug_out.get("ocr_detect_boxes"),
    )
    out += _render_rotation_layers(page_meta, debug_out.get("rotation"))
    out += _render_drawing_layers(page_meta, debug_out.get("drawing"))
    return out
