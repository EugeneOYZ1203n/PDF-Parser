"""FastIntoPaddle's Phase3Backend entrypoint -- similarity grouping -> FAST
filter -> reclassify -> (layer, color, width) separation -> spatial
clustering -> per-cluster PaddleOCR detect -> overlap reassignment ->
rotation refine -> PaddleOCR recognize. Fully self-contained -- imports
nothing from P3_Vector_Parsing/VectorClassification or P2_Raster_To_Vec.
"""
from __future__ import annotations

from rastervec.commons.models import Page, Text, Vector
from rastervec.P3_Vector_Parsing.FastIntoPaddle.config import FAST_PADDLE_SEQ_MERGE_TOLERANCE
from rastervec.P3_Vector_Parsing.FastIntoPaddle.paddle_engine import recognize_segments
from rastervec.P3_Vector_Parsing.FastIntoPaddle.steps import (
    build_drawing_output,
    cluster_buckets,
    detect_text_paddle_per_cluster,
    filter_vectors_fast,
    reassign_by_overlap,
    reclassify_by_similarity,
    rotate_paddle_detections,
    separate_by_layer_color_width,
    similarity_group,
)

STEP_NAMES = [
    "similarity", "fast", "reclassify", "separation",
    "clusters", "paddle_detect", "assignment", "rotate", "ocr", "drawing",
]


def parse(
    vectors_p1: list[Vector], vectors_p2: list[Vector], page: Page,
    *, enable_fast: bool = True, verbose: bool = False, compute=None, progress_counter=None,
    debug_out: "dict | None" = None,
) -> tuple[list[Vector], list[Text]]:
    """Combines Phase 1's raw native vectors and Phase 2's raster-derived
    vectors into one flat pool, then runs the FastIntoPaddle chain. When
    `debug_out` is given (a plain dict), every step's own result object is
    stashed into it verbatim -- see `render_debug` below."""
    vectors = list(vectors_p1) + list(vectors_p2)

    groups = similarity_group(vectors)
    fast = filter_vectors_fast(
        vectors, page, enable_fast=enable_fast, verbose=verbose,
        compute=compute, progress_counter=progress_counter,
    )
    reclass = reclassify_by_similarity(fast.passed, fast.dropped, groups)
    buckets = separate_by_layer_color_width(reclass.passed)
    clusters = cluster_buckets(buckets, FAST_PADDLE_SEQ_MERGE_TOLERANCE)
    cluster_detections = detect_text_paddle_per_cluster(clusters)
    reassignment = reassign_by_overlap(clusters, cluster_detections)
    segments = rotate_paddle_detections(clusters, cluster_detections, reassignment.text)

    recognize_fn = None
    if compute is not None:
        from rastervec.P3_Vector_Parsing.FastIntoPaddle.paddle_engine import _recognize_crops_job

        recognize_fn = lambda crops: compute.apply(_recognize_crops_job, (crops,))  # noqa: E731
    texts = recognize_segments(segments, recognize_fn=recognize_fn)

    drawing = build_drawing_output(reassignment.drawing, reclass.dropped)

    if debug_out is not None:
        debug_out["similarity_groups"] = groups
        debug_out["fast"] = fast
        debug_out["reclassify"] = reclass
        debug_out["clusters"] = clusters
        debug_out["cluster_detections"] = cluster_detections
        debug_out["reassignment"] = reassignment
        debug_out["segments"] = segments
        debug_out["texts"] = texts
        debug_out["drawing"] = drawing

    return drawing, texts


# ---------------------------------------------------------------------------
# Debug rendering -- this backend's own render function over its own
# `debug_out` shape, built only from the three generic primitives in
# `commons/renderer`. Nothing here is shared with VectorClassification/
# LegacyRecreation/Junction.
# ---------------------------------------------------------------------------
_C_SIMILARITY = "#7c3aed"
_C_FAST_PASS = "#059669"
_C_FAST_DROP = "#dc2626"
_C_RECLASS_PASS = "#059669"
_C_CLUSTER = "#ea580c"
_C_RENDER_BBOX = "#2563eb"
_C_DETECTION = "#16a34a"
_C_ASSIGNED_TEXT = "#16a34a"
_C_ASSIGNED_DRAWING = "#dc2626"
_C_OCR = "#16a34a"
_C_DRAWING = "#111827"


def _hex_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def render_debug(debug_out: "dict | None", page_meta) -> "list[tuple[str, str, str, bytes]]":
    """One (stage, label, hex, pdf_bytes) tuple per debug layer, built
    straight from `parse()`'s own `debug_out` stash."""
    from rastervec.commons.helpers.geometry import union_bbox
    from rastervec.commons.renderer import render_boxes_pdf, render_text_pdf, render_vectors_pdf

    out: "list[tuple[str, str, str, bytes]]" = []
    if not debug_out:
        return out

    groups = debug_out.get("similarity_groups") or []
    sim_vectors = [v for g in groups for v in g.members]
    out.append(("similarity", "vectors", _C_SIMILARITY, render_vectors_pdf(
        page_meta, sim_vectors, color_of=lambda _v: _hex_rgb(_C_SIMILARITY),
    )))

    fast = debug_out.get("fast")
    if fast is not None:
        passed_boxes = [v.bbox for v in fast.passed]
        dropped_boxes = [v.bbox for v in fast.dropped]
        out.append(("fast", "passed", _C_FAST_PASS, render_boxes_pdf(
            page_meta, [(b, _hex_rgb(_C_FAST_PASS)) for b in passed_boxes],
        )))
        out.append(("fast", "dropped", _C_FAST_DROP, render_boxes_pdf(
            page_meta, [(b, _hex_rgb(_C_FAST_DROP)) for b in dropped_boxes],
        )))

    reclass = debug_out.get("reclassify")
    if reclass is not None:
        out.append(("reclassify", "final pass", _C_RECLASS_PASS, render_boxes_pdf(
            page_meta, [(v.bbox, _hex_rgb(_C_RECLASS_PASS)) for v in reclass.passed],
        )))

    clusters = debug_out.get("clusters") or []
    cluster_boxes = [union_bbox([v.bbox for v in c]) for c in clusters if c]
    out.append(("clusters", "cluster bbox", _C_CLUSTER, render_boxes_pdf(
        page_meta, [(b, _hex_rgb(_C_CLUSTER)) for b in cluster_boxes],
    )))

    cluster_detections = debug_out.get("cluster_detections") or []
    detection_boxes = [
        d.bbox for cd in cluster_detections if cd is not None for d in cd.detections
    ]
    out.append(("paddle_detect", "cluster render bbox", _C_RENDER_BBOX, render_boxes_pdf(
        page_meta, [(b, _hex_rgb(_C_RENDER_BBOX)) for b in cluster_boxes],
    )))
    out.append(("paddle_detect", "detected box", _C_DETECTION, render_boxes_pdf(
        page_meta, [(b, _hex_rgb(_C_DETECTION)) for b in detection_boxes],
    )))

    reassignment = debug_out.get("reassignment")
    if reassignment is not None:
        assigned_text = [v for cluster in reassignment.text for grp in cluster for v in grp]
        out.append(("assignment", "assigned to text", _C_ASSIGNED_TEXT, render_vectors_pdf(
            page_meta, assigned_text, color_of=lambda _v: _hex_rgb(_C_ASSIGNED_TEXT),
        )))
        out.append(("assignment", "reassigned to drawing", _C_ASSIGNED_DRAWING, render_vectors_pdf(
            page_meta, reassignment.drawing, color_of=lambda _v: _hex_rgb(_C_ASSIGNED_DRAWING),
        )))

    segments = debug_out.get("segments") or []
    seg_boxes = [union_bbox([v.bbox for v in s.vectors]) for s in segments if s.vectors]
    out.append(("rotate", "rotated crop bbox", _C_DETECTION, render_boxes_pdf(
        page_meta, [(b, _hex_rgb(_C_DETECTION)) for b in seg_boxes],
    )))

    texts = debug_out.get("texts") or []
    out.append(("ocr", "recognized text", _C_OCR, render_text_pdf(
        page_meta, texts, color_of=lambda _t: _hex_rgb(_C_OCR),
    )))

    drawing = debug_out.get("drawing") or []
    out.append(("drawing", "drawing vectors", _C_DRAWING, render_vectors_pdf(
        page_meta, drawing, color_of=lambda _v: _hex_rgb(_C_DRAWING),
    )))

    return out
