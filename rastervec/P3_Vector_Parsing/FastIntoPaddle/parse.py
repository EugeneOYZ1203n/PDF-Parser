"""FastIntoPaddle's Phase3Backend entrypoint -- similarity grouping -> FAST
filter -> reclassify -> (layer, color, width) separation -> spatial
clustering -> per-cluster PaddleOCR detect -> overlap reassignment ->
rotation refine -> PaddleOCR recognize. Fully self-contained -- imports
nothing from P3_Vector_Parsing/VectorClassification or P2_Raster_To_Vec.
"""
from __future__ import annotations

from typing import Callable

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

DebugLayer = "tuple[str, str, str, bytes]"
OnDebugLayer = "Callable[[str, str, str, bytes], None]"


def parse(
    vectors_p1: list[Vector], vectors_p2: list[Vector], page: Page,
    *, enable_fast: bool = True, verbose: bool = False, compute=None, progress_counter=None,
    debug_out: "dict | None" = None, on_debug_layer: "OnDebugLayer | None" = None,
) -> tuple[list[Vector], list[Text]]:
    """Combines Phase 1's raw native vectors and Phase 2's raster-derived
    vectors into one flat pool, then runs the FastIntoPaddle chain.

    Two independent, optional debug outlets:
    - `debug_out` (a plain dict): every step's own result object is stashed
      into it verbatim, for `render_debug` to render as a post-hoc batch
      later (standalone/notebook use).
    - `on_debug_layer` (a callback): each step's own debug layer(s) are
      rendered immediately, right after that step runs, and handed to the
      callback one at a time -- so a caller (e.g. the report generator)
      never has to hold this backend's heavier step-local data (crop
      images, PaddleOCR detection renders) any longer than that one step's
      own rendering needs it."""
    vectors = list(vectors_p1) + list(vectors_p2)
    page_meta = page.meta

    def _emit(layers: "list[DebugLayer]") -> None:
        if on_debug_layer is not None:
            for layer in layers:
                on_debug_layer(*layer)

    groups = similarity_group(vectors)
    _emit(_render_similarity_layers(page_meta, groups))

    fast = filter_vectors_fast(
        vectors, page, enable_fast=enable_fast, verbose=verbose,
        compute=compute, progress_counter=progress_counter,
    )
    _emit(_render_fast_layers(page_meta, fast))

    reclass = reclassify_by_similarity(fast.passed, fast.dropped, groups)
    _emit(_render_reclassify_layers(page_meta, reclass))

    buckets = separate_by_layer_color_width(reclass.passed)
    clusters = cluster_buckets(buckets, FAST_PADDLE_SEQ_MERGE_TOLERANCE)
    _emit(_render_clusters_layers(page_meta, clusters))

    cluster_detections = detect_text_paddle_per_cluster(clusters)
    _emit(_render_paddle_detect_layers(page_meta, clusters, cluster_detections))

    reassignment = reassign_by_overlap(clusters, cluster_detections)
    _emit(_render_assignment_layers(page_meta, reassignment))

    segments = rotate_paddle_detections(clusters, cluster_detections, reassignment.text)
    _emit(_render_rotate_layers(page_meta, segments))

    recognize_fn = None
    if compute is not None:
        from rastervec.P3_Vector_Parsing.FastIntoPaddle.paddle_engine import _recognize_crops_job

        recognize_fn = lambda crops: compute.apply(_recognize_crops_job, (crops,))  # noqa: E731
    texts = recognize_segments(segments, recognize_fn=recognize_fn)
    _emit(_render_ocr_layers(page_meta, texts))

    drawing = build_drawing_output(reassignment.drawing, reclass.dropped)
    _emit(_render_drawing_layers(page_meta, drawing))

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
# Debug rendering -- one small `_render_<stage>_layers` helper per pipeline
# step, built only from the three generic primitives in `commons/renderer`.
# Each is called two ways: inline from `parse()` (right after that step
# computes its own data, the streaming/`on_debug_layer` path) and from
# `render_debug` below (reading the same data back out of a fully-populated
# `debug_out`, the batch/standalone path) -- so the rendering logic itself
# is never duplicated between the two call styles. Nothing here is shared
# with VectorClassification/LegacyRecreation/Junction.
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


def _render_similarity_layers(page_meta, groups) -> "list[DebugLayer]":
    from rastervec.commons.renderer import render_vectors_pdf

    groups = groups or []
    sim_vectors = [v for g in groups for v in g.members]
    return [("similarity", "vectors", _C_SIMILARITY, render_vectors_pdf(
        page_meta, sim_vectors, color_of=lambda _v: _hex_rgb(_C_SIMILARITY),
    ))]


def _render_fast_layers(page_meta, fast) -> "list[DebugLayer]":
    from rastervec.commons.renderer import render_boxes_pdf

    if fast is None:
        return []
    passed_boxes = [v.bbox for v in fast.passed]
    dropped_boxes = [v.bbox for v in fast.dropped]
    return [
        ("fast", "passed", _C_FAST_PASS, render_boxes_pdf(
            page_meta, [(b, _hex_rgb(_C_FAST_PASS)) for b in passed_boxes],
        )),
        ("fast", "dropped", _C_FAST_DROP, render_boxes_pdf(
            page_meta, [(b, _hex_rgb(_C_FAST_DROP)) for b in dropped_boxes],
        )),
    ]


def _render_reclassify_layers(page_meta, reclass) -> "list[DebugLayer]":
    from rastervec.commons.renderer import render_boxes_pdf

    if reclass is None:
        return []
    return [("reclassify", "final pass", _C_RECLASS_PASS, render_boxes_pdf(
        page_meta, [(v.bbox, _hex_rgb(_C_RECLASS_PASS)) for v in reclass.passed],
    ))]


def _cluster_boxes(clusters) -> list:
    from rastervec.commons.helpers.geometry import union_bbox

    return [union_bbox([v.bbox for v in c]) for c in (clusters or []) if c]


def _render_clusters_layers(page_meta, clusters) -> "list[DebugLayer]":
    from rastervec.commons.renderer import render_boxes_pdf

    cluster_boxes = _cluster_boxes(clusters)
    return [("clusters", "cluster bbox", _C_CLUSTER, render_boxes_pdf(
        page_meta, [(b, _hex_rgb(_C_CLUSTER)) for b in cluster_boxes],
    ))]


def _render_paddle_detect_layers(page_meta, clusters, cluster_detections) -> "list[DebugLayer]":
    from rastervec.commons.renderer import render_boxes_pdf

    cluster_boxes = _cluster_boxes(clusters)
    cluster_detections = cluster_detections or []
    detection_boxes = [
        d.bbox for cd in cluster_detections if cd is not None for d in cd.detections
    ]
    return [
        ("paddle_detect", "cluster render bbox", _C_RENDER_BBOX, render_boxes_pdf(
            page_meta, [(b, _hex_rgb(_C_RENDER_BBOX)) for b in cluster_boxes],
        )),
        ("paddle_detect", "detected box", _C_DETECTION, render_boxes_pdf(
            page_meta, [(b, _hex_rgb(_C_DETECTION)) for b in detection_boxes],
        )),
    ]


def _render_assignment_layers(page_meta, reassignment) -> "list[DebugLayer]":
    from rastervec.commons.renderer import render_vectors_pdf

    if reassignment is None:
        return []
    assigned_text = [v for cluster in reassignment.text for grp in cluster for v in grp]
    return [
        ("assignment", "assigned to text", _C_ASSIGNED_TEXT, render_vectors_pdf(
            page_meta, assigned_text, color_of=lambda _v: _hex_rgb(_C_ASSIGNED_TEXT),
        )),
        ("assignment", "reassigned to drawing", _C_ASSIGNED_DRAWING, render_vectors_pdf(
            page_meta, reassignment.drawing, color_of=lambda _v: _hex_rgb(_C_ASSIGNED_DRAWING),
        )),
    ]


def _render_rotate_layers(page_meta, segments) -> "list[DebugLayer]":
    from rastervec.commons.helpers.geometry import union_bbox
    from rastervec.commons.renderer import render_boxes_pdf

    seg_boxes = [union_bbox([v.bbox for v in s.vectors]) for s in (segments or []) if s.vectors]
    return [("rotate", "rotated crop bbox", _C_DETECTION, render_boxes_pdf(
        page_meta, [(b, _hex_rgb(_C_DETECTION)) for b in seg_boxes],
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
    from a fully-populated `debug_out` (e.g. `run_pipeline(..., verbose=True)`
    without an `on_debug_layer` callback)."""
    if not debug_out:
        return []
    out: "list[DebugLayer]" = []
    out += _render_similarity_layers(page_meta, debug_out.get("similarity_groups"))
    out += _render_fast_layers(page_meta, debug_out.get("fast"))
    out += _render_reclassify_layers(page_meta, debug_out.get("reclassify"))
    out += _render_clusters_layers(page_meta, debug_out.get("clusters"))
    out += _render_paddle_detect_layers(page_meta, debug_out.get("clusters"), debug_out.get("cluster_detections"))
    out += _render_assignment_layers(page_meta, debug_out.get("reassignment"))
    out += _render_rotate_layers(page_meta, debug_out.get("segments"))
    out += _render_ocr_layers(page_meta, debug_out.get("texts"))
    out += _render_drawing_layers(page_meta, debug_out.get("drawing"))
    return out
