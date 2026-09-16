"""VectorClassification's Phase3Backend entrypoint -- the restored 12-step
Vector_Classification chain + Radon word-segmentation + PaddleOCR
recognition-only OCR, exactly as it ran before being retired in favor of
FastIntoPaddle. Fully self-contained (own fast_detect.py/paddle_engine.py/
layer_color_separation.py/radon.py/config.py) -- imports nothing from
P3_Vector_Parsing/FastIntoPaddle or P2_Raster_To_Vec.
"""
from __future__ import annotations

from rastervec.commons.models import Page, Vector, Text
from rastervec.P3_Vector_Parsing.VectorClassification.classify_vectors import classify_vectors
from rastervec.P3_Vector_Parsing.VectorClassification.fast_filter import (
    detect_text_fast,
    elect_unique_segments,
    group_similar_segments,
)
from rastervec.P3_Vector_Parsing.VectorClassification.ocr import recognize_unique_words, restore_word_texts
from rastervec.P3_Vector_Parsing.VectorClassification.radon import segment_clusters

STEP_NAMES = ["classify", "fast", "segment", "similarity", "ocr", "restore", "drawing"]

DebugLayer = "tuple[str, str, str, bytes]"
OnDebugLayer = "Callable[[str, str, str, bytes], None]"


def parse(
    vectors_p1: list[Vector], vectors_p2: list[Vector], page: Page,
    *, enable_fast: bool = True, verbose: bool = False, compute=None, progress_counter=None,
    debug_out: "dict | None" = None, on_debug_layer: "OnDebugLayer | None" = None,
) -> tuple[list[Vector], list[Text]]:
    """Combines Phase 1's raw native vectors and Phase 2's raster-derived
    vectors into one flat pool, then runs the old chain: classify -> FAST
    filter -> Radon word-segmentation -> similarity dedup -> OCR recognize
    -> restore each word occurrence's text -> merge every dropped Vector as
    drawing content.

    Two independent, optional debug outlets (see `FastIntoPaddle/parse.py`
    for the shared convention): `debug_out` stashes every stage's own
    intermediate object verbatim for `render_debug` to render as a
    post-hoc batch later; `on_debug_layer` renders and emits each stage's
    layers immediately, right after that stage runs. The 12-step
    classification chain itself (`classify_vectors`) is one atomic call
    either way -- its own per-step kept/dropped breakdown is rendered as
    soon as it returns, still well before the later (heavier) fast/segment/
    ocr stages run."""
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

    word_segments = segment_clusters(fast.passed)
    _emit(_render_segment_layers(page_meta, word_segments))

    groups = group_similar_segments(word_segments)
    uniques, metas = elect_unique_segments(word_segments, groups)

    unique_texts = recognize_unique_words(uniques, compute=compute, progress_counter=progress_counter)
    restored = restore_word_texts(unique_texts, metas)
    _emit(_render_ocr_layers(page_meta, restored))

    drawing = list(cls.drawing_vectors) + list(fast.dropped_vectors)
    _emit(_render_drawing_layers(page_meta, drawing))

    if debug_out is not None:
        debug_out["classification"] = cls
        debug_out["fast_passed"] = fast.passed
        debug_out["fast_dropped"] = fast.dropped_vectors
        debug_out["word_segments"] = word_segments
        debug_out["restored"] = restored
        debug_out["drawing"] = drawing

    return drawing, restored


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
_C_SEGMENT = "#2563eb"
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


def _render_segment_layers(page_meta, word_segments) -> "list[DebugLayer]":
    from rastervec.commons.helpers.geometry import union_bbox
    from rastervec.commons.renderer import render_boxes_pdf

    segs = word_segments or []
    seg_boxes = [union_bbox([v.bbox for v in s.vectors]) for s in segs if s.vectors]
    return [("segment", "word boxes", _C_SEGMENT, render_boxes_pdf(
        page_meta, [(b, _hex_rgb(_C_SEGMENT)) for b in seg_boxes],
    ))]


def _render_ocr_layers(page_meta, restored) -> "list[DebugLayer]":
    from rastervec.commons.renderer import render_text_pdf

    return [("ocr", "recognized text", _C_OCR, render_text_pdf(
        page_meta, restored or [], color_of=lambda _t: _hex_rgb(_C_OCR),
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
    out += _render_segment_layers(page_meta, debug_out.get("word_segments"))
    out += _render_ocr_layers(page_meta, debug_out.get("restored"))
    out += _render_drawing_layers(page_meta, debug_out.get("drawing"))
    return out
