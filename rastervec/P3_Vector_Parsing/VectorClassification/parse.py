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


def parse(
    vectors_p1: list[Vector], vectors_p2: list[Vector], page: Page,
    *, enable_fast: bool = True, verbose: bool = False, compute=None, progress_counter=None,
    debug_out: "dict | None" = None,
) -> tuple[list[Vector], list[Text]]:
    """Combines Phase 1's raw native vectors and Phase 2's raster-derived
    vectors into one flat pool, then runs the old chain: classify -> FAST
    filter -> Radon word-segmentation -> similarity dedup -> OCR recognize
    -> restore each word occurrence's text -> merge every dropped Vector as
    drawing content. When `debug_out` is given (a plain dict), this stage's
    own intermediate objects are stashed into it verbatim (no shape
    conversion) for `render_debug` to read back and visualise -- see that
    function below."""
    all_vectors = list(vectors_p1) + list(vectors_p2)

    cls = classify_vectors(all_vectors, page, verbose=verbose)

    flat_clusters = [
        [v for group in cluster for v in group] for cluster in cls.text_clusters
    ]
    fast = detect_text_fast(
        flat_clusters, page, enable_fast=enable_fast, verbose=verbose,
        compute=compute, progress_counter=progress_counter,
    )

    word_segments = segment_clusters(fast.passed)

    groups = group_similar_segments(word_segments)
    uniques, metas = elect_unique_segments(word_segments, groups)

    unique_texts = recognize_unique_words(uniques, compute=compute, progress_counter=progress_counter)
    restored = restore_word_texts(unique_texts, metas)

    drawing = list(cls.drawing_vectors) + list(fast.dropped_vectors)

    if debug_out is not None:
        debug_out["classification"] = cls
        debug_out["fast_passed"] = fast.passed
        debug_out["fast_dropped"] = fast.dropped_vectors
        debug_out["word_segments"] = word_segments
        debug_out["restored"] = restored
        debug_out["drawing"] = drawing

    return drawing, restored


# ---------------------------------------------------------------------------
# Debug rendering -- this backend's own render function over its own
# `debug_out` shape, built only from the three generic primitives in
# `commons/renderer` (render_boxes_pdf/render_text_pdf/render_vectors_pdf).
# Nothing here is shared with FastIntoPaddle/LegacyRecreation/Junction.
# ---------------------------------------------------------------------------
_C_KEPT = "#059669"
_C_DROPPED = "#dc2626"
_C_FAST_PASS = "#059669"
_C_FAST_DROP = "#dc2626"
_C_SEGMENT = "#2563eb"
_C_OCR = "#16a34a"
_C_DRAWING = "#111827"


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


def render_debug(debug_out: "dict | None", page_meta) -> "list[tuple[str, str, str, bytes]]":
    """One (stage, label, hex, pdf_bytes) tuple per debug layer, built
    straight from `parse()`'s own `debug_out` stash. Called by
    `generate_pipeline_report.py` after a `verbose=True` run."""
    from rastervec.commons.helpers.geometry import union_bbox
    from rastervec.commons.renderer import render_boxes_pdf, render_text_pdf, render_vectors_pdf

    out: "list[tuple[str, str, str, bytes]]" = []
    if not debug_out:
        return out

    cls = debug_out.get("classification")
    if cls is not None and cls.clustering:
        steps_per_bucket = [stage.steps for stage in cls.clustering.values() if stage.steps]
        if steps_per_bucket:
            n_steps = len(steps_per_bucket[0])
            for i in range(n_steps):
                label = steps_per_bucket[0][i].label
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

    fast_passed = debug_out.get("fast_passed") or []
    fast_dropped = debug_out.get("fast_dropped") or []
    passed_boxes = [union_bbox([v.bbox for v in c]) for c in fast_passed if c]
    dropped_boxes = [v.bbox for v in fast_dropped]
    out.append(("fast", "passed", _C_FAST_PASS, render_boxes_pdf(
        page_meta, [(b, _hex_rgb(_C_FAST_PASS)) for b in passed_boxes],
    )))
    out.append(("fast", "dropped", _C_FAST_DROP, render_boxes_pdf(
        page_meta, [(b, _hex_rgb(_C_FAST_DROP)) for b in dropped_boxes],
    )))

    segs = debug_out.get("word_segments") or []
    seg_boxes = [union_bbox([v.bbox for v in s.vectors]) for s in segs if s.vectors]
    out.append(("segment", "word boxes", _C_SEGMENT, render_boxes_pdf(
        page_meta, [(b, _hex_rgb(_C_SEGMENT)) for b in seg_boxes],
    )))

    restored = debug_out.get("restored") or []
    out.append(("ocr", "recognized text", _C_OCR, render_text_pdf(
        page_meta, restored, color_of=lambda _t: _hex_rgb(_C_OCR),
    )))

    drawing = debug_out.get("drawing") or []
    out.append(("drawing", "drawing vectors", _C_DRAWING, render_vectors_pdf(
        page_meta, drawing, color_of=lambda _v: _hex_rgb(_C_DRAWING),
    )))

    return out
