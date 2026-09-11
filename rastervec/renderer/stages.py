"""One `render_<stage>` per pipeline stage, each returning a **one-page
stage PDF** (bytes) visualizing that stage's output off a
`run_pipeline(..., verbose=True)` `PipelineResult`.

`scripts/generate_pipeline_report.py` calls these; the real pipeline never
does. Every function builds on three shared primitives in
`rastervec/renderer/pdf.py` -- `render_text_pdf` (reconstruct text),
`render_vectors_pdf` (reconstruct vectors, recoloured), `render_boxes_pdf`
(bbox outlines) -- plus the local `_compose` helper here for the few stages
that need several of those layers on one page.

Kept matplotlib-free and free of any `rastervec.pipelines` import at module
level: this module is imported (for its name re-exports) by
`Vector_Classification/classification.py`, `OCR/fast_detect.py`,
`pipelines/_steps.py` and others, so a heavy import here would land in the
real pipeline's import graph. Every stage module keeps its one-line
`from rastervec.renderer.stages import render_x` re-export.
"""
from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import pymupdf as fitz
from PIL import Image

from rastervec.helpers.geometry import union_bbox
from rastervec.renderer import render_reconstructed_pdf, render_text_pdf, render_vectors_pdf

if TYPE_CHECKING:
    from rastervec.models import PageMeta
    from rastervec.pipelines.result import PipelineResult

# ---------------------------------------------------------------------------
# Palette -- also the single source of truth the viewer's colour legend reads.
# ---------------------------------------------------------------------------
C_NATIVE = "#1d4ed8"
C_DRAWING = "#111827"
C_TEXT_CANDIDATE = "#059669"
C_GROUP_BBOX = "#7c3aed"
C_CLUSTER_BBOX = "#ea580c"
C_ORIG_BBOX = "#2563eb"
C_SEGMENT_BBOX = "#059669"
C_LINE_GAP = "#dc2626"
C_WORD_GAP = "#f59e0b"
C_GROWN_BBOX = "#0d9488"
C_VEC_ASSIGNED = "#16a34a"
C_VEC_DROPPED = "#dc2626"
C_SEG_DROPPED = "#b91c1c"
C_FAST_PASS = "#059669"
C_FAST_DROP = "#dc2626"
C_TILE_SKIP = "#9ca3af"
C_OCR_PASS = "#059669"
C_OCR_FAIL = "#dc2626"
C_OCR_BOX = "#2563eb"

_DROP_CATEGORIES = [
    "dropped_oversized", "duplicate_runs", "dropped_tiny", "dropped_mixed_fill_rule",
    "dropped_perimeter", "dropped_low_density", "dropped_constant_spacing",
    "dropped_low_variety",
]

# Human-readable legend per generated stage PDF (filename -> [(label, hex)]).
STAGE_COLOR_LEGEND: dict[str, list[tuple[str, str]]] = {
    "native_text.pdf": [("native word", C_NATIVE)],
    "vector_extraction.pdf": [("(one colour per vector type)", "#888888")],
    "separation.pdf": [("(one colour per (layer, colour) bucket)", "#888888")],
    "vector_classification.pdf": [],  # filled in below, once _drop_color exists
    "fast_heatmap.pdf": [
        ("text heatmap", "#dc2626"), ("skipped tile", C_TILE_SKIP),
        ("passed cluster", C_FAST_PASS), ("dropped vector", C_FAST_DROP),
    ],
    "segmentation.pdf": [
        ("original cluster bbox", C_ORIG_BBOX), ("segment bbox", C_SEGMENT_BBOX),
        ("post-growth segment bbox", C_GROWN_BBOX), ("line gap", C_LINE_GAP),
        ("word gap", C_WORD_GAP), ("vectors assigned", C_VEC_ASSIGNED),
        ("vectors dropped", C_VEC_DROPPED), ("segments dropped", C_SEG_DROPPED),
    ],
    "similarity.pdf": [("(one colour per similarity group)", "#888888")],
    "paddle_ocr.pdf": [
        ("predicted text (ok)", C_OCR_PASS), ("predicted text (blank)", C_OCR_FAIL),
        ("OCR-detected box", C_OCR_BOX),
    ],
    "drawing_vectors.pdf": [("drawing vector", C_DRAWING)],
    "reconstructed.pdf": [("reconstructed page", "#111827")],
}

STAGE_ARTIFACTS = {
    "native": "native_text.pdf",
    "vectors": "vector_extraction.pdf",
    "separation": "separation.pdf",
    "classify": "vector_classification.pdf",
    "fast": "fast_heatmap.pdf",
    "segment": "segmentation.pdf",
    "similarity": "similarity.pdf",
    "ocr": "paddle_ocr.pdf",
    "drawing": "drawing_vectors.pdf",
    "reconstructed": "reconstructed.pdf",
}


def _hex_to_rgb01(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def _hash_color(key) -> tuple[float, float, float]:
    import colorsys
    import hashlib

    digest = hashlib.md5(repr(key).encode()).hexdigest()
    hue = (int(digest[:8], 16) % 360) / 360.0
    return colorsys.hsv_to_rgb(hue, 0.62, 0.85)


def _drop_color(name: str) -> str:
    r, g, b = _hash_color(("drop", name))
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


STAGE_COLOR_LEGEND["vector_classification.pdf"] = (
    [("text-candidate vectors", C_TEXT_CANDIDATE), ("group bbox", C_GROUP_BBOX),
     ("cluster bbox", C_CLUSTER_BBOX)]
    + [(name, _drop_color(name)) for name in _DROP_CATEGORIES]
)


def _meta(res: "PipelineResult", page_meta: "PageMeta | None") -> "PageMeta":
    return page_meta if page_meta is not None else res.page.meta


def _entry_vectors(entry: list) -> list:
    """Flatten one classification category entry (a group `list[Vector]` or a
    cluster `list[list[Vector]]`) to a flat `list[Vector]`."""
    if entry and isinstance(entry[0], list):
        return [v for g in entry for v in g]
    return entry


# ---------------------------------------------------------------------------
# _compose: several layers onto one page.
# ---------------------------------------------------------------------------
def _compose(
    page_meta: "PageMeta",
    *,
    image: "Image.Image | None" = None,
    vector_layers: "list[tuple[list, str]] | None" = None,
    rect_layers: "list[tuple[list, str, bool]] | None" = None,
    text_layer: "list[tuple[str, tuple, float, tuple]] | None" = None,
) -> bytes:
    """Build one page sized/rotated to `page_meta`, painting (in order): a
    full-page raster `image`; each `(vectors, hex)` in `vector_layers` as
    recoloured strokes; each `(bboxes, hex, filled)` in `rect_layers`; then
    `text_layer` `(text, bbox, rotation, rgb)` tuples. Returns PDF bytes."""
    from rastervec.renderer._shapes import replay_drawing_paths

    doc = fitz.open()
    try:
        page = doc.new_page(width=page_meta.width, height=page_meta.height)
        page.set_rotation(page_meta.rotation)

        if image is not None:
            buf = io.BytesIO()
            image.convert("RGB").save(buf, format="PNG")
            page.insert_image(
                fitz.Rect(0, 0, page_meta.width, page_meta.height),
                stream=buf.getvalue(), keep_proportion=False,
            )

        import dataclasses

        for vectors, hexcol in vector_layers or []:
            rgb = _hex_to_rgb01(hexcol)
            recol = [
                dataclasses.replace(
                    v, type="s", color=rgb, fill=None, dashes=None,
                    width=max(v.width or 0.0, 1.0), closePath=False,
                    blendmode="Normal", opacity=1.0, stroke_opacity=1.0, fill_opacity=None,
                )
                for v in vectors
            ]
            if recol:
                replay_drawing_paths(page, recol)

        for bboxes, hexcol, filled in rect_layers or []:
            rgb = _hex_to_rgb01(hexcol)
            for b in bboxes:
                if b is None:
                    continue
                kw = {"color": rgb, "width": 1.0}
                if filled:
                    kw["fill"] = rgb
                    kw["fill_opacity"] = 0.25
                page.draw_rect(fitz.Rect(*b), **kw)

        if text_layer:
            base_font = fitz.Font("helv")
            span = base_font.ascender - base_font.descender
            for text, bbox, rotation, rgb in text_layer:
                if not text or not text.strip():
                    continue
                x0, y0, x1, y1 = bbox
                fs = max((y1 - y0) / span, 1.0)
                natural = base_font.text_length(text, fontsize=fs)
                if natural > (x1 - x0) > 0:
                    fs = max(fs * (x1 - x0) / natural, 1.0)
                try:
                    page.insert_text(
                        fitz.Point(x0, y0 + base_font.ascender * fs), text,
                        fontsize=fs, color=rgb,
                    )
                except Exception:  # noqa: BLE001 -- a preview; never fail the run over one label
                    pass

        return doc.tobytes()
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# Native text
# ---------------------------------------------------------------------------
def render_native(res: "PipelineResult", *, page_meta: "PageMeta | None" = None) -> bytes:
    return render_text_pdf(
        _meta(res, page_meta), res.native_words or [],
        color_of=lambda _t: _hex_to_rgb01(C_NATIVE),
    )


# ---------------------------------------------------------------------------
# Vector extraction -- one colour per Vector.type
# ---------------------------------------------------------------------------
def render_vectors(res: "PipelineResult", *, page_meta: "PageMeta | None" = None) -> bytes:
    vectors = res.vectors_raw or []
    return render_vectors_pdf(
        _meta(res, page_meta), vectors,
        color_of=lambda v: _hash_color(("type", v.type)),
    )


# ---------------------------------------------------------------------------
# Layer / (layer, colour) separation
# ---------------------------------------------------------------------------
def render_layers(res: "PipelineResult", *, page_meta: "PageMeta | None" = None) -> bytes:
    vbl = res.vectors_by_layer or {}
    color_by_id = {
        id(v): _hash_color(("layer", name))
        for name, vs in vbl.items() for v in vs
    }
    flat = [v for vs in vbl.values() for v in vs]
    return render_vectors_pdf(
        _meta(res, page_meta), flat,
        color_of=lambda v: color_by_id.get(id(v), (0.0, 0.0, 0.0)),
    )


def render_layer_color_buckets(
    res: "PipelineResult", *, page_meta: "PageMeta | None" = None
) -> bytes:
    """`separation.pdf`: every vector coloured by its (layer, colour) bucket."""
    vblc = res.vectors_by_layer_color or {}
    color_by_id: dict[int, tuple] = {}
    flat = []
    for layer, by_color in vblc.items():
        for color, vs in by_color.items():
            rgb = _hash_color(("bucket", layer, color))
            for v in vs:
                color_by_id[id(v)] = rgb
                flat.append(v)
    return render_vectors_pdf(
        _meta(res, page_meta), flat,
        color_of=lambda v: color_by_id.get(id(v), (0.0, 0.0, 0.0)),
    )


# ---------------------------------------------------------------------------
# Vector classification
# ---------------------------------------------------------------------------
def _classification_groups_and_clusters(res: "PipelineResult"):
    """`(group_bboxes, cluster_bboxes)` in page space from `res.clustering`."""
    group_bboxes: list = []
    cluster_bboxes: list = []
    for stage in (res.clustering or {}).values():
        steps = stage.steps
        spatial_i = next(
            (i for i, s in enumerate(steps) if s.label.lower().startswith("spatial")), None
        )
        if spatial_i is not None and spatial_i > 0:
            for entry in steps[spatial_i - 1].categories["kept"].groups:
                vs = _entry_vectors(entry)
                if vs:
                    group_bboxes.append(union_bbox([v.bbox for v in vs]))
        if steps:
            for entry in steps[-1].categories["kept"].groups:
                vs = _entry_vectors(entry)
                if vs:
                    cluster_bboxes.append(union_bbox([v.bbox for v in vs]))
    return group_bboxes, cluster_bboxes


def render_clustering_steps(
    res: "PipelineResult", *, page_meta: "PageMeta | None" = None
) -> bytes:
    """`vector_classification.pdf`: every filter-dropped vector coloured by
    which filter dropped it, the surviving text-candidate vectors in one
    colour, and group / cluster bboxes."""
    drop_layers: list[tuple[list, str]] = []
    for name in _DROP_CATEGORIES:
        vs: list = []
        for stage in (res.clustering or {}).values():
            for step in stage.steps:
                cat = step.categories.get(name)
                if cat is not None and cat.role == "dropped":
                    for entry in cat.groups:
                        vs.extend(_entry_vectors(entry))
        if vs:
            drop_layers.append((vs, _drop_color(name)))

    candidates = [v for c in (res.text_clusters or []) for v in _entry_vectors(c)]
    if candidates:
        drop_layers.append((candidates, C_TEXT_CANDIDATE))

    groups, clusters = _classification_groups_and_clusters(res)
    return _compose(
        _meta(res, page_meta),
        vector_layers=drop_layers,
        rect_layers=[(groups, C_GROUP_BBOX, False), (clusters, C_CLUSTER_BBOX, False)],
    )


def render_text_candidates(
    res: "PipelineResult", *, page_meta: "PageMeta | None" = None
) -> bytes:
    clusters = [_entry_vectors(c) for c in (res.text_clusters or [])]
    bboxes = [union_bbox([v.bbox for v in c]) for c in clusters if c]
    return _compose(
        _meta(res, page_meta),
        vector_layers=[([v for c in clusters for v in c], C_TEXT_CANDIDATE)],
        rect_layers=[(bboxes, C_CLUSTER_BBOX, False)],
    )


# ---------------------------------------------------------------------------
# Segment (Radon)
# ---------------------------------------------------------------------------
def render_radon(res: "PipelineResult", *, page_meta: "PageMeta | None" = None) -> bytes:
    """`segmentation.pdf`: original cluster bbox, per-word segment bbox, and
    the detected line / word gap markers, from `res.segmentation_debug`."""
    dbg = res.segmentation_debug or []
    cluster_b = [d["cluster_bbox"] for d in dbg]
    seg_b = [b for d in dbg for b in d["segment_bboxes"]]
    grown_b = [b for d in dbg for b in d.get("grown_segment_bboxes", [])]
    line_g = [b for d in dbg for b in d["line_gap_lines"]]
    word_g = [b for d in dbg for b in d["word_gap_lines"]]
    vec_ok = [b for d in dbg for b in d.get("assigned_vector_bboxes", [])]
    vec_drop = [b for d in dbg for b in d.get("dropped_vector_bboxes", [])]
    seg_drop = [b for d in dbg for b in d.get("dropped_segment_bboxes", [])]
    return _compose(
        _meta(res, page_meta),
        rect_layers=[
            (cluster_b, C_ORIG_BBOX, False),
            (seg_b, C_SEGMENT_BBOX, False),
            (grown_b, C_GROWN_BBOX, False),
            (line_g, C_LINE_GAP, True),
            (word_g, C_WORD_GAP, True),
            (vec_ok, C_VEC_ASSIGNED, False),
            (vec_drop, C_VEC_DROPPED, False),
            (seg_drop, C_SEG_DROPPED, True),
        ],
    )


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------
def render_similarity(res: "PipelineResult", *, page_meta: "PageMeta | None" = None) -> bytes:
    segments = res.word_segments or []
    groups = res.similarity_groups or [[i] for i in range(len(segments))]
    rect_layers: list = []
    for gi, group in enumerate(groups):
        r, g, b = _hash_color(("simgroup", gi))
        hexc = "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))
        bboxes = [
            union_bbox([v.bbox for v in segments[i].vectors])
            for i in group if segments[i].vectors
        ]
        if bboxes:
            rect_layers.append((bboxes, hexc, False))
    return _compose(_meta(res, page_meta), rect_layers=rect_layers)


# ---------------------------------------------------------------------------
# FAST text detection
# ---------------------------------------------------------------------------
def _fast_heat_image(page_image: "Image.Image", mask: "np.ndarray | None") -> "Image.Image":
    base = page_image.convert("RGB")
    if mask is None:
        return base
    heat = (np.clip(mask, 0.0, 1.0) * 255).astype("uint8")
    zeros = Image.new("L", base.size, 0)
    heat_img = Image.merge("RGB", (Image.fromarray(heat).resize(base.size), zeros, zeros))
    return Image.blend(base, heat_img, alpha=0.5)


def render_fast(
    res: "PipelineResult", *, enable_fast: bool = True, page_meta: "PageMeta | None" = None
) -> bytes:
    pm = _meta(res, page_meta)
    fr = res.fast_result
    passed = res.fast_passed or []
    dropped = res.fast_dropped_vectors or []
    image = None
    if fr is not None and fr.page_image is not None:
        image = _fast_heat_image(fr.page_image, fr.page_mask)
    return _compose(
        pm,
        image=image,
        rect_layers=[
            (list(getattr(fr, "skipped_tiles", None) or []), C_TILE_SKIP, True),
            ([union_bbox([v.bbox for v in c]) for c in passed if c], C_FAST_PASS, False),
            ([v.bbox for v in dropped], C_FAST_DROP, False),
        ],
    )


# ---------------------------------------------------------------------------
# Drawing output
# ---------------------------------------------------------------------------
def render_drawing(res: "PipelineResult", *, page_meta: "PageMeta | None" = None) -> bytes:
    return render_vectors_pdf(
        _meta(res, page_meta), res.vectors or [],
        color_of=lambda _v: _hex_to_rgb01(C_DRAWING),
    )


# ---------------------------------------------------------------------------
# OCR / restore
# ---------------------------------------------------------------------------
def render_ocr_results(res: "PipelineResult", *, page_meta: "PageMeta | None" = None) -> bytes:
    """`paddle_ocr.pdf`: predicted text drawn at each restored word position
    (green = read, red = blank) plus the OCR-detected boxes."""
    restored = res.restored_texts or []
    text_layer = [
        (t.text or "", tuple(t.bbox), t.angle(),
         _hex_to_rgb01(C_OCR_PASS if t.text.strip() else C_OCR_FAIL))
        for t in restored
    ]
    return _compose(
        _meta(res, page_meta),
        rect_layers=[([tuple(t.bbox) for t in restored], C_OCR_BOX, False)],
        text_layer=text_layer,
    )


def render_restore(res: "PipelineResult", *, page_meta: "PageMeta | None" = None) -> bytes:
    return render_ocr_results(res, page_meta=page_meta)


# ---------------------------------------------------------------------------
# Final reconstruction
# ---------------------------------------------------------------------------
def render_reconstructed(res: "PipelineResult", *, page_meta: "PageMeta | None" = None) -> bytes:
    native = [t for t in (res.texts or []) if t.source == "native"]
    ocr = [t for t in (res.texts or []) if t.source == "ocr"]
    return render_reconstructed_pdf(
        _meta(res, page_meta),
        native_words=native, drawing_vectors=res.vectors or [], ocr_results=ocr,
    )


# ---------------------------------------------------------------------------
# Per-layer split -- one single-purpose PDF per visual element, so the viewer
# toggles a layer by loading / not loading its PDF (no colour-keying, so no
# anti-alias fringe from a partially-knocked-out colour).
#
# `render_stage_layers(res, stage_key)` returns `[(label, hex, pdf_bytes)]`
# in draw order. The layer set for a stage is FIXED (an empty layer still
# emits a blank one-page PDF) so every layer PDF has the same page count as
# every other and the viewer can index them all by the same page position.
# The composite `render_<stage>` functions above are unchanged (still used by
# tests / the notebook).
# ---------------------------------------------------------------------------
def _drop_vectors(res: "PipelineResult", name: str) -> list:
    vs: list = []
    for stage in (res.clustering or {}).values():
        for step in stage.steps:
            cat = step.categories.get(name)
            if cat is not None and cat.role == "dropped":
                for entry in cat.groups:
                    vs.extend(_entry_vectors(entry))
    return vs


def _blank(page_meta: "PageMeta") -> bytes:
    return _compose(page_meta)


def render_stage_layers(
    res: "PipelineResult", stage_key: str, *, page_meta: "PageMeta | None" = None
) -> "list[tuple[str, str, bytes]]":
    pm = _meta(res, page_meta)
    V = lambda vs, hx: _compose(pm, vector_layers=[(vs, hx)]) if vs else _blank(pm)  # noqa: E731
    R = lambda bx, hx, fill=False: _compose(pm, rect_layers=[(bx, hx, fill)]) if bx else _blank(pm)  # noqa: E731

    if stage_key == "native":
        return [("native word", C_NATIVE, render_native(res, page_meta=pm))]

    if stage_key == "vectors":
        return [("vectors (colour per type)", "#888888", render_vectors(res, page_meta=pm))]

    if stage_key == "separation":
        return [("bucket (colour per layer+colour)", "#888888",
                 render_layer_color_buckets(res, page_meta=pm))]

    if stage_key == "classify":
        groups, clusters = _classification_groups_and_clusters(res)
        cand = [v for c in (res.text_clusters or []) for v in _entry_vectors(c)]
        out = [
            ("text candidate", C_TEXT_CANDIDATE, V(cand, C_TEXT_CANDIDATE)),
            ("group bbox", C_GROUP_BBOX, R(groups, C_GROUP_BBOX)),
            ("cluster bbox", C_CLUSTER_BBOX, R(clusters, C_CLUSTER_BBOX)),
        ]
        for name in _DROP_CATEGORIES:
            out.append((name, _drop_color(name), V(_drop_vectors(res, name), _drop_color(name))))
        return out

    if stage_key == "fast":
        fr = res.fast_result
        heat = None
        if fr is not None and fr.page_image is not None:
            heat = _compose(pm, image=_fast_heat_image(fr.page_image, fr.page_mask))
        passed = [union_bbox([v.bbox for v in c]) for c in (res.fast_passed or []) if c]
        dropped = [v.bbox for v in (res.fast_dropped_vectors or [])]
        return [
            ("text heatmap", "#dc2626", heat if heat is not None else _blank(pm)),
            ("skipped tile", C_TILE_SKIP,
             R(list(getattr(fr, "skipped_tiles", None) or []), C_TILE_SKIP, True)),
            ("passed cluster", C_FAST_PASS, R(passed, C_FAST_PASS)),
            ("dropped vector", C_FAST_DROP, R(dropped, C_FAST_DROP)),
        ]

    if stage_key == "segment":
        dbg = res.segmentation_debug or []
        return [
            ("original cluster bbox", C_ORIG_BBOX,
             R([d["cluster_bbox"] for d in dbg], C_ORIG_BBOX)),
            ("segment bbox", C_SEGMENT_BBOX,
             R([b for d in dbg for b in d["segment_bboxes"]], C_SEGMENT_BBOX)),
            ("post-growth segment bbox", C_GROWN_BBOX,
             R([b for d in dbg for b in d.get("grown_segment_bboxes", [])], C_GROWN_BBOX)),
            ("line gap", C_LINE_GAP,
             R([b for d in dbg for b in d["line_gap_lines"]], C_LINE_GAP, True)),
            ("word gap", C_WORD_GAP,
             R([b for d in dbg for b in d["word_gap_lines"]], C_WORD_GAP, True)),
            ("vectors assigned", C_VEC_ASSIGNED,
             R([b for d in dbg for b in d.get("assigned_vector_bboxes", [])], C_VEC_ASSIGNED)),
            ("vectors dropped", C_VEC_DROPPED,
             R([b for d in dbg for b in d.get("dropped_vector_bboxes", [])], C_VEC_DROPPED)),
            ("segments dropped", C_SEG_DROPPED,
             R([b for d in dbg for b in d.get("dropped_segment_bboxes", [])], C_SEG_DROPPED, True)),
        ]

    if stage_key == "similarity":
        return [("similarity group (colour per group)", "#888888",
                 render_similarity(res, page_meta=pm))]

    if stage_key == "ocr":
        restored = res.restored_texts or []
        ok = [t for t in restored if (t.text or "").strip()]
        blank = [t for t in restored if not (t.text or "").strip()]
        ok_text = [
            (t.text, tuple(t.bbox), t.angle(), _hex_to_rgb01(C_OCR_PASS)) for t in ok
        ]
        return [
            ("recognised text", C_OCR_PASS,
             _compose(pm, text_layer=ok_text) if ok_text else _blank(pm)),
            ("blank-read box", C_OCR_FAIL, R([tuple(t.bbox) for t in blank], C_OCR_FAIL)),
            ("OCR box", C_OCR_BOX, R([tuple(t.bbox) for t in restored], C_OCR_BOX)),
        ]

    if stage_key == "drawing":
        return [("drawing vector", C_DRAWING, render_drawing(res, page_meta=pm))]

    if stage_key == "reconstructed":
        return [("reconstructed page", "#111827", render_reconstructed(res, page_meta=pm))]

    raise ValueError(f"unknown stage_key {stage_key!r}")
