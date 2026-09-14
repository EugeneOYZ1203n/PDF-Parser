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
`OCR/fast_detect.py`, `pipelines/_steps.py` and others, so a heavy import
here would land in the real pipeline's import graph. Every stage module
keeps its one-line `from rastervec.commons.renderer.stages import render_x`
re-export.

The old Vector_Classification/pixel-Radon pipeline's own stage renderers
(`render_layers`, `render_layer_color_buckets` (nested-dict version),
`render_clustering_steps`, `render_text_candidates`, `render_radon`,
`render_similarity`, `render_restore`) moved to `archive/rastervec/
renderer/stages_old.py` when that pipeline was retired -- this module now
only covers the current pipeline's own stages.
"""
from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import pymupdf as fitz
from PIL import Image

from rastervec.commons.helpers.geometry import union_bbox
from rastervec.commons.renderer import render_reconstructed_pdf, render_text_pdf, render_vectors_pdf

if TYPE_CHECKING:
    from rastervec.commons.models import PageMeta
    from rastervec.pipelines.result import PipelineResult

# ---------------------------------------------------------------------------
# Palette -- also the single source of truth the viewer's colour legend reads.
# ---------------------------------------------------------------------------
C_NATIVE = "#1d4ed8"
C_DRAWING = "#111827"
C_CLUSTER_BBOX = "#ea580c"
C_RENDER_BBOX = "#2563eb"
C_DETECTION_BOX = "#16a34a"
C_ASSIGNED_TEXT = "#16a34a"
C_ASSIGNED_DRAWING = "#dc2626"
C_ROTATED_BBOX = "#0d9488"
C_FAST_PASS = "#059669"
C_FAST_DROP = "#dc2626"
C_TILE_SKIP = "#9ca3af"
C_TILE_GRID = "#6b7280"
C_RECLASS_PASS = "#059669"
C_OCR_PASS = "#059669"
C_OCR_FAIL = "#dc2626"
C_OCR_BOX = "#2563eb"
# New pluggable core.pipeline engine's generic phase1/phase2/final layers.
C_P2_VECTOR = "#7c3aed"
C_P2_TEXT = "#c026d3"
C_P3_TEXT = "#16a34a"

# Human-readable legend per generated stage PDF (filename -> [(label, hex)]).
STAGE_COLOR_LEGEND: dict[str, list[tuple[str, str]]] = {
    "native_text.pdf": [("native word", C_NATIVE)],
    "vector_extraction.pdf": [("(one colour per vector type)", "#888888")],
    "fast_heatmap.pdf": [
        ("text heatmap", "#dc2626"), ("skipped tile", C_TILE_SKIP), ("tile grid", C_TILE_GRID),
        ("kept vector", C_FAST_PASS), ("dropped vector", C_FAST_DROP),
    ],
    "similarity.pdf": [("(one colour per similarity group)", "#888888")],
    "reclassify.pdf": [("final pass", C_RECLASS_PASS)],
    "separation.pdf": [("(one colour per (layer, colour, width) bucket)", "#888888")],
    "clusters.pdf": [("cluster bbox", C_CLUSTER_BBOX)],
    "paddle_detect.pdf": [
        ("cluster render bbox", C_RENDER_BBOX), ("detected box", C_DETECTION_BOX),
    ],
    "assignment.pdf": [
        ("vector assigned to text", C_ASSIGNED_TEXT), ("vector reassigned to drawing", C_ASSIGNED_DRAWING),
    ],
    "rotate.pdf": [
        ("detection bbox", C_RENDER_BBOX), ("rotated crop bbox", C_ROTATED_BBOX),
    ],
    "paddle_ocr.pdf": [
        ("predicted text (ok)", C_OCR_PASS), ("predicted text (blank)", C_OCR_FAIL),
        ("OCR-detected box", C_OCR_BOX),
    ],
    "drawing_vectors.pdf": [("drawing vector", C_DRAWING)],
    "reconstructed.pdf": [("reconstructed page", "#111827")],
    "phase1.pdf": [("native word", C_NATIVE), ("raw vector", "#888888")],
    "phase2.pdf": [("phase2 vector", C_P2_VECTOR), ("phase2 text", C_P2_TEXT)],
    "final.pdf": [("final vector", C_DRAWING), ("final text", C_P3_TEXT)],
}

STAGE_ARTIFACTS = {
    "native": "native_text.pdf",
    "vectors": "vector_extraction.pdf",
    "similarity": "similarity.pdf",
    "fast": "fast_heatmap.pdf",
    "reclassify": "reclassify.pdf",
    "separation": "separation.pdf",
    "clusters": "clusters.pdf",
    "paddle_detect": "paddle_detect.pdf",
    "assignment": "assignment.pdf",
    "rotate": "rotate.pdf",
    "ocr": "paddle_ocr.pdf",
    "drawing": "drawing_vectors.pdf",
    "reconstructed": "reconstructed.pdf",
    # New pluggable core.pipeline engine (P2/P3 registry) -- a deliberately
    # small, generic artifact set (see render_stage_layers' "phase1"/
    # "phase2"/"final" branches below) since the three P3 backends share no
    # code and populate verbose intermediates differently.
    "phase1": "phase1.pdf",
    "phase2": "phase2.pdf",
    "final": "final.pdf",
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


def _meta(res: "PipelineResult", page_meta: "PageMeta | None") -> "PageMeta":
    return page_meta if page_meta is not None else res.page.meta


# ---------------------------------------------------------------------------
# _compose: several layers onto one page.
# ---------------------------------------------------------------------------
def _compose(
    page_meta: "PageMeta",
    *,
    image: "Image.Image | None" = None,
    vector_layers: "list[tuple[list, str]] | None" = None,
    rect_layers: "list[tuple[list, str, bool]] | None" = None,
    poly_layers: "list[tuple[list, str]] | None" = None,
    text_layer: "list[tuple[str, tuple, float, tuple]] | None" = None,
) -> bytes:
    """Build one page sized/rotated to `page_meta`, painting (in order): a
    full-page raster `image`; each `(vectors, hex)` in `vector_layers` as
    recoloured strokes; each `(bboxes, hex, filled)` in `rect_layers`; each
    `(polys, hex)` in `poly_layers` as closed, unfilled polylines (`polys`
    is a list of `(N, 2)` point sequences -- for a genuinely rotated
    rectangle, not just an axis-aligned bbox); then `text_layer` `(text,
    bbox, rotation, rgb)` tuples. Returns PDF bytes."""
    from rastervec.commons.renderer._shapes import replay_drawing_paths

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

        for polys, hexcol in poly_layers or []:
            rgb = _hex_to_rgb01(hexcol)
            for poly in polys:
                if poly is None or len(poly) < 2:
                    continue
                pts = [fitz.Point(float(px), float(py)) for px, py in poly]
                page.draw_polyline(pts + [pts[0]], color=rgb, width=1.0)

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
# Vector-level similarity grouping -- one colour per group (run before FAST)
# ---------------------------------------------------------------------------
def render_similarity(res: "PipelineResult", *, page_meta: "PageMeta | None" = None) -> bytes:
    groups = res.similarity_groups or []
    color_by_id: dict[int, tuple] = {}
    flat = []
    for gi, g in enumerate(groups):
        rgb = _hash_color(("similarity_group", gi))
        for v in g.members:
            color_by_id[id(v)] = rgb
            flat.append(v)
    return render_vectors_pdf(
        _meta(res, page_meta), flat,
        color_of=lambda v: color_by_id.get(id(v), (0.0, 0.0, 0.0)),
    )


# ---------------------------------------------------------------------------
# FAST text detection (per-Vector)
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
            (list(getattr(fr, "all_tiles", None) or []), C_TILE_GRID, False),
            ([union_bbox([v.bbox for v in c]) for c in passed if c], C_FAST_PASS, False),
            ([v.bbox for v in dropped], C_FAST_DROP, False),
        ],
    )


# ---------------------------------------------------------------------------
# Reclassify FAST's per-vector verdict up to a similarity-group consensus
# (fail -> pass only) -- shows just the final passed set.
# ---------------------------------------------------------------------------
def render_reclassify(res: "PipelineResult", *, page_meta: "PageMeta | None" = None) -> bytes:
    rc = res.reclassify_result
    final_passed = rc.passed if rc else []
    return _compose(
        _meta(res, page_meta),
        rect_layers=[([v.bbox for v in final_passed], C_RECLASS_PASS, False)],
    )


# ---------------------------------------------------------------------------
# (layer, colour, width) separation
# ---------------------------------------------------------------------------
def render_separation(res: "PipelineResult", *, page_meta: "PageMeta | None" = None) -> bytes:
    """`separation.pdf`: every vector coloured by its own flat
    `res.separation_buckets` entry (one bucket = one (layer, colour, width)
    group)."""
    buckets = res.separation_buckets or []
    color_by_id: dict[int, tuple] = {}
    flat = []
    for bi, bucket in enumerate(buckets):
        rgb = _hash_color(("bucket", bi))
        for v in bucket:
            color_by_id[id(v)] = rgb
            flat.append(v)
    return render_vectors_pdf(
        _meta(res, page_meta), flat,
        color_of=lambda v: color_by_id.get(id(v), (0.0, 0.0, 0.0)),
    )


# ---------------------------------------------------------------------------
# seqno-consecutive clusters
# ---------------------------------------------------------------------------
def render_clusters(res: "PipelineResult", *, page_meta: "PageMeta | None" = None) -> bytes:
    clusters = res.spatial_clusters or []
    bboxes = [union_bbox([v.bbox for v in c]) for c in clusters if c]
    return _compose(_meta(res, page_meta), rect_layers=[(bboxes, C_CLUSTER_BBOX, False)])


# ---------------------------------------------------------------------------
# Per-cluster PaddleOCR detect
# ---------------------------------------------------------------------------
def render_paddle_detect(res: "PipelineResult", *, page_meta: "PageMeta | None" = None) -> bytes:
    clusters = res.spatial_clusters or []
    cluster_detections = res.cluster_detections or []
    render_bboxes = [union_bbox([v.bbox for v in c]) for c in clusters if c]
    detection_boxes = [
        d.bbox for cd in cluster_detections if cd is not None for d in cd.detections
    ]
    return _compose(
        _meta(res, page_meta),
        rect_layers=[
            (render_bboxes, C_RENDER_BBOX, False),
            (detection_boxes, C_DETECTION_BOX, False),
        ],
    )


# ---------------------------------------------------------------------------
# Overlap-based text/drawing reassignment
# ---------------------------------------------------------------------------
def render_assignment(res: "PipelineResult", *, page_meta: "PageMeta | None" = None) -> bytes:
    text_vecs = res.reassigned_text or []
    drawing_vecs = res.reassigned_drawing or []
    return _compose(
        _meta(res, page_meta),
        vector_layers=[(text_vecs, C_ASSIGNED_TEXT), (drawing_vecs, C_ASSIGNED_DRAWING)],
    )


# ---------------------------------------------------------------------------
# Rotation refinement (vector-geometry Radon sweep) -- no splitting
# ---------------------------------------------------------------------------
def _rotated_crop_quad(seg) -> "list[tuple[float, float]] | None":
    """The segment's own tight bbox in its deskewed (`seg.angle`) frame,
    rotated back into page space -- a genuinely tilted quad matching the
    detected angle, not a plain axis-aligned `union_bbox` (which is what
    this used to draw despite the "rotated" name/colour)."""
    from rastervec.OCR.radon import cluster_centre, rotate_pts, rotated_segments

    if not seg.vectors:
        return None
    _, (x0, x1, y0, y1) = rotated_segments(seg.vectors, seg.angle)
    centre = cluster_centre(seg.vectors)
    corners = rotate_pts(
        np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], dtype=float), -seg.angle, centre,
    )
    return [(float(px), float(py)) for px, py in corners]


def render_rotate(res: "PipelineResult", *, page_meta: "PageMeta | None" = None) -> bytes:
    """`rotate.pdf`: each detection's own bbox, and the final rotated
    crop's own true (tilted) bbox, rotated to the detected angle."""
    dbg = res.rotation_debug or []
    detection_b = [d["detection_bbox"] for d in dbg]
    rotated_polys = [
        q for seg in (res.rotated_segments or []) if (q := _rotated_crop_quad(seg)) is not None
    ]
    return _compose(
        _meta(res, page_meta),
        rect_layers=[(detection_b, C_RENDER_BBOX, False)],
        poly_layers=[(rotated_polys, C_ROTATED_BBOX)],
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
# OCR
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
# toggles a layer by loading / not loading its file (no colour-keying, so no
# anti-alias fringe from a partially-knocked-out colour).
#
# `render_stage_layers(res, stage_key)` returns `[(label, hex, pdf_bytes)]`
# in draw order. The layer set for a stage is FIXED (an empty layer still
# emits a blank one-page PDF) so every layer PDF has the same page count as
# every other and the viewer can index them all by the same page position.
# The composite `render_<stage>` functions above are unchanged (still used by
# tests).
# ---------------------------------------------------------------------------
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

    if stage_key == "similarity":
        return [("vectors (colour per similarity group)", "#888888",
                 render_similarity(res, page_meta=pm))]

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
            ("tile grid", C_TILE_GRID, R(list(getattr(fr, "all_tiles", None) or []), C_TILE_GRID)),
            ("kept vector", C_FAST_PASS, R(passed, C_FAST_PASS)),
            ("dropped vector", C_FAST_DROP, R(dropped, C_FAST_DROP)),
        ]

    if stage_key == "reclassify":
        rc = res.reclassify_result
        final_passed = rc.passed if rc else []
        return [("final pass", C_RECLASS_PASS, R([v.bbox for v in final_passed], C_RECLASS_PASS))]

    if stage_key == "separation":
        return [("bucket (colour per layer+colour+width)", "#888888",
                 render_separation(res, page_meta=pm))]

    if stage_key == "clusters":
        clusters = res.spatial_clusters or []
        bboxes = [union_bbox([v.bbox for v in c]) for c in clusters if c]
        return [("cluster bbox", C_CLUSTER_BBOX, R(bboxes, C_CLUSTER_BBOX))]

    if stage_key == "paddle_detect":
        clusters = res.spatial_clusters or []
        cluster_detections = res.cluster_detections or []
        render_bboxes = [union_bbox([v.bbox for v in c]) for c in clusters if c]
        detection_boxes = [
            d.bbox for cd in cluster_detections if cd is not None for d in cd.detections
        ]
        return [
            ("cluster render bbox", C_RENDER_BBOX, R(render_bboxes, C_RENDER_BBOX)),
            ("detected box", C_DETECTION_BOX, R(detection_boxes, C_DETECTION_BOX)),
        ]

    if stage_key == "assignment":
        return [
            ("vector assigned to text", C_ASSIGNED_TEXT, V(res.reassigned_text or [], C_ASSIGNED_TEXT)),
            ("vector reassigned to drawing", C_ASSIGNED_DRAWING,
             V(res.reassigned_drawing or [], C_ASSIGNED_DRAWING)),
        ]

    if stage_key == "rotate":
        dbg = res.rotation_debug or []
        detection_b = [d["detection_bbox"] for d in dbg]
        rotated_polys = [
            q for seg in (res.rotated_segments or []) if (q := _rotated_crop_quad(seg)) is not None
        ]
        rotated_pdf = (
            _compose(pm, poly_layers=[(rotated_polys, C_ROTATED_BBOX)])
            if rotated_polys else _blank(pm)
        )
        return [
            ("detection bbox", C_RENDER_BBOX, R(detection_b, C_RENDER_BBOX)),
            ("rotated crop bbox", C_ROTATED_BBOX, rotated_pdf),
        ]

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

    # -----------------------------------------------------------------
    # New pluggable core.pipeline engine (P2/P3 registry). Deliberately
    # generic -- the three P3 backends share no code and populate `res.extra`
    # differently, so this renders only what every combination guarantees:
    # phase1's raw extraction, phase2's raw output, and the final result.
    # -----------------------------------------------------------------
    if stage_key == "phase1":
        phase1 = (res.extra or {}).get("phase1") if hasattr(res, "extra") else None
        native = list(getattr(phase1, "texts", None) or [])
        raw_vectors = list(getattr(phase1, "vectors", None) or [])
        return [
            ("native word", C_NATIVE,
             render_text_pdf(pm, native, color_of=lambda _t: _hex_to_rgb01(C_NATIVE))
             if native else _blank(pm)),
            ("raw vector", "#888888", V(raw_vectors, "#888888")),
        ]

    if stage_key == "phase2":
        extra = res.extra or {} if hasattr(res, "extra") else {}
        p2_vectors = list(extra.get("phase2_vectors") or [])
        p2_texts = list(extra.get("phase2_texts") or [])
        return [
            ("phase2 vector", C_P2_VECTOR, V(p2_vectors, C_P2_VECTOR)),
            ("phase2 text", C_P2_TEXT, _text_layer_pdf(pm, p2_texts, C_P2_TEXT)),
        ]

    if stage_key == "final":
        return [
            ("final vector", C_DRAWING, render_drawing(res, page_meta=pm)),
            ("final text", C_P3_TEXT, _text_layer_pdf(pm, res.texts or [], C_P3_TEXT)),
        ]

    raise ValueError(f"unknown stage_key {stage_key!r}")


def _text_layer_pdf(page_meta: "PageMeta", texts, hexcol: str) -> bytes:
    """One-page PDF of `texts` (any `Text` list) drawn via `_compose`'s
    generic `text_layer` -- used by the new pluggable engine's phase2/final
    layers, which don't distinguish native vs OCR provenance the way the old
    engine's `render_ocr_results`/`render_reconstructed` do."""
    rgb = _hex_to_rgb01(hexcol)
    layer = [(t.text or "", tuple(t.bbox), t.angle(), rgb) for t in texts]
    return _compose(page_meta, text_layer=layer) if layer else _blank(page_meta)
