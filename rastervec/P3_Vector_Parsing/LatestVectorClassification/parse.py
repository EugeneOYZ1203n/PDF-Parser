"""LatestVectorClassification's Phase3Backend entrypoint -- the merged
VectorClassification + CollinearVectorClass backend:

1. `classify_vectors` -- layer/color/width buckets; collinear drawing
   removal (long, regular dashed/repeated lines); pattern-lattice removal
   (similar Vectors repeated on a regular lattice); seqno merge + spatial
   clustering; per cluster, parallel-group length outliers and the
   crossed-grid removal -> drawing. Also the page's global potential
   angles (every >= 2-member collinear group's angle, page-wide, deduped).
2. Per cluster: render (unrotated), PaddleOCR detect.
3. Per detected quad: the quad's **long-edge angle**, snapped to the
   nearest global potential angle when within `QUAD_ANGLE_SNAP_TOL_DEG`
   (`line_geometry.snap_angle`; unchanged otherwise), sets the rotation;
   the region around the quad is rotated by it and the quad cropped out
   upright (`paddle_engine.upright_crop` -- a real rotation, never a
   re-boxed axis-aligned bbox). No Hough, no minAreaRect. The quad itself
   (`Text.quad_points`) stays as detected.
4. Page-wide batched recognition: PaddleOCR's 0/180 angle classifier, then
   recognise; every crop whose score (confidence, halved for a single
   character, 0 when blank) is below `RETRY_CONFIDENCE_THRESHOLD` is also
   recognised at +90/180/270 (no classifier) and the best attempt wins --
   compared on score x `ENGLISH_WORD_MULTIPLIER` ** (dictionary English
   words in the read, `english_words.selection_score`). Text angle = (snapped)
   long-edge angle + classifier flip + retry rotation.
5. Text/drawing split (`_text_vectors_by_quad`): a vector of an OCR'd
   cluster is text when some non-blank quad (the rotated quad itself)
   detected in its own cluster owns it -- tiered, cheapest first: bbox
   fully inside the quad -> owned; bbox area larger than the quad -> not;
   fewer than `TEXT_SEGMENT_OVERLAP_FRAC` of its pieces touching the quad
   -> not; else more than `TEXT_INK_INSIDE_FRAC` of its ink inside.
   Drawing = everything classification dropped + every other OCR'd-cluster
   vector. No FAST, nothing transitive.

Fully self-contained (own paddle_engine.py/config.py/line geometry) --
imports nothing from a sibling P3 backend or P2.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from rastervec.commons.helpers.geometry import (
    bboxes_intersect, compute_origin, transform_direction, union_bbox,
)
from rastervec.commons.models import Page, Vector, Text
from rastervec.commons.renderer import ocr_prep, pixel_to_page_bbox, pixel_to_page_points
from rastervec.commons.step_timing import StepClock
from rastervec.P3_Vector_Parsing.LatestVectorClassification.classify_vectors import (
    CROSSED_CATEGORY,
    FLAGGED_KEPT_CATEGORY,
    PATTERN_CATEGORY,
    STEP_LABELS,
    classify_vectors,
)
from rastervec.P3_Vector_Parsing.LatestVectorClassification.config import (
    ANGLE_TOL_DEG,
    COLLINEAR_OFFSET_TOL_PT,
    DETECT_RENDER_CHUNK_SIZE,
    INK_CURVE_SAMPLES,
    MAX_RENDER_DPI,
    MIN_PARALLEL_GROUP_SIZE,
    MIN_RENDER_SIDE_PX,
    OCR_BATCH_SIZE,
    OCR_DPI,
    OCR_LANG,
    OCR_VERSION,
    QUAD_ANGLE_SNAP_TOL_DEG,
    RENDER_PADDING_EXTRA_PT,
    RETRY_CONFIDENCE_THRESHOLD,
    STRAIGHT_TOL_PT,
    TEXT_INK_INSIDE_FRAC,
    TEXT_SEGMENT_OVERLAP_FRAC,
)
from rastervec.P3_Vector_Parsing.LatestVectorClassification.english_words import (
    selection_score,
)
from rastervec.P3_Vector_Parsing.LatestVectorClassification.line_geometry import (
    bbox_inside_quad,
    golden_hues,
    group_collinear,
    group_parallel,
    ink_fraction_in_quad,
    piece_overlap_fraction,
    quad_area,
    snap_angle,
    split_straight,
)
from rastervec.P3_Vector_Parsing.LatestVectorClassification.paddle_engine import (
    PaddleDetectBackend,
    PaddleRecBackend,
    _detect_job,
    _normalize_bgr,
    _recognize_crops_job,
    _recognize_crops_raw_job,
    quad_long_edge_angle,
    quad_region,
    reorder_quad_reading,
    score,
    upright_crop,
)

STEP_NAMES = ["classify", "ocr", "drawing"]

DebugLayer = "tuple[str, str, str, bytes]"
OnDebugLayer = "Callable[[str, str, str, bytes], None]"


def _cluster_render_padding(vectors: list[Vector]) -> float:
    """Page-space margin for a cluster's OCR render frame -- half its max
    stroke width plus `RENDER_PADDING_EXTRA_PT`."""
    return max((v.width or 0.0) for v in vectors) / 2.0 + RENDER_PADDING_EXTRA_PT


def _quad_envelope(quad) -> tuple[float, float, float, float]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return (min(xs), min(ys), max(xs), max(ys))


def _quad_owns(v: Vector, quad, env, area: float) -> bool:
    """Whether the detect `quad` (envelope `env`, area `area`) owns `v` --
    tiered, cheapest test first, the first decisive one wins:

    0. bbox misses the envelope -> no.
    1. bbox fully inside the quad -> yes.
    2. bbox area larger than the quad's -> no.
    3. fewer than `TEXT_SEGMENT_OVERLAP_FRAC` of `v`'s pieces touch the quad
       (`line_geometry.piece_overlap_fraction`, curves as chords) -> no;
       otherwise more than `TEXT_INK_INSIDE_FRAC` of its ink (path length)
       inside the quad (`line_geometry.ink_fraction_in_quad`). A vector with
       no pieces goes straight to the ink test (its bbox-centre fallback).
    """
    if not bboxes_intersect(v.bbox, env):
        return False
    if bbox_inside_quad(v.bbox, quad):
        return True
    x0, y0, x1, y1 = v.bbox
    if (x1 - x0) * (y1 - y0) > area:
        return False
    if piece_overlap_fraction(v, quad) < TEXT_SEGMENT_OVERLAP_FRAC:  # nan -> False
        return False
    return ink_fraction_in_quad(v, quad, curve_samples=INK_CURVE_SAMPLES) > TEXT_INK_INSIDE_FRAC


def _text_vectors_by_quad(
    clusters: list[list[Vector]], quads_by_cluster: dict[int, list[tuple]],
) -> tuple[list[Vector], list[Vector]]:
    """`(text, drawing)` over every vector of `clusters`: a vector is text
    when some quad in `quads_by_cluster[its cluster index]` (only non-blank
    recognitions are passed in -- the rotated quad polygon, not its
    envelope) owns it per `_quad_owns`."""
    text: list[Vector] = []
    drawing: list[Vector] = []
    for ci, cluster in enumerate(clusters):
        quads = [(q, _quad_envelope(q), quad_area(q)) for q in quads_by_cluster.get(ci, [])]
        for v in cluster:
            owned = any(_quad_owns(v, q, env, area) for q, env, area in quads)
            (text if owned else drawing).append(v)
    return text, drawing


def parse(
    vectors_p1: list[Vector], vectors_p2: list[Vector], page: Page,
    *, verbose: bool = False, compute=None,
    debug_out: "dict | None" = None, on_debug_layer: "OnDebugLayer | None" = None,
    step_durations: "dict | None" = None, keep_debug_arrays: bool = True,
) -> tuple[list[Vector], list[Text]]:
    """Phase 1 + Phase 2 vectors -> `(drawing, texts)`; see the module
    docstring for the stages. Render+detect run in chunks of
    `DETECT_RENDER_CHUNK_SIZE` clusters; recognition batches page-wide
    (`OCR_BATCH_SIZE`). `compute` (a Pool-2 proxy) dispatches detect and
    recognition jobs. `debug_out` stashes each stage's data for
    `render_debug`; `on_debug_layer` streams each stage's layers as soon as
    it runs. `keep_debug_arrays=False` keeps image arrays out of
    `debug_out`. `step_durations` receives per-step seconds
    (`classify_separate`/`_collinear`/`_pattern`/`_seqno`/`_spatial`/`_outliers`/
    `_crossings`/`_collect`, `ocr_render`/`ocr_detect`/`ocr_crop`/
    `ocr_recognize`/`ocr_assemble`, `quad_ownership`, `drawing`,
    `debug_render`)."""
    all_vectors = list(vectors_p1) + list(vectors_p2)
    page_meta = page.meta
    clock = StepClock(step_durations)

    def _emit(render: "Callable[[], list[DebugLayer]]") -> None:
        if on_debug_layer is None:
            return
        with clock("debug_render"):
            for layer in render():
                on_debug_layer(*layer)

    # `classify_vectors` times its own steps (`classify_*`) on this clock;
    # no outer `classify` block, or `phase3.other` would subtract it twice.
    cls = classify_vectors(all_vectors, page, verbose=verbose, clock=clock)
    with clock("classify_collect"):
        flat_clusters = [
            [v for group in cluster for v in group] for cluster in cls.text_clusters
        ]
    ocr_clusters = [g for g in flat_clusters if g]
    _emit(lambda: _render_classification_layers(page_meta, cls))
    _emit(lambda: _render_geometry_layers(page_meta, ocr_clusters))

    rec_backend = PaddleRecBackend()
    det_backend = PaddleDetectBackend()

    keep_arrays = debug_out is not None and keep_debug_arrays
    cluster_detections: list[tuple[np.ndarray, list]] = []
    detect_boxes: list[tuple] = []
    detect_quads: list[tuple] = []
    page_quads: list[dict] = []
    for chunk_start in range(0, len(ocr_clusters), DETECT_RENDER_CHUNK_SIZE):
        chunk = ocr_clusters[chunk_start:chunk_start + DETECT_RENDER_CHUNK_SIZE]

        # Stage 1: render each cluster (in-process -- needs fitz).
        clusters_render: list[dict] = []
        with clock("ocr_render"):
            for offset, group_vectors in enumerate(chunk):
                padding = _cluster_render_padding(group_vectors)
                try:
                    image, dpi_used = ocr_prep.render_cluster_with_dynamic_dpi(
                        group_vectors, OCR_DPI, MIN_RENDER_SIDE_PX, MAX_RENDER_DPI, padding,
                    )
                except ValueError:
                    continue
                clusters_render.append({
                    "cluster_idx": chunk_start + offset, "group_vectors": group_vectors,
                    "padding": padding, "dpi_used": dpi_used,
                    "bgr": _normalize_bgr(np.asarray(image)),
                })

        # Stage 2: PaddleOCR detect on the unrotated renders.
        with clock("ocr_detect"):
            if compute is not None and clusters_render:
                quads_per_cluster = compute.starmap(
                    _detect_job, [(c["bgr"], OCR_VERSION, OCR_LANG) for c in clusters_render],
                )
            else:
                quads_per_cluster = [det_backend.detect(c["bgr"]) for c in clusters_render]

        # Stage 3: per quad -- page-space bbox + quad, long-edge angle
        # (snapped to the nearest global potential angle within
        # QUAD_ANGLE_SNAP_TOL_DEG -- render pixel space is page space scaled,
        # unrotated, so the angles compare directly), and the upright crop
        # (region rotated by that angle, quad cut out).
        with clock("ocr_crop"):
            for c, quads in zip(clusters_render, quads_per_cluster):
                if keep_arrays:
                    cluster_detections.append((c["bgr"], quads))
                for quad in quads:
                    pts = np.asarray(quad, dtype=np.float64).reshape(4, 2).tolist()
                    bbox = pixel_to_page_bbox(c["group_vectors"], c["dpi_used"], pts, c["padding"])
                    qpts = tuple(pixel_to_page_points(
                        c["group_vectors"], c["dpi_used"], pts, c["padding"],
                    ))
                    detect_boxes.append(bbox)
                    detect_quads.append(qpts)
                    raw_angle = quad_long_edge_angle(quad)
                    snapped = snap_angle(raw_angle, cls.global_angles, QUAD_ANGLE_SNAP_TOL_DEG)
                    # Not re-wrapped to [-90, 90): a raw 89.9 snapping to 90
                    # must stay 90, not become -90 (a 180-degree crop flip).
                    angle = raw_angle if snapped is None else snapped
                    crop = upright_crop(c["bgr"], quad, angle)
                    # Crops come out of the BGR render; the recognisers do
                    # their own RGB->BGR flip, so hand them RGB.
                    page_quads.append({
                        "cluster_idx": c["cluster_idx"], "group_vectors": c["group_vectors"],
                        "crop": crop[:, :, ::-1], "bbox": bbox, "quad_pts": qpts,
                        "quad_angle": angle,
                        "quad_angle_raw": raw_angle,
                        "snapped": snapped is not None,
                        "region": quad_region(c["bgr"], quad)[0][:, :, ::-1] if keep_arrays else None,
                    })

    with clock("ocr_recognize"):
        def _recognize_batches(crops: list[np.ndarray], job, local) -> list:
            out: list = []
            for start in range(0, len(crops), OCR_BATCH_SIZE):
                batch = crops[start:start + OCR_BATCH_SIZE]
                if compute is not None:
                    out.extend(compute.apply(job, (batch, OCR_VERSION, OCR_LANG)))
                else:
                    out.extend(local(batch))
            return out

        # Stage 4: pass 1 -- angle classifier (0/180) + recognise.
        crops = [pq["crop"] for pq in page_quads]
        boxes = _recognize_batches(crops, _recognize_crops_job, rec_backend.recognize_crops)
        scores = [score(box) for box in boxes]
        recog_crops = list(crops)
        last_attempt_crop: dict[int, np.ndarray] = dict(enumerate(crops)) if keep_arrays else {}
        winning_k: list[int] = [0] * len(boxes)

        # Stage 5: every crop whose raw score is below
        # RETRY_CONFIDENCE_THRESHOLD is also read at +90, +180, +270 (no
        # classifier, batched page-wide per rotation). Attempts are compared
        # on `selection_score` (raw score x 1.2 per dictionary English word);
        # the highest wins, ties to the earlier attempt -- pass 1 is always a
        # candidate. `scores` keeps the winner's raw score.
        retry_idx = [i for i, s in enumerate(scores) if s < RETRY_CONFIDENCE_THRESHOLD]
        selected = [selection_score(box) for box in boxes]
        for k in (1, 2, 3):
            if not retry_idx:
                break
            retry_crops = [np.ascontiguousarray(np.rot90(crops[i], k)) for i in retry_idx]
            retry_boxes = _recognize_batches(
                retry_crops, _recognize_crops_raw_job, rec_backend.recognize_crops_raw,
            )
            for i, rbox, rcrop in zip(retry_idx, retry_boxes, retry_crops):
                if keep_arrays:
                    last_attempt_crop[i] = rcrop
                rsel = selection_score(rbox)
                if rsel[0] > selected[i][0]:
                    boxes[i], scores[i], recog_crops[i], winning_k[i] = rbox, score(rbox), rcrop, k
                    selected[i] = rsel
        retried = set(retry_idx)

    # Stage 6: assemble output, page-wide, in original cluster/quad order.
    with clock("ocr_assemble"):
        texts: list[Text] = []
        recog_bucket_crops: dict[str, list[tuple[np.ndarray, str]]] = {
            "0": [], "1": [], "2": [], "3": [], "failed": [],
        }
        blank_boxes: list[tuple] = []
        blank_quads: list[tuple] = []
        rotation_entries: list[dict] = []
        owner_quads: dict[int, list[tuple]] = {}
        for idx, (pq, box, recog_crop, k) in enumerate(zip(page_quads, boxes, recog_crops, winning_k)):
            if keep_arrays:
                if box.text:
                    recog_bucket_crops[str(k)].append((recog_crop, box.text))
                else:
                    recog_bucket_crops["failed"].append((last_attempt_crop[idx], box.text))
            flip = box.flip_deg if k == 0 else 0
            final_angle = (pq["quad_angle"] + flip + 90.0 * k) % 360.0 if box.text else None
            rotation_entries.append({
                "bbox": pq["bbox"],
                "quad_pts": pq["quad_pts"],
                "quad_angle_deg": pq["quad_angle"],
                "quad_angle_raw_deg": pq["quad_angle_raw"],
                "snapped": pq["snapped"],
                "cls_flip_deg": boxes[idx].flip_deg if k == 0 else None,
                "retry_count": k if box.text else None,
                "retried": idx in retried,
                "score": selected[idx][0] if idx in retried else scores[idx],
                "raw_score": scores[idx],
                "english_words": selected[idx][1],
                "final_angle_deg": final_angle,
            })
            if not box.text:
                blank_boxes.append(pq["bbox"])
                blank_quads.append(pq["quad_pts"])
                continue
            owner_quads.setdefault(pq["cluster_idx"], []).append(pq["quad_pts"])
            direction = transform_direction((1.0, 0.0), final_angle)
            texts.append(Text(
                text=box.text, bbox=pq["bbox"], direction=direction,
                origin=compute_origin(pq["bbox"], direction),
                font="", font_size=0.0, color=None, flags=0,
                ascender=None, descender=None, wmode=0,
                block_no=0, line_no=0, word_no=0,
                page_index=page_meta.index, seqno=min(v.seqno for v in pq["group_vectors"]),
                confidence=box.confidence, source="ocr", orientation_source="ocr",
                quad_points=reorder_quad_reading(pq["quad_pts"], final_angle),
            ))
        retry_stats: dict = {"0": 0, "1": 0, "2": 0, "3": 0, "failed": 0}
        for entry in rotation_entries:
            key = "failed" if entry["retry_count"] is None else str(entry["retry_count"])
            retry_stats[key] = retry_stats.get(key, 0) + 1
    _emit(lambda: _render_ocr_layers(page_meta, texts, blank_quads, detect_quads))
    _emit(lambda: _render_rotation_layers(page_meta, rotation_entries))
    _emit(lambda: _render_retry_layers(page_meta, rotation_entries))

    with clock("quad_ownership"):
        text_vectors, ocr_drawing = _text_vectors_by_quad(ocr_clusters, owner_quads)
    _emit(lambda: _render_ownership_layers(page_meta, text_vectors, ocr_drawing))

    with clock("drawing"):
        drawing = list(cls.drawing_vectors) + ocr_drawing
    _emit(lambda: _render_drawing_layers(page_meta, drawing))

    if debug_out is not None:
        debug_out["classification"] = cls
        debug_out["global_angles"] = cls.global_angles
        debug_out["ocr_clusters"] = ocr_clusters
        debug_out["texts"] = texts
        debug_out["quad_rotation_regions"] = (
            [pq["region"] for pq in page_quads] if keep_arrays else []
        )
        debug_out["paddle_classifier_crops"] = crops if keep_arrays else []
        debug_out["recog_bucket_crops"] = recog_bucket_crops
        debug_out["cluster_detections"] = cluster_detections
        debug_out["ocr_blank_boxes"] = blank_boxes
        debug_out["ocr_detect_boxes"] = detect_boxes
        debug_out["ocr_blank_quads"] = blank_quads
        debug_out["ocr_detect_quads"] = detect_quads
        debug_out["rotation"] = rotation_entries
        debug_out["retry_stats"] = retry_stats
        debug_out["ownership"] = {"text": text_vectors, "drawing": ocr_drawing}
        debug_out["drawing"] = drawing

    return drawing, texts


# ---------------------------------------------------------------------------
# Debug rendering -- one small `_render_<stage>_layers` helper per pipeline
# stage, built from `commons/renderer`'s debug primitives. Each is called two
# ways: inline from `parse()` (streaming) and from `render_debug` below
# (batch, reading the same data back out of `debug_out`). Nothing here is
# shared with a sibling backend.
# ---------------------------------------------------------------------------
_C_KEPT = "#059669"
_C_OCR = "#16a34a"
_C_OCR_BLANK = "#9333ea"
_C_OCR_DETECT = "#2563eb"
_C_DRAWING = "#111827"
_C_DROPPED = "#dc2626"
_C_FLAGGED = "#f59e0b"
_C_ANGLE_QUAD = "#0891b2"
_C_ANGLE_FINAL = "#65a30d"
_C_ANGLE_SNAPPED = "#7c3aed"
_C_CLS_FLIP = "#be185d"
_C_RETRY_1 = "#f59e0b"
_C_RETRY_2 = "#ea580c"
_C_RETRY_3 = "#b91c1c"
_C_RETRY_KEPT = "#64748b"
_C_OWN_TEXT = "#16a34a"
_C_OWN_DRAWING = "#dc2626"
_C_GEOM_COLLINEAR = "#0ea5e9"
_C_GEOM_PARALLEL = "#f97316"
_C_GEOM_SINGLETON = "#9ca3af"


def _hex_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


def _slug(label: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in label.lower())
    while "__" in keep:
        keep = keep.replace("__", "_")
    return keep.strip("_") or "step"


def _flatten_entries(entries: list) -> list[Vector]:
    """A step category's `groups` is `list[list[Vector]]` or
    `list[list[list[Vector]]]` (tiered clusters) -- flatten either down to a
    plain `list[Vector]`."""
    out: list[Vector] = []
    for entry in entries:
        if entry and isinstance(entry[0], list):
            for g in entry:
                out.extend(g)
        else:
            out.extend(entry)
    return out


def _entry_bbox(entry):
    vectors = _flatten_entries([entry])
    return union_bbox([v.bbox for v in vectors]) if vectors else None


def _render_classification_layers(page_meta, cls) -> "list[DebugLayer]":
    """Per classification step: a `kept bbox` layer when the kept boxes
    differ from the previous step's (skipped for the collinear and pattern
    steps, whose kept entries are single Vectors the inspector already
    shows), plus one `dropped <category>` vector layer per dropped category
    with any content -- the pattern step's colours each lattice group with
    its own hue -- except the crossings step's, which is its own
    `intersection` stage: `dropped to drawing (N)` (the dominant-grid
    Vectors) and `flagged, kept (off-grid) (N)` (crossed often enough, but
    not on the dominant grid)."""
    from rastervec.commons.renderer import render_boxes_pdf, render_vectors_pdf

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
        stage = f"classify_{i + 1:02d}_{_slug(label)}"
        kept_groups: list = []
        dropped: dict[str, list] = {}
        flagged: list = []
        for steps in steps_per_bucket:
            if i >= len(steps):
                continue
            for name, cat in steps[i].categories.items():
                if cat.role == "kept":
                    kept_groups.extend(cat.groups)
                elif cat.role == "dropped":
                    dropped.setdefault(name, []).extend(cat.groups)
                elif name == FLAGGED_KEPT_CATEGORY:
                    flagged.extend(cat.groups)
        for name, entries in dropped.items():
            vectors = _flatten_entries(entries)
            if name == CROSSED_CATEGORY:
                # Its own stage, always emitted (empty on a page where the
                # crossings step dropped nothing).
                out.append(("intersection", f"dropped to drawing ({len(vectors)})", _C_DROPPED,
                            render_vectors_pdf(page_meta, vectors, color_of=lambda _v: _hex_rgb(_C_DROPPED))))
                flagged_vectors = _flatten_entries(flagged)
                out.append(("intersection", f"flagged, kept (off-grid) ({len(flagged_vectors)})", _C_FLAGGED,
                            render_vectors_pdf(page_meta, flagged_vectors,
                                               color_of=lambda _v: _hex_rgb(_C_FLAGGED))))
                continue
            if name == PATTERN_CATEGORY and vectors:
                colors = _group_colors(entries)
                out.append((stage, f"dropped {name} ({len(entries)} groups)", _C_DROPPED, render_vectors_pdf(
                    page_meta, vectors, color_of=lambda v, c=colors: c[id(v)],
                )))
                continue
            if vectors:
                out.append((stage, f"dropped {name}", _C_DROPPED, render_vectors_pdf(
                    page_meta, vectors, color_of=lambda _v: _hex_rgb(_C_DROPPED),
                )))
        if label in STEP_LABELS[:2]:
            continue
        kept_boxes = sorted(
            tuple(b) for b in (_entry_bbox(g) for g in kept_groups if g) if b is not None
        )
        if kept_boxes == prev_boxes:
            continue
        prev_boxes = kept_boxes
        out.append((stage, "kept bbox", _C_KEPT, render_boxes_pdf(
            page_meta, [(b, _hex_rgb(_C_KEPT)) for b in kept_boxes],
        )))
    return out


def _group_colors(groups: "list[list[Vector]]") -> "dict[int, tuple]":
    """One golden-ratio hue per group: color by `id(v)`."""
    import colorsys

    return {
        id(v): colorsys.hsv_to_rgb(hue, 0.9, 0.85)
        for hue, group in zip(golden_hues(len(groups)), groups) for v in group
    }


def _geometry_colors(clusters: "list[list[Vector]]", grouper) -> "tuple[dict[int, tuple], list[Vector], list[Vector]]":
    """Per cluster, `grouper(straight)` -> groups; every group of at least
    `MIN_PARALLEL_GROUP_SIZE` gets its own golden-ratio hue (restarting per
    cluster), every smaller group is a singleton. Returns `(color by
    id(v), grouped, singletons)`."""
    import colorsys

    colors: dict[int, tuple] = {}
    grouped: list[Vector] = []
    singles: list[Vector] = []
    for cluster in clusters:
        straight, _other = split_straight(cluster, STRAIGHT_TOL_PT)
        groups = grouper(straight)
        big = [g for g in groups if len(g) >= MIN_PARALLEL_GROUP_SIZE]
        for hue, group in zip(golden_hues(len(big)), big):
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.85)
            for v, _fit in group:
                colors[id(v)] = rgb
                grouped.append(v)
        singles.extend(v for g in groups if len(g) < MIN_PARALLEL_GROUP_SIZE for v, _fit in g)
    return colors, grouped, singles


def _render_geometry_layers(page_meta, clusters: "list[list[Vector]] | None") -> "list[DebugLayer]":
    """Collinear and parallel groups of straight vectors within each final
    classification cluster -- the same `line_geometry.group_collinear`/
    `group_parallel` calls (same tolerances) the collinear-drawing and
    length-outlier steps use -- one hue per group; size-1 groups in their
    own gray layer."""
    from rastervec.commons.renderer import render_vectors_pdf

    clusters = clusters or []
    out: "list[DebugLayer]" = []
    for name, hexcolor, grouper in (
        ("collinear", _C_GEOM_COLLINEAR,
         lambda st: group_collinear(st, ANGLE_TOL_DEG, COLLINEAR_OFFSET_TOL_PT)),
        ("parallel", _C_GEOM_PARALLEL, lambda st: group_parallel(st, ANGLE_TOL_DEG)),
    ):
        colors, grouped, singles = _geometry_colors(clusters, grouper)
        out.append(("geometry", f"{name} groups", hexcolor, render_vectors_pdf(
            page_meta, grouped, color_of=lambda v, c=colors: c[id(v)],
        )))
        out.append(("geometry", f"{name} singletons", _C_GEOM_SINGLETON, render_vectors_pdf(
            page_meta, singles, color_of=lambda _v: _hex_rgb(_C_GEOM_SINGLETON),
        )))
    return out


def _render_ocr_layers(page_meta, texts, blank_quads=None, detect_quads=None) -> "list[DebugLayer]":
    """Detect / passed / failed boxes are drawn as the detector's own quads,
    not their axis-aligned envelopes."""
    from rastervec.commons.renderer import render_quads_pdf, render_text_pdf

    texts = texts or []
    passed_quads = [t.quad_points or _bbox_quad(t.bbox) for t in texts]
    return [
        ("ocr", "detect bbox", _C_OCR_DETECT, render_quads_pdf(
            page_meta, [(q, _hex_rgb(_C_OCR_DETECT)) for q in (detect_quads or [])],
        )),
        ("ocr", "passed bbox", _C_OCR, render_quads_pdf(
            page_meta, [(q, _hex_rgb(_C_OCR)) for q in passed_quads],
        )),
        ("ocr", "failed bbox", _C_OCR_BLANK, render_quads_pdf(
            page_meta, [(q, _hex_rgb(_C_OCR_BLANK)) for q in (blank_quads or [])],
        )),
        ("ocr", "passed text", _C_OCR, render_text_pdf(
            page_meta, texts, color_of=lambda _t: _hex_rgb(_C_OCR),
        )),
    ]


def _bbox_quad(bbox) -> tuple:
    x0, y0, x1, y1 = bbox
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _render_angle_arrows_pdf(page_meta, entries: list[dict], angle_key: str, hexcolor: str) -> bytes:
    """One direction arrow (`draw.arrow_spec`) per `entries` item whose
    `entries[i][angle_key]` isn't `None`, centred on that entry's own
    `"bbox"` and pointing along the angle (the same y-down convention
    `Text.direction` uses)."""
    from rastervec.commons.renderer.draw import arrow_spec, render_specs_pdf

    color = _hex_rgb(hexcolor)
    specs: list = []
    for entry in entries:
        angle = entry.get(angle_key)
        if angle is None:
            continue
        x0, y0, x1, y1 = entry["bbox"]
        length = max(10.0, 0.4 * max(x1 - x0, y1 - y0))
        specs += arrow_spec(((x0 + x1) / 2.0, (y0 + y1) / 2.0), angle, length, color)
    return render_specs_pdf(page_meta, specs)


def _render_rotation_layers(page_meta, rotation_entries: "list[dict] | None") -> "list[DebugLayer]":
    """Per detected quad: `raw quad angle` (the detected long-edge angle),
    `snapped angle (N)` (only the N quads whose angle snapped to a global
    potential angle -- the angle their crop was rotated by), `final angle` (a
    non-blank result's text direction: (snapped) quad angle + classifier flip
    + retry rotation), and `cls flipped` (quads the classifier turned 180 on
    the winning pass)."""
    from rastervec.commons.renderer import render_quads_pdf

    entries = rotation_entries or []
    flipped = [e["quad_pts"] for e in entries if e.get("cls_flip_deg") == 180]
    snapped = [e for e in entries if e.get("snapped")]
    return [
        ("rotation", "raw quad angle", _C_ANGLE_QUAD,
         _render_angle_arrows_pdf(page_meta, entries, "quad_angle_raw_deg", _C_ANGLE_QUAD)),
        ("rotation", f"snapped angle ({len(snapped)})", _C_ANGLE_SNAPPED,
         _render_angle_arrows_pdf(page_meta, snapped, "quad_angle_deg", _C_ANGLE_SNAPPED)),
        ("rotation", "final angle", _C_ANGLE_FINAL,
         _render_angle_arrows_pdf(page_meta, entries, "final_angle_deg", _C_ANGLE_FINAL)),
        ("rotation", f"cls flipped ({len(flipped)})", _C_CLS_FLIP, render_quads_pdf(
            page_meta, [(q, _hex_rgb(_C_CLS_FLIP)) for q in flipped],
        )),
    ]


def _render_retry_layers(page_meta, rotation_entries: "list[dict] | None") -> "list[DebugLayer]":
    """Bbox layers for the low-confidence retry sweep: `N retry` -- the
    detections whose winning read came from the +N*90-degree rotation --
    and `retried, kept pass 1` -- retried, but pass 1 still scored best."""
    from rastervec.commons.renderer import render_boxes_pdf

    entries = rotation_entries or []
    layers: "list[DebugLayer]" = []
    for n, color in ((1, _C_RETRY_1), (2, _C_RETRY_2), (3, _C_RETRY_3)):
        boxes = [e["bbox"] for e in entries if e.get("retry_count") == n]
        layers.append((
            "retry", f"{n} retry", color,
            render_boxes_pdf(page_meta, [(b, _hex_rgb(color)) for b in boxes]),
        ))
    kept = [e["bbox"] for e in entries if e.get("retried") and e.get("retry_count") == 0]
    layers.append((
        "retry", "retried, kept pass 1", _C_RETRY_KEPT,
        render_boxes_pdf(page_meta, [(b, _hex_rgb(_C_RETRY_KEPT)) for b in kept]),
    ))
    return layers


def _render_ownership_layers(page_meta, text_vectors, drawing_vectors) -> "list[DebugLayer]":
    """The OCR'd clusters' vectors split by quad ink ownership: `text
    vectors` (more than `TEXT_INK_INSIDE_FRAC` of their ink inside a
    non-blank quad) and `drawing vectors` (the rest)."""
    from rastervec.commons.renderer import render_vectors_pdf

    return [
        ("ownership", "text vectors", _C_OWN_TEXT, render_vectors_pdf(
            page_meta, text_vectors or [], color_of=lambda _v: _hex_rgb(_C_OWN_TEXT),
        )),
        ("ownership", "drawing vectors", _C_OWN_DRAWING, render_vectors_pdf(
            page_meta, drawing_vectors or [], color_of=lambda _v: _hex_rgb(_C_OWN_DRAWING),
        )),
    ]


def _render_drawing_layers(page_meta, drawing) -> "list[DebugLayer]":
    from rastervec.commons.renderer import render_vectors_pdf

    return [("drawing", "drawing vectors", _C_DRAWING, render_vectors_pdf(
        page_meta, drawing or [], color_of=lambda _v: _hex_rgb(_C_DRAWING),
    ))]


def render_debug(debug_out: "dict | None", page_meta) -> "list[DebugLayer]":
    """Batch/standalone counterpart to the `on_debug_layer` streaming path
    above: one (stage, label, hex, pdf_bytes) tuple per debug layer, built
    from a fully-populated `debug_out`."""
    if not debug_out:
        return []
    ownership = debug_out.get("ownership") or {}
    out: "list[DebugLayer]" = []
    out += _render_classification_layers(page_meta, debug_out.get("classification"))
    out += _render_geometry_layers(page_meta, debug_out.get("ocr_clusters"))
    out += _render_ocr_layers(
        page_meta, debug_out.get("texts"), debug_out.get("ocr_blank_quads"),
        debug_out.get("ocr_detect_quads"),
    )
    out += _render_rotation_layers(page_meta, debug_out.get("rotation"))
    out += _render_retry_layers(page_meta, debug_out.get("rotation"))
    out += _render_ownership_layers(page_meta, ownership.get("text"), ownership.get("drawing"))
    out += _render_drawing_layers(page_meta, debug_out.get("drawing"))
    return out
