"""CollinearVectorClass's Phase3Backend entrypoint -- VectorClassification's
classify -> PaddleOCR detect/recognize pipeline, with vector-geometry
signals from the three probe notebooks (collinear, parallel, intersection):

1. `classify_vectors` -- layer/color/width buckets; collinear groups (long,
   regular dashed/repeated lines -> drawing) and the page's global potential
   angles; seqno merge + spatial clustering; per cluster, parallel-group
   length outliers and heavily crossed Vectors -> drawing.
2. Per cluster: render, then **rotate the render before detect** by the
   cluster's dominant parallel-group direction (`rotation.
   pre_detect_direction`); PaddleOCR detect on the rotated render; quads are
   mapped back through the inverse rotation to page space.
3. Per detected quad: the parallel groups and connected components of the
   cluster vectors under it pick the final direction (`rotation.
   final_direction` -- Hough mod 180, snapped to a parallel-group or global
   angle, only when there are several components and not exactly one
   group); the quad's crop out of the rotated render is rotated by the
   remaining difference.
4. Page-wide batched recognition with **no angle classifier**; every crop
   whose score (confidence, halved for a single character, 0 when blank) is
   below `RETRY_CONFIDENCE_THRESHOLD` is also recognised at +90/180/270 and
   the best-scoring attempt wins.
5. Drawing = everything classification dropped + every OCR'd vector not
   connected to a recognised quad.

Fully self-contained (own paddle_engine.py/config.py/line geometry) --
imports nothing from a sibling P3 backend or P2.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from rastervec.commons.helpers.clustering import group_by_overlap
from rastervec.commons.helpers.geometry import (
    bboxes_intersect, compute_origin, transform_direction, union_bbox,
)
from rastervec.commons.models import Page, Vector, Text
from rastervec.commons.renderer import ocr_prep, pixel_to_page_bbox, pixel_to_page_points
from rastervec.commons.step_timing import StepClock
from rastervec.P3_Vector_Parsing.CollinearVectorClass.classify_vectors import (
    CROSSED_CATEGORY,
    classify_vectors,
)
from rastervec.P3_Vector_Parsing.CollinearVectorClass.config import (
    DETECT_RENDER_CHUNK_SIZE,
    MAX_RENDER_DPI,
    MIN_RENDER_SIDE_PX,
    OCR_BATCH_SIZE,
    OCR_DPI,
    OCR_LANG,
    OCR_VERSION,
    RENDER_PADDING_EXTRA_PT,
    RETRY_CONFIDENCE_THRESHOLD,
)
from rastervec.P3_Vector_Parsing.CollinearVectorClass.paddle_engine import (
    PaddleDetectBackend,
    PaddleRecBackend,
    _detect_job,
    _normalize_bgr,
    _recognize_crops_raw_job,
    axis_aligned_crop,
    hough_angle_deg,
    normalize_rotation,
    rotate_image,
    score,
    unrotate_points,
)
from rastervec.P3_Vector_Parsing.CollinearVectorClass.rotation import (
    final_direction,
    parallel_groups,
    pre_detect_direction,
)

STEP_NAMES = ["classify", "ocr", "drawing"]

DebugLayer = "tuple[str, str, str, bytes]"
OnDebugLayer = "Callable[[str, str, str, bytes], None]"


def _cluster_render_padding(vectors: list[Vector]) -> float:
    """Page-space margin for a cluster's OCR render frame -- half its max
    stroke width plus `RENDER_PADDING_EXTRA_PT`."""
    return max((v.width or 0.0) for v in vectors) / 2.0 + RENDER_PADDING_EXTRA_PT


def _vectors_under(bbox: tuple[float, float, float, float], cluster_vectors: list[Vector]) -> list[Vector]:
    return [v for v in cluster_vectors if bboxes_intersect(v.bbox, bbox)]


def _component_count(vectors: list[Vector]) -> int:
    """Connected components of `vectors` by bbox overlap/touch."""
    if len(vectors) <= 1:
        return len(vectors)
    return len(group_by_overlap(vectors, get_bbox=lambda v: v.bbox))


def _drawing_extra_vectors(
    ocr_vectors: list[Vector], accepted_boxes: list[tuple[float, float, float, float]],
) -> list[Vector]:
    """Every vector from `ocr_vectors` whose bbox-overlap connected
    component contains none of `accepted_boxes` (quads with non-blank
    recognitions) -- rejected by OCR or never detected at all."""
    if not ocr_vectors:
        return []
    items = [("vec", i) for i in range(len(ocr_vectors))] + [
        ("box", j) for j in range(len(accepted_boxes))
    ]
    bbox_of = {
        **{("vec", i): v.bbox for i, v in enumerate(ocr_vectors)},
        **{("box", j): b for j, b in enumerate(accepted_boxes)},
    }
    components = group_by_overlap(items, get_bbox=lambda it: bbox_of[it])
    drawing_vecs: list[Vector] = []
    for component in components:
        if any(kind == "box" for kind, _ in component):
            continue
        drawing_vecs.extend(ocr_vectors[i] for kind, i in component if kind == "vec")
    return drawing_vecs


def parse(
    vectors_p1: list[Vector], vectors_p2: list[Vector], page: Page,
    *, verbose: bool = False, compute=None,
    debug_out: "dict | None" = None, on_debug_layer: "OnDebugLayer | None" = None,
    step_durations: "dict | None" = None, keep_debug_arrays: bool = True,
) -> tuple[list[Vector], list[Text]]:
    """Phase 1 + Phase 2 vectors -> `(drawing, texts)`; see the module
    docstring for the stages. Same staging, Pool-2 dispatch (`compute`),
    debug outlets (`debug_out`, streaming `on_debug_layer`), `step_durations`
    and `keep_debug_arrays` contract as VectorClassification's `parse`;
    `debug_out` additionally carries `global_angles` and `cluster_rotation`
    (per rendered cluster: bbox, pre-detect angle and rule), and each
    `rotation` entry carries the pre-detect/hough/final angles, the
    deciding `rule`, the winning `score` and whether it was `retried`.
    There is no angle classifier: `paddle_classifier_crops` is always empty
    and `retry_count` is the winning rotation (0 = pass 1, `None` = every
    attempt blank)."""
    all_vectors = list(vectors_p1) + list(vectors_p2)
    page_meta = page.meta
    clock = StepClock(step_durations)

    def _emit(render: "Callable[[], list[DebugLayer]]") -> None:
        if on_debug_layer is None:
            return
        with clock("debug_render"):
            for layer in render():
                on_debug_layer(*layer)

    with clock("classify"):
        cls = classify_vectors(all_vectors, page, verbose=verbose)
        flat_clusters = [
            [v for group in cluster for v in group] for cluster in cls.text_clusters
        ]
    _emit(lambda: _render_classification_layers(page_meta, cls))

    ocr_clusters = [g for g in flat_clusters if g]
    global_angles = cls.global_angles

    rec_backend = PaddleRecBackend()
    det_backend = PaddleDetectBackend()

    # Stages 1-3 run in chunks of DETECT_RENDER_CHUNK_SIZE clusters so the
    # page's (possibly huge) cluster renders are never all alive at once;
    # recognition still batches page-wide via `page_quads`.
    keep_cluster_detections = debug_out is not None and keep_debug_arrays
    cluster_detections: list[tuple[np.ndarray, list]] = []
    cluster_rotation: list[dict] = []
    detect_boxes: list[tuple] = []
    # The detector's own quads mapped back through the inverse rotation --
    # genuinely oriented in page space; `detect_boxes` is their envelope.
    detect_quads: list[tuple] = []
    page_quads: list[dict] = []
    for chunk_start in range(0, len(ocr_clusters), DETECT_RENDER_CHUNK_SIZE):
        chunk = ocr_clusters[chunk_start:chunk_start + DETECT_RENDER_CHUNK_SIZE]

        # Stage 1: render each cluster (in-process -- needs fitz), then
        # rotate the render by the cluster's dominant parallel-group
        # direction so detect sees roughly horizontal text.
        clusters_render: list[dict] = []
        with clock("ocr_render"):
            for group_vectors in chunk:
                padding = _cluster_render_padding(group_vectors)
                try:
                    image, dpi_used = ocr_prep.render_cluster_with_dynamic_dpi(
                        group_vectors, OCR_DPI, MIN_RENDER_SIDE_PX, MAX_RENDER_DPI, padding,
                    )
                except ValueError:
                    continue
                decision = pre_detect_direction(parallel_groups(group_vectors))
                pre_angle = normalize_rotation(decision.angle)
                rotated, m = rotate_image(_normalize_bgr(np.asarray(image)), pre_angle)
                clusters_render.append({
                    "group_vectors": group_vectors, "padding": padding,
                    "dpi_used": dpi_used, "bgr": rotated, "m": m, "pre_angle": pre_angle,
                })
                cluster_rotation.append({
                    "bbox": union_bbox([v.bbox for v in group_vectors]),
                    "angle_deg": pre_angle, "rule": decision.rule,
                })

        # Stage 2: PaddleOCR detect on the rotated renders.
        with clock("ocr_detect"):
            if compute is not None and clusters_render:
                quads_per_cluster = compute.starmap(
                    _detect_job, [(c["bgr"], OCR_VERSION, OCR_LANG) for c in clusters_render],
                )
            else:
                quads_per_cluster = [det_backend.detect(c["bgr"]) for c in clusters_render]

        # Stage 3: per quad -- page bbox through the inverse rotation, then
        # the final direction from the vectors under it, then the crop out
        # of the rotated render rotated by what's left.
        with clock("ocr_recognize"):
            for c, quads in zip(clusters_render, quads_per_cluster):
                if keep_cluster_detections:
                    cluster_detections.append((c["bgr"], quads))
                for quad in quads:
                    unrotated = unrotate_points(quad, c["m"]).tolist()
                    bbox = pixel_to_page_bbox(
                        c["group_vectors"], c["dpi_used"], unrotated, c["padding"],
                    )
                    qpts = tuple(pixel_to_page_points(
                        c["group_vectors"], c["dpi_used"], unrotated, c["padding"],
                    ))
                    detect_boxes.append(bbox)
                    detect_quads.append(qpts)
                    under = _vectors_under(bbox, c["group_vectors"])
                    base = axis_aligned_crop(c["bgr"], quad)
                    hough_mask: list = [None]

                    def _hough(base=base, pre_angle=c["pre_angle"], hough_mask=hough_mask):
                        angle, mask = hough_angle_deg(base)
                        hough_mask[0] = mask
                        return None if angle is None else (angle + pre_angle) % 180.0

                    decision = final_direction(
                        _component_count(under), parallel_groups(under), _hough, global_angles,
                    )
                    final_angle = normalize_rotation(decision.angle)
                    crop, _m = rotate_image(base, normalize_rotation(final_angle - c["pre_angle"]))
                    # Crops come out of the BGR render; recognize_crops_raw
                    # does its own RGB->BGR flip, so hand it RGB.
                    page_quads.append({
                        "group_vectors": c["group_vectors"], "crop": crop[:, :, ::-1], "bbox": bbox,
                        "quad_pts": qpts,
                        "pre_angle": c["pre_angle"], "final_angle": final_angle,
                        "hough_angle": decision.hough_angle, "rule": decision.rule,
                        "base_crop": base if keep_cluster_detections else None,
                        "hough_mask": hough_mask[0] if keep_cluster_detections else None,
                    })

    with clock("ocr_recognize"):
        # Stage 4: pass-1 recognition, no angle classifier, OCR_BATCH_SIZE
        # batches across the whole page.
        def _recognize_batches(crops: list[np.ndarray]) -> list:
            out: list = []
            for start in range(0, len(crops), OCR_BATCH_SIZE):
                batch = crops[start:start + OCR_BATCH_SIZE]
                if compute is not None:
                    out.extend(compute.apply(_recognize_crops_raw_job, (batch, OCR_VERSION, OCR_LANG)))
                else:
                    out.extend(rec_backend.recognize_crops_raw(batch))
            return out

        boxes = _recognize_batches([pq["crop"] for pq in page_quads])
        scores = [score(box) for box in boxes]
        recog_crops = [pq["crop"] for pq in page_quads]
        last_attempt_crop: dict[int, np.ndarray] = (
            dict(enumerate(recog_crops)) if keep_cluster_detections else {}
        )
        winning_k = [0] * len(boxes)

        # Stage 5: every crop whose pass-1 score is below
        # RETRY_CONFIDENCE_THRESHOLD is also recognised at +90, +180 and +270
        # (all three, batched page-wide per rotation); the highest score
        # wins, ties to the earlier attempt -- so pass 1 is always a
        # candidate and a retry can never make a result worse.
        retry_idx = [i for i, s in enumerate(scores) if s < RETRY_CONFIDENCE_THRESHOLD]
        for k in (1, 2, 3):
            if not retry_idx:
                break
            retry_crops = [np.rot90(page_quads[i]["crop"], k) for i in retry_idx]
            for i, rbox, rcrop in zip(retry_idx, _recognize_batches(retry_crops), retry_crops):
                if keep_cluster_detections:
                    last_attempt_crop[i] = rcrop
                if score(rbox) > scores[i]:
                    boxes[i], scores[i], recog_crops[i], winning_k[i] = rbox, score(rbox), rcrop, k
        retried = set(retry_idx)

    # Stage 6: assemble output, page-wide, in original cluster/quad order.
    texts: list[Text] = []
    recog_bucket_crops: dict[str, list[tuple[np.ndarray, str]]] = {
        "0": [], "1": [], "2": [], "3": [], "failed": [],
    }
    blank_boxes: list[tuple] = []
    blank_quads: list[tuple] = []
    rotation_entries: list[dict] = []
    for idx, (pq, box, recog_crop, k) in enumerate(zip(page_quads, boxes, recog_crops, winning_k)):
        retry_n = k if box.text else None
        if keep_cluster_detections:
            if box.text:
                recog_bucket_crops[str(k)].append((recog_crop, box.text))
            else:
                recog_bucket_crops["failed"].append((last_attempt_crop[idx], box.text))
        bbox = pq["bbox"]
        best_angle = normalize_rotation(pq["final_angle"] + 90.0 * k) if box.text else None
        rotation_entries.append({
            "bbox": bbox,
            "rule": pq["rule"],
            "pre_detect_angle_deg": pq["pre_angle"],
            "hough_angle_deg": pq["hough_angle"],
            "final_angle_deg": pq["final_angle"],
            "retry_count": retry_n,
            "retried": idx in retried,
            "score": scores[idx],
            "best_angle_deg": best_angle,
            "base_crop": pq["base_crop"],
            "dilated_ink_mask": pq["hough_mask"],
            "minarea_mask": None,
        })
        if not box.text:
            blank_boxes.append(bbox)
            blank_quads.append(pq["quad_pts"])
            continue
        direction = transform_direction((1.0, 0.0), best_angle)
        texts.append(Text(
            text=box.text, bbox=bbox, direction=direction,
            origin=compute_origin(bbox, direction),
            font="", font_size=0.0, color=None, flags=0,
            ascender=None, descender=None, wmode=0,
            block_no=0, line_no=0, word_no=0,
            page_index=page_meta.index, seqno=min(v.seqno for v in pq["group_vectors"]),
            confidence=box.confidence, source="ocr", orientation_source="ocr",
            quad_points=pq["quad_pts"],
        ))
    retry_stats: dict = {"0": 0, "1": 0, "2": 0, "3": 0, "failed": 0}
    for entry in rotation_entries:
        key = "failed" if entry["retry_count"] is None else str(entry["retry_count"])
        retry_stats[key] = retry_stats.get(key, 0) + 1
    _emit(lambda: _render_ocr_layers(page_meta, texts, blank_quads, detect_quads))
    _emit(lambda: _render_rotation_layers(page_meta, rotation_entries, cluster_rotation))
    _emit(lambda: _render_rule_layers(page_meta, rotation_entries, cluster_rotation))
    _emit(lambda: _render_retry_layers(page_meta, rotation_entries))

    with clock("drawing"):
        ocr_vectors = [v for cluster in ocr_clusters for v in cluster]
        accepted_boxes = [txt.bbox for txt in texts]
        drawing = list(cls.drawing_vectors) + _drawing_extra_vectors(ocr_vectors, accepted_boxes)
    _emit(lambda: _render_drawing_layers(page_meta, drawing))

    if debug_out is not None:
        debug_out["classification"] = cls
        debug_out["global_angles"] = global_angles
        debug_out["cluster_rotation"] = cluster_rotation
        debug_out["texts"] = texts
        debug_out["paddle_classifier_crops"] = []  # no angle classifier
        debug_out["recog_bucket_crops"] = recog_bucket_crops
        debug_out["cluster_detections"] = cluster_detections
        debug_out["ocr_blank_boxes"] = blank_boxes
        debug_out["ocr_detect_boxes"] = detect_boxes
        debug_out["ocr_blank_quads"] = blank_quads
        debug_out["ocr_detect_quads"] = detect_quads
        debug_out["rotation"] = rotation_entries
        debug_out["retry_stats"] = retry_stats
        debug_out["drawing"] = drawing

    return drawing, texts


# ---------------------------------------------------------------------------
# Debug rendering -- one small `_render_<stage>_layers` helper per pipeline
# stage (same convention as `LegacyRecreation/parse.py`), built only from the
# three generic primitives in `commons/renderer`
# (render_boxes_pdf/render_text_pdf/render_vectors_pdf). Each is called two
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
_C_ANGLE_PRE = "#0891b2"
_C_ANGLE_HOUGH = "#ea580c"
_C_ANGLE_FINAL = "#65a30d"
_C_ANGLE_BEST = "#dc2626"
_C_RETRY_1 = "#f59e0b"
_C_RETRY_2 = "#ea580c"
_C_RETRY_3 = "#b91c1c"
_C_RETRY_KEPT = "#64748b"
_RULE_COLORS = ("#2563eb", "#16a34a", "#d97706", "#9333ea", "#0891b2", "#be185d", "#4d7c0f", "#7c2d12")


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
    """Per classification step: a `kept bbox` layer when the kept boxes
    differ from the previous step's (skipped for step 1, whose kept entries
    are single Vectors the inspector already shows), plus one `dropped
    <category>` vector layer per dropped category with any content
    (collinear drawing groups, length outliers) -- except the crossings
    step's, which is its own `intersection / dropped to drawing (N)` layer."""
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
        for steps in steps_per_bucket:
            if i >= len(steps):
                continue
            for name, cat in steps[i].categories.items():
                if cat.role == "kept":
                    kept_groups.extend(cat.groups)
                elif cat.role == "dropped":
                    dropped.setdefault(name, []).extend(cat.groups)
        for name, entries in dropped.items():
            vectors = _flatten_entries(entries)
            if name == CROSSED_CATEGORY:
                # Its own stage, always emitted (empty on a page where the
                # crossings step dropped nothing).
                out.append(("intersection", f"dropped to drawing ({len(vectors)})", _C_DROPPED,
                            render_vectors_pdf(page_meta, vectors, color_of=lambda _v: _hex_rgb(_C_DROPPED))))
                continue
            if vectors:
                out.append((stage, f"dropped {name}", _C_DROPPED, render_vectors_pdf(
                    page_meta, vectors, color_of=lambda _v: _hex_rgb(_C_DROPPED),
                )))
        if i == 0:
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


def _render_ocr_layers(page_meta, texts, blank_quads=None, detect_quads=None) -> "list[DebugLayer]":
    """Detect / passed / failed boxes are drawn as the detector's own quads
    (mapped back through the pre-detect rotation), not their axis-aligned
    envelopes."""
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
    `Text.direction` uses). Visualizes the rotation pipeline's angle
    sources side by side as toggleable layers -- see
    `_render_rotation_layers`."""
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


def _render_rotation_layers(
    page_meta, rotation_entries: "list[dict] | None", cluster_rotation: "list[dict] | None" = None,
) -> "list[DebugLayer]":
    """Arrow layers: the per-cluster pre-detect rotation (at each cluster's
    bbox), then per detected quad the page-space Hough reading (only where
    Hough ran), the final pre-recognition direction and the resolved best
    angle of a non-blank result (final + the winning retry rotation)."""
    entries = rotation_entries or []
    return [
        ("rotation", "pre-detect angle", _C_ANGLE_PRE,
         _render_angle_arrows_pdf(page_meta, cluster_rotation or [], "angle_deg", _C_ANGLE_PRE)),
        ("rotation", "hough angle", _C_ANGLE_HOUGH,
         _render_angle_arrows_pdf(page_meta, entries, "hough_angle_deg", _C_ANGLE_HOUGH)),
        ("rotation", "final angle", _C_ANGLE_FINAL,
         _render_angle_arrows_pdf(page_meta, entries, "final_angle_deg", _C_ANGLE_FINAL)),
        ("rotation", "best angle", _C_ANGLE_BEST,
         _render_angle_arrows_pdf(page_meta, entries, "best_angle_deg", _C_ANGLE_BEST)),
    ]


def _render_rule_layers(
    page_meta, rotation_entries: "list[dict] | None", cluster_rotation: "list[dict] | None" = None,
) -> "list[DebugLayer]":
    """One bbox layer per rotation rule that fired: stage `rotation_rule_
    cluster` for the pre-detect rules (cluster bboxes), `rotation_rule_quad`
    for the final per-quad rules (detected-quad bboxes)."""
    from rastervec.commons.renderer import render_boxes_pdf

    out: "list[DebugLayer]" = []
    for stage, entries in (
        ("rotation_rule_cluster", cluster_rotation or []),
        ("rotation_rule_quad", rotation_entries or []),
    ):
        by_rule: dict[str, list] = {}
        for e in entries:
            by_rule.setdefault(e["rule"], []).append(e["bbox"])
        for k, rule in enumerate(sorted(by_rule)):
            color = _RULE_COLORS[k % len(_RULE_COLORS)]
            out.append((stage, f"{rule} ({len(by_rule[rule])})", color, render_boxes_pdf(
                page_meta, [(b, _hex_rgb(color)) for b in by_rule[rule]],
            )))
    return out


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
    out += _render_ocr_layers(
        page_meta, debug_out.get("texts"), debug_out.get("ocr_blank_quads"),
        debug_out.get("ocr_detect_quads"),
    )
    out += _render_rotation_layers(page_meta, debug_out.get("rotation"), debug_out.get("cluster_rotation"))
    out += _render_rule_layers(page_meta, debug_out.get("rotation"), debug_out.get("cluster_rotation"))
    out += _render_retry_layers(page_meta, debug_out.get("rotation"))
    out += _render_drawing_layers(page_meta, debug_out.get("drawing"))
    return out
