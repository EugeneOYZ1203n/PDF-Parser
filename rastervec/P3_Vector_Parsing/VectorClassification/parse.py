"""VectorClassification's Phase3Backend entrypoint -- the reduced 2-step
Vector_Classification chain (seqno-overlap merge + spatial clustering),
then full PaddleOCR detect/recognize directly over every classification
cluster (no FAST filtering stage, no re-clustering step in between -- a
cluster already *is* a `list[Vector]`, matching
archive/raster_parser/scripts/type2_dump_extraction_pipeline.py::
run_ocr_extraction's `paddle_engine.py::PaddleDetectBackend`/
`PaddleRecBackend` detect-then-recognize pair, the same pair
`P3_Vector_Parsing/LegacyRecreation/parse.py` already ports independently).
Detect still runs once per cluster (each cluster's render is independent),
but recognize (and the blank-retry sweep) batches across every cluster's
quads at once -- see `parse()`'s own docstring for the exact staging and
Pool-2 dispatch. After recognition, an optional FAST pass
(`config.FAST_FILTER_ENABLED`, `fast_filter.py`) decides which vectors under
a recognized detect box are really text -- the rest go to `drawing`. Fully
self-contained (own paddle_engine.py/fast_detect.py/line_geometry.py/
layer_color_separation.py/config.py) -- imports nothing from
P3_Vector_Parsing/LegacyRecreation, CollinearVectorClass, P2_Raster_To_Vec
or the deprecated top-level OCR/.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from rastervec.commons.helpers.clustering import group_by_overlap
from rastervec.commons.helpers.geometry import (
    bboxes_intersect, compute_origin, transform_direction,
)
from rastervec.commons.models import Page, Vector, Text
from rastervec.commons.renderer import ocr_prep, pixel_to_page_bbox, pixel_to_page_points
from rastervec.commons.step_timing import StepClock
from rastervec.P3_Vector_Parsing.VectorClassification.classify_vectors import classify_vectors
from rastervec.P3_Vector_Parsing.VectorClassification import fast_filter
from rastervec.P3_Vector_Parsing.VectorClassification.config import (
    ANGLE_TOL_DEG,
    COLLINEAR_OFFSET_TOL_PT,
    DETECT_RENDER_CHUNK_SIZE,
    FAST_CROP_MAX_ASPECT,
    FAST_CROP_PADDING_PX,
    FAST_DEBUG_HEATMAP_MAX_SIDE,
    FAST_FILTER_ENABLED,
    FAST_GROUP_CENTER_DIST_PT,
    FAST_HEAT_THRESHOLD,
    FAST_INK_FRACTION,
    FAST_INK_GRAY_THRESHOLD,
    MAX_RENDER_DPI,
    MIN_RENDER_SIDE_PX,
    OCR_BATCH_SIZE,
    OCR_DPI,
    OCR_LANG,
    OCR_VERSION,
    MIN_GROUP_SIZE,
    RENDER_PADDING_EXTRA_PT,
    STRAIGHT_TOL_PT,
)
from rastervec.P3_Vector_Parsing.VectorClassification.fast_detect import (
    FastDetector,
    _fast_job,
    default_weights_path,
)
from rastervec.P3_Vector_Parsing.VectorClassification.paddle_engine import (
    PaddleDetectBackend,
    PaddleRecBackend,
    _detect_job,
    _normalize_bgr,
    _normalize_rotation,
    _recognize_crops_job,
    _recognize_crops_raw_job,
    hough_deskew,
)

STEP_NAMES = ["classify", "ocr", "fast_filter", "drawing"]

DebugLayer = "tuple[str, str, str, bytes]"
OnDebugLayer = "Callable[[str, str, str, bytes], None]"


def _cluster_render_padding(vectors: list[Vector]) -> float:
    """Page-space PDF-point margin for a seqno-cluster's own OCR render
    frame -- half its own max stroke width (so a stroke at the bbox edge
    isn't clipped) plus `RENDER_PADDING_EXTRA_PT`, matching
    `LegacyRecreation/parse.py`'s own identical helper."""
    return max((v.width or 0.0) for v in vectors) / 2.0 + RENDER_PADDING_EXTRA_PT


def _quad_allows_rotation(
    bbox: tuple[float, float, float, float], cluster_vectors: list[Vector],
) -> bool:
    """Whether `hough_deskew` should attempt a rotation correction for the
    detected quad at page-space `bbox`: only when the vectors underneath it
    (`cluster_vectors` filtered to this quad's own bbox via
    `bboxes_intersect`) form more than one connected component by bbox
    overlap (`group_by_overlap` -- a pure overlap/touch union-find, no
    distance slack). A single connected component (0/1 overlapping
    vectors, or several that all touch/overlap into one blob) is one
    glyph/shape with no baseline to measure."""
    vectors_in_quad = [v for v in cluster_vectors if bboxes_intersect(v.bbox, bbox)]
    if len(vectors_in_quad) <= 1:
        return False
    components = group_by_overlap(vectors_in_quad, get_bbox=lambda v: v.bbox)
    return len(components) > 1


def _drawing_extra_vectors(
    ocr_vectors: list[Vector], accepted_boxes: list[tuple[float, float, float, float]],
) -> list[Vector]:
    """Every vector from `ocr_vectors` that fails to connect, by bbox
    overlap (`group_by_overlap` -- the same pure overlap/touch union-find
    `_quad_allows_rotation` uses), to any of `accepted_boxes` (a detected
    quad whose recognition actually returned non-blank text). Covers both
    "rejected from OCR" (a vector under a detected-but-blank quad) and
    "never detected at all" (no quad over it whatsoever) in one pass --
    either way, that vector's own connected component contains no
    accepted-box marker. A vector transitively touching an accepted box
    through other overlapping vectors is NOT drawing, even if its own bbox
    doesn't directly touch the box itself -- this replaces FAST's former
    role of deciding what reaches `drawing`."""
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


def _dedupe_by_id(vectors) -> list[Vector]:
    seen: set[int] = set()
    out: list[Vector] = []
    for v in vectors:
        if id(v) not in seen:
            seen.add(id(v))
            out.append(v)
    return out


def _run_fast(groups: "list[fast_filter.FastGroup]", compute, *, keep_inputs: bool) -> None:
    """Sets each group's `heat` (FAST's score map, cut back to its crop),
    then drops its `fast_input` unless `keep_inputs` (debug images)."""
    if not groups:
        return
    if compute is not None:
        weights = default_weights_path()
        masks = compute.starmap(_fast_job, [(weights, g.fast_input) for g in groups])
    else:
        from PIL import Image

        detector = FastDetector()
        masks = [detector.detect(Image.fromarray(g.fast_input)) for g in groups]
    for g, mask in zip(groups, masks):
        g.heat = fast_filter.crop_heat(g, mask)
        if not keep_inputs:
            g.fast_input = None


def _fast_debug_record(
    groups: "list[fast_filter.FastGroup]", page_quads: list[dict], text_ids: "set[int]",
) -> dict:
    """The cheap, render-ready part of the FAST stage for its debug layers
    (`_render_fast_layers`): per group its page bbox, member quads,
    acceptance and a downscaled uint8 heatmap; plus the vectors kept as
    text and the scored-but-dropped ones (scored under an accepted group,
    failed the ink-fraction test everywhere)."""
    kept: list[Vector] = []
    dropped: list[Vector] = []
    seen: set[int] = set()
    for g in groups:
        for v, _frac in g.scores:
            if id(v) in seen:
                continue
            seen.add(id(v))
            (kept if id(v) in text_ids else dropped).append(v)
    return {
        "groups": [{
            "page_bbox": g.page_bbox,
            "crop_page_bbox": g.crop_page_bbox,
            "quads": [page_quads[i]["quad_pts"] for i in g.quad_idxs],
            "accepted": g.accepted,
            "heat": (
                fast_filter.downscale_heat(g.heat, FAST_DEBUG_HEATMAP_MAX_SIDE)
                if g.heat is not None else None
            ),
            "scores": [frac for _v, frac in g.scores],
        } for g in groups],
        "kept": kept,
        "dropped": dropped,
    }


def parse(
    vectors_p1: list[Vector], vectors_p2: list[Vector], page: Page,
    *, verbose: bool = False, compute=None,
    debug_out: "dict | None" = None, on_debug_layer: "OnDebugLayer | None" = None,
    step_durations: "dict | None" = None, keep_debug_arrays: bool = True,
) -> tuple[list[Vector], list[Text]]:
    """Combines Phase 1's raw native vectors and Phase 2's raster-derived
    vectors into one flat pool, then: classify (reduced 2-step chain, per
    `(layer,color)` bucket) -> per classification cluster (a plain
    `list[Vector]`, no FAST filter and no re-clustering step in between --
    every cluster classification produces goes straight to OCR), render +
    PaddleOCR's full detect+recognize pass -> merge into `drawing` every
    vector that classification itself dropped (always empty now that
    neither remaining classification step drops anything) plus every
    vector that OCR rejected or never detected at all (see
    `_drawing_extra_vectors` -- this replaces FAST's former role of
    deciding what reaches `drawing`, now decided downstream of OCR's own
    detect+recognize results instead of upstream of them).

    The OCR pass is staged page-wide rather than looped one cluster at a
    time: every cluster is rendered first (stage 1, in-process --
    rendering needs this process's own shared fitz document, which a Pool-2
    worker never has), then every cluster's PaddleOCR *detect* call runs
    (stage 2, one independent call per cluster -- dispatched across Pool-2
    workers via `paddle_engine._detect_job` when `compute` is given, else
    called in-process), then every detected quad across the *whole page* is
    deskewed/cropped into one flat pool (stage 3, gated per-quad by
    `_quad_allows_rotation` -- see that function) and *recognized* in
    `config.OCR_BATCH_SIZE`-sized batches (stage 4, each batch dispatched via
    `paddle_engine._recognize_crops_job` when `compute` is given), with the
    blank-recognition retry sweep (stage 5) batched the same way across the
    whole page's still-blank crops at each pass, not just one cluster's. A
    page with many small text clusters therefore makes a handful of batched
    PaddleOCR calls instead of one call per cluster.

    Two independent, optional debug outlets (see `LegacyRecreation/parse.py`
    for the shared convention): `debug_out` stashes every stage's own
    intermediate object verbatim for `render_debug` to render as a
    post-hoc batch later; `on_debug_layer` renders and emits each stage's
    layers immediately, right after that stage runs. The classification
    chain itself (`classify_vectors`) is one atomic call either way -- its
    own per-step kept/dropped breakdown is rendered as soon as it returns,
    still well before the later (heavier) OCR stages run.

    `step_durations`, when given, receives wall-clock seconds per step
    (`commons.step_timing.StepClock`; debug rendering excluded) --
    `classify`, then the page-wide OCR stages split into
    `ocr_render`/`ocr_detect`/`ocr_recognize` (summed across every cluster's/
    batch's own share of that stage), and `drawing`.

    `keep_debug_arrays=False` keeps the full-size image arrays out of
    `debug_out` (`cluster_detections`/`paddle_classifier_crops` stay empty,
    `recog_bucket_crops` stays all-empty-buckets, `rotation` entries carry no
    crops/masks) -- see `core.pipeline.run_pipeline`."""
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
        flat_clusters = [
            [v for group in cluster for v in group] for cluster in cls.text_clusters
        ]
    _emit(lambda: _render_classification_layers(page_meta, cls))

    ocr_clusters = [g for g in flat_clusters if g]
    _emit(lambda: _render_geometry_layers(page_meta, ocr_clusters))

    rec_backend = PaddleRecBackend()
    det_backend = PaddleDetectBackend()

    # Stages 1-3 (render -> detect -> deskew/crop) run in bounded chunks of
    # `DETECT_RENDER_CHUNK_SIZE` clusters, not the whole page's clusters at
    # once: a cluster's render can be tens of MB (a large title block/border
    # at even the base OCR_DPI), so holding every cluster's render
    # simultaneously on a page with hundreds of clusters risks exhausting
    # memory. Recognition (stage 4/5 below) is unaffected -- crops are far
    # smaller than full cluster renders, so that stage still batches across
    # the WHOLE page at once via `page_quads`, accumulated here chunk by
    # chunk. `cluster_detections`/`classifier_crops`/`ocr_crops` (raw
    # per-cluster/per-quad bgr, debug-only -- read back by
    # `render_debug`/`debug_out`) and `hough_deskew`'s own per-quad debug
    # crop/masks (`keep_debug=`) are all only kept when something actually
    # asked for debug output, for the same reason -- each holds a full-size
    # image array per cluster/quad, page-wide, for the rest of this
    # function's run. `keep_debug_arrays=False` skips them even with a
    # `debug_out` (the cheap boxes/angles/stats are still stashed).
    keep_cluster_detections = debug_out is not None and keep_debug_arrays
    cluster_detections: list[tuple[np.ndarray, list]] = []
    detect_boxes: list[tuple] = []
    # The detector's own (rotated, cv2.minAreaRect) quads in page space --
    # `detect_boxes` is only their axis-aligned envelope, used for logic.
    detect_quads: list[tuple] = []
    page_quads: list[dict] = []
    # FAST input (used in stage 7): one crop per anchored group of a
    # cluster's detect quads, cut out of the cluster render here (inside the
    # chunk loop) so only the small crops -- never the full renders --
    # outlive the chunk.
    fast_groups: list[fast_filter.FastGroup] = []
    for chunk_start in range(0, len(ocr_clusters), DETECT_RENDER_CHUNK_SIZE):
        chunk = ocr_clusters[chunk_start:chunk_start + DETECT_RENDER_CHUNK_SIZE]

        # Stage 1: render this chunk's clusters, in-process --
        # `render_cluster_with_dynamic_dpi` uses this process's own shared
        # fitz document, and Pool-2 workers never import fitz/pymupdf, so
        # rendering can't be dispatched there. Everything downstream
        # (detect/recognize) works off the plain numpy `bgr` array this
        # produces.
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
                clusters_render.append({
                    "group_vectors": group_vectors, "padding": padding,
                    "dpi_used": dpi_used, "bgr": _normalize_bgr(np.asarray(image)),
                })

        # Stage 2: PaddleOCR detect, one call per cluster in this chunk.
        # Each cluster's render is independent of every other, so -- unlike
        # the old per-cluster detect+recognize loop -- these fan out across
        # Pool-2 workers via `_detect_job` when a caller passes `compute`,
        # instead of running one at a time in the calling process.
        with clock("ocr_detect"):
            if compute is not None and clusters_render:
                quads_per_cluster = compute.starmap(
                    _detect_job, [(c["bgr"], OCR_VERSION, OCR_LANG) for c in clusters_render],
                )
            else:
                quads_per_cluster = [det_backend.detect(c["bgr"]) for c in clusters_render]

        # Stage 3: per detected quad, deskew+crop (cheap CPU work, stays
        # in-process) and compute its page-space bbox (once, not twice --
        # see this session's earlier bbox-caching fix). This chunk's quads
        # are accumulated into the page-wide `page_quads` list so
        # recognition (stage 4) and the blank-retry sweep (stage 5) batch
        # across the WHOLE PAGE, not one chunk's handful of quads at a
        # time -- only `clusters_render`'s own bgr arrays are chunk-scoped
        # (freed once this chunk's iteration ends), the much smaller crops
        # they produce are not.
        fast_pending: list[tuple] = []
        with clock("ocr_recognize"):
            for c, quads in zip(clusters_render, quads_per_cluster):
                if keep_cluster_detections:
                    cluster_detections.append((c["bgr"], quads))
                if not quads:
                    continue
                quad_bboxes = [
                    pixel_to_page_bbox(c["group_vectors"], c["dpi_used"], quad.tolist(), c["padding"])
                    for quad in quads
                ]
                quad_pts = [
                    tuple(pixel_to_page_points(
                        c["group_vectors"], c["dpi_used"], quad.tolist(), c["padding"],
                    ))
                    for quad in quads
                ]
                detect_boxes.extend(quad_bboxes)
                detect_quads.extend(quad_pts)
                if FAST_FILTER_ENABLED:
                    fast_pending.append((c, quads, quad_bboxes, len(page_quads)))
                for quad, bbox, qpts in zip(quads, quad_bboxes, quad_pts):
                    # hough_deskew's crop is cropped straight out of `bgr`,
                    # so it's already BGR -- reverse channels back before
                    # recognize_crops, which does its own RGB->BGR flip
                    # internally (same gotcha LegacyRecreation's own
                    # identical loop works around).
                    allow_rotation = _quad_allows_rotation(bbox, c["group_vectors"])
                    crop, rd = hough_deskew(
                        c["bgr"], quad, keep_debug=keep_cluster_detections,
                        allow_rotation=allow_rotation,
                    )
                    page_quads.append({
                        "group_vectors": c["group_vectors"], "crop": crop[:, :, ::-1],
                        "rd": rd, "bbox": bbox, "quad_pts": qpts,
                    })

        with clock("fast_filter"):
            for c, quads, quad_bboxes, first_idx in fast_pending:
                fast_groups.extend(fast_filter.build_groups(
                    c["group_vectors"], c["bgr"], c["dpi_used"], c["padding"],
                    quads, quad_bboxes, first_idx,
                    center_dist_pt=FAST_GROUP_CENTER_DIST_PT,
                    pad_px=FAST_CROP_PADDING_PX, max_aspect=FAST_CROP_MAX_ASPECT,
                ))

    with clock("ocr_recognize"):
        # Stage 4: recognize, chunked by OCR_BATCH_SIZE across the whole
        # page's quads at once -- a page with hundreds of small clusters
        # now makes a handful of batched engine calls instead of one call
        # per cluster. Dispatched to Pool 2 per batch when `compute` is
        # given (mirrors `FastIntoPaddle/paddle_engine.py::
        # recognize_segments`'s `recognize_fn` hook, before that module was
        # removed), else called in-process.
        def _recognize_batches(crops: list[np.ndarray], job, local_fn) -> list:
            out: list = []
            for start in range(0, len(crops), OCR_BATCH_SIZE):
                batch = crops[start:start + OCR_BATCH_SIZE]
                if compute is not None:
                    out.extend(compute.apply(job, (batch, OCR_VERSION, OCR_LANG)))
                else:
                    out.extend(local_fn(batch))
            return out

        crops_all = [pq["crop"] for pq in page_quads]
        boxes = _recognize_batches(crops_all, _recognize_crops_job, rec_backend.recognize_crops)

        # `crops_all` is also exactly what recognize_crops's own internal
        # text_classifier call saw (pass 1, pre-flip, every detection) --
        # debug-only, read back via debug_out["paddle_classifier_crops"], so
        # only kept when a caller actually asked for debug output.
        classifier_inputs: list[np.ndarray] = list(crops_all) if keep_cluster_detections else []

        # box.flip_deg is pass 1's own classifier 0/180 decision -- captured
        # now, before a later retry pass can overwrite `boxes`, since it's
        # always worth recording even if a retry ends up winning.
        # `recog_crops`/`retry_counts`/`retry_extra_degs` track what
        # actually got used per detection as the sweep below runs.
        flip_angle_degs = [float(box.flip_deg) for box in boxes]
        recog_crops = [
            np.rot90(pq["crop"], 2) if box.flip_deg else pq["crop"]
            for pq, box in zip(page_quads, boxes)
        ]
        # last_attempt_crop tracks, per detection index, the most recent crop
        # handed to recognize_crops/recognize_crops_raw regardless of
        # outcome -- a permanently-blank detection's own `recog_crops[i]`
        # never advances past its pass-1 value (only overwritten on
        # success, below), so this is the only way to recover "the last
        # rotation variant actually tried" for the eventual "failed" debug
        # bucket. Debug-only, same gating as classifier_inputs.
        last_attempt_crop: dict[int, np.ndarray] = (
            {i: c for i, c in enumerate(recog_crops)} if keep_cluster_detections else {}
        )

        retry_counts: list = [0 if box.text else None for box in boxes]
        retry_extra_degs: list = [None] * len(boxes)

        # Stage 5: blank-recognition retry sweep, +90 -> 180 -> 270 relative
        # to each crop's own pass-1 base (hough_deskew's output, not the
        # classifier-flipped variant), now batched across every still-blank
        # crop on the WHOLE PAGE at each pass (not just one cluster's),
        # same OCR_BATCH_SIZE-chunked dispatch as stage 4. No classifier
        # call in a retry pass -- sweeping all 4 quarter-turns already
        # covers whatever the classifier's own 0/180 choice would have
        # picked. `last_attempt_crop` is updated for every attempted retry
        # crop (whether or not it recovers text), so a permanently-blank
        # detection's final debug image is its own last-tried rotation.
        for k, extra_deg in ((1, 90.0), (2, 180.0), (3, 270.0)):
            blank_idx = [i for i, box in enumerate(boxes) if not box.text]
            if not blank_idx:
                break
            retry_crops = [np.rot90(page_quads[i]["crop"], k) for i in blank_idx]
            retry_boxes = _recognize_batches(
                retry_crops, _recognize_crops_raw_job, rec_backend.recognize_crops_raw,
            )
            for i, rbox, rcrop in zip(blank_idx, retry_boxes, retry_crops):
                if keep_cluster_detections:
                    last_attempt_crop[i] = rcrop
                if rbox.text:
                    boxes[i] = rbox
                    recog_crops[i] = rcrop
                    retry_counts[i] = k
                    retry_extra_degs[i] = extra_deg

    # Stage 6: assemble output, page-wide, in original cluster/quad order.
    texts: list[Text] = []
    recog_bucket_crops: dict[str, list[tuple[np.ndarray, str]]] = {
        "0": [], "1": [], "2": [], "3": [], "failed": [],
    }
    blank_boxes: list[tuple] = []
    blank_quads: list[tuple] = []
    rotation_entries: list[dict] = []
    for idx, (pq, box, recog_crop, flip_a, retry_n, retry_extra) in enumerate(zip(
        page_quads, boxes, recog_crops, flip_angle_degs, retry_counts, retry_extra_degs,
    )):
        if keep_cluster_detections:
            if box.text:
                recog_bucket_crops[str(retry_n)].append((recog_crop, box.text))
            else:
                recog_bucket_crops["failed"].append((last_attempt_crop[idx], box.text))
        rd, bbox = pq["rd"], pq["bbox"]
        best_angle = None
        if box.text:
            effective_extra = retry_extra if retry_extra is not None else flip_a
            best_angle = _normalize_rotation(rd.combined_angle_deg + effective_extra)
        rotation_entries.append({
            "bbox": bbox,
            "hough_angle_deg": rd.hough_angle_deg,
            "minarea_angle_deg": rd.minarea_angle_deg,
            "combined_angle_deg": rd.combined_angle_deg,
            "flip_angle_deg": flip_a,
            "retry_count": retry_n,
            "best_angle_deg": best_angle,
            "base_crop": rd.base_crop,
            "dilated_ink_mask": rd.dilated_ink_mask,
            "minarea_mask": rd.minarea_mask,
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
    _emit(lambda: _render_rotation_layers(page_meta, rotation_entries))
    _emit(lambda: _render_retry_layers(page_meta, rotation_entries))

    # Stage 7: FAST text/drawing split. A group is accepted when any of its
    # quads recognized non-blank text; only accepted groups go through FAST
    # (one call per group, page-wide; dispatched to Pool 2 via `_fast_job`
    # when `compute` is given) -- a blank-only group's vectors are drawing
    # whatever FAST says, so it's never run for them. Then every vector
    # under an accepted group is scored against that group's heatmap (see
    # `fast_filter.split_text_vectors`). Recognition above is unaffected.
    fast_debug: "dict | None" = None
    text_ids: "set[int] | None" = None
    if FAST_FILTER_ENABLED:
        with clock("fast_filter"):
            for g in fast_groups:
                g.accepted = any(boxes[i].text for i in g.quad_idxs)
                if not g.accepted and not keep_cluster_detections:
                    g.fast_input = None
            _run_fast(
                [g for g in fast_groups if g.accepted], compute,
                keep_inputs=keep_cluster_detections,
            )
            text_ids = fast_filter.split_text_vectors(
                fast_groups, heat_threshold=FAST_HEAT_THRESHOLD,
                ink_fraction=FAST_INK_FRACTION, gray_threshold=FAST_INK_GRAY_THRESHOLD,
            )
        if on_debug_layer is not None or debug_out is not None:
            fast_debug = _fast_debug_record(fast_groups, page_quads, text_ids)
        _emit(lambda: _render_fast_layers(page_meta, fast_debug))

    with clock("drawing"):
        ocr_vectors = [v for cluster in ocr_clusters for v in cluster]
        if text_ids is not None:
            extra = _dedupe_by_id(v for v in ocr_vectors if id(v) not in text_ids)
        else:
            accepted_boxes = [txt.bbox for txt in texts]
            extra = _drawing_extra_vectors(ocr_vectors, accepted_boxes)
        drawing = list(cls.drawing_vectors) + extra
    _emit(lambda: _render_drawing_layers(page_meta, drawing))

    if debug_out is not None:
        debug_out["classification"] = cls
        debug_out["texts"] = texts
        debug_out["paddle_classifier_crops"] = classifier_inputs
        debug_out["recog_bucket_crops"] = recog_bucket_crops
        debug_out["cluster_detections"] = cluster_detections
        debug_out["ocr_blank_boxes"] = blank_boxes
        debug_out["ocr_detect_boxes"] = detect_boxes
        debug_out["ocr_blank_quads"] = blank_quads
        debug_out["ocr_detect_quads"] = detect_quads
        debug_out["rotation"] = rotation_entries
        debug_out["retry_stats"] = retry_stats
        debug_out["drawing"] = drawing
        debug_out["fast"] = fast_debug
        # Full-size FAST inputs + heatmaps, one (rgb, heat) per group, for
        # the `for_fast/` debug images -- only with keep_debug_arrays.
        debug_out["fast_images"] = [
            (g.fast_input, g.heat) for g in fast_groups
            if keep_cluster_detections and g.fast_input is not None and g.heat is not None
        ]

    return drawing, texts


# ---------------------------------------------------------------------------
# Debug rendering -- one small `_render_<stage>_layers` helper per pipeline
# stage (same convention as `LegacyRecreation/parse.py`), built only from the
# three generic primitives in `commons/renderer`
# (render_boxes_pdf/render_text_pdf/render_vectors_pdf). Each is called two
# ways: inline from `parse()` (streaming) and from `render_debug` below
# (batch, reading the same data back out of `debug_out`). Nothing here is
# shared with LegacyRecreation/Junction.
# ---------------------------------------------------------------------------
_C_KEPT = "#059669"
_C_OCR = "#16a34a"
_C_OCR_BLANK = "#9333ea"
_C_OCR_DETECT = "#2563eb"
_C_DRAWING = "#111827"
_C_ANGLE_HOUGH = "#ea580c"
_C_ANGLE_MINAREA = "#0891b2"
_C_ANGLE_COMBINED = "#65a30d"
_C_ANGLE_FLIP = "#9333ea"
_C_ANGLE_BEST = "#dc2626"
_C_RETRY_1 = "#f59e0b"
_C_RETRY_2 = "#ea580c"
_C_RETRY_3 = "#b91c1c"
_C_FAST_CROP = "#7c3aed"
_C_FAST_CROP_REJECTED = "#9ca3af"
_C_FAST_HEAT = "#dc2626"
_C_FAST_KEPT = "#16a34a"
_C_FAST_DROPPED = "#e11d48"
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


def _render_ocr_layers(page_meta, texts, blank_quads=None, detect_quads=None) -> "list[DebugLayer]":
    """Detect / passed / failed boxes are drawn as the detector's own quads
    (PaddleOCR's DB detector returns rotated `cv2.minAreaRect` quads), not
    their axis-aligned envelopes."""
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
    `Text.direction` uses). Visualizes the rotation pipeline's five angle
    sources (hough/minarea/combined/flip/best) side by side as toggleable
    layers -- see `_render_rotation_layers`."""
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
    """Five arrow layers, one per angle value `parse.py`'s per-quad loop
    tracks per detection (`debug_out["rotation"]`): Hough's raw line-angle
    reading, `cv2.minAreaRect`'s raw reading, the two combined+10-degree-
    snapped (before classifier/retry), the pass-1 classifier's raw 0/180
    flip decision, and the fully-resolved "best" angle actually used for a
    non-blank result (combined + whichever of flip/retry recovered text)."""
    entries = rotation_entries or []
    return [
        ("rotation", "hough angle", _C_ANGLE_HOUGH,
         _render_angle_arrows_pdf(page_meta, entries, "hough_angle_deg", _C_ANGLE_HOUGH)),
        ("rotation", "minarea angle", _C_ANGLE_MINAREA,
         _render_angle_arrows_pdf(page_meta, entries, "minarea_angle_deg", _C_ANGLE_MINAREA)),
        ("rotation", "combined angle", _C_ANGLE_COMBINED,
         _render_angle_arrows_pdf(page_meta, entries, "combined_angle_deg", _C_ANGLE_COMBINED)),
        ("rotation", "flip angle", _C_ANGLE_FLIP,
         _render_angle_arrows_pdf(page_meta, entries, "flip_angle_deg", _C_ANGLE_FLIP)),
        ("rotation", "best angle", _C_ANGLE_BEST,
         _render_angle_arrows_pdf(page_meta, entries, "best_angle_deg", _C_ANGLE_BEST)),
    ]


def _render_retry_layers(page_meta, rotation_entries: "list[dict] | None") -> "list[DebugLayer]":
    """Three bbox-highlight layers (not arrows) -- one per blank-recognition
    retry count (1, 2, 3), each showing the bboxes of detections that only
    recovered non-blank text after that many extra +90-degree passes (see
    `parse.py`'s per-quad loop's retry sweep)."""
    from rastervec.commons.renderer import render_boxes_pdf

    entries = rotation_entries or []
    layers: "list[DebugLayer]" = []
    for n, color in ((1, _C_RETRY_1), (2, _C_RETRY_2), (3, _C_RETRY_3)):
        boxes = [e["bbox"] for e in entries if e.get("retry_count") == n]
        layers.append((
            "retry", f"{n} retry", color,
            render_boxes_pdf(page_meta, [(b, _hex_rgb(color)) for b in boxes]),
        ))
    return layers


def _render_fast_layers(page_meta, fast_debug: "dict | None") -> "list[DebugLayer]":
    """FAST text/drawing split (`parse()` stage 7): the crops FAST saw
    (accepted groups purple, groups whose quads all recognized blank gray --
    those are never scored), the detect quads grouped into them, each
    group's heatmap as a red overlay (alpha = heat) over its crop, and the
    vectors FAST kept as text vs. the scored ones it sent to drawing."""
    from rastervec.commons.renderer import render_boxes_pdf, render_quads_pdf, render_vectors_pdf
    from rastervec.commons.renderer.draw import image_spec, render_specs_pdf

    fast_debug = fast_debug or {"groups": [], "kept": [], "dropped": []}
    groups = fast_debug["groups"]
    crop_boxes = [
        (g["crop_page_bbox"], _hex_rgb(_C_FAST_CROP if g["accepted"] else _C_FAST_CROP_REJECTED))
        for g in groups
    ]
    member_quads = [
        (q, _hex_rgb(_C_FAST_CROP if g["accepted"] else _C_FAST_CROP_REJECTED), "[2 2] 0")
        for g in groups for q in g["quads"]
    ]
    r, gr, b = (int(c * 255) for c in _hex_rgb(_C_FAST_HEAT))
    heat_specs = []
    for g in groups:
        heat = g["heat"]
        if heat is None or heat.size == 0:
            continue
        rgba = np.empty(heat.shape + (4,), dtype=np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = r, gr, b
        rgba[..., 3] = heat
        heat_specs.append(image_spec(rgba, g["crop_page_bbox"]))
    return [
        ("fast", "group crop bbox", _C_FAST_CROP, render_boxes_pdf(page_meta, crop_boxes)),
        ("fast", "group member quads", _C_FAST_CROP, render_quads_pdf(page_meta, member_quads, width=0.75)),
        ("fast", "heatmap", _C_FAST_HEAT, render_specs_pdf(page_meta, heat_specs)),
        ("fast", "kept as text", _C_FAST_KEPT, render_vectors_pdf(
            page_meta, fast_debug["kept"], color_of=lambda _v: _hex_rgb(_C_FAST_KEPT),
        )),
        ("fast", "dropped to drawing", _C_FAST_DROPPED, render_vectors_pdf(
            page_meta, fast_debug["dropped"], color_of=lambda _v: _hex_rgb(_C_FAST_DROPPED),
        )),
    ]


def _geometry_colors(clusters: "list[list[Vector]]", grouper) -> "tuple[dict[int, tuple], list[Vector], list[Vector]]":
    """Per cluster, `grouper(straight)` -> groups; every group of at least
    `MIN_GROUP_SIZE` gets its own golden-ratio hue (restarting per cluster,
    so neighbouring groups within a cluster always differ), every smaller
    group is a singleton. Returns `(color by id(v), grouped, singletons)`."""
    import colorsys

    from rastervec.P3_Vector_Parsing.VectorClassification.line_geometry import (
        golden_hues, split_straight,
    )

    colors: dict[int, tuple] = {}
    grouped: list[Vector] = []
    singles: list[Vector] = []
    for cluster in clusters:
        straight, _other = split_straight(cluster, STRAIGHT_TOL_PT)
        groups = grouper(straight)
        big = [g for g in groups if len(g) >= MIN_GROUP_SIZE]
        for hue, group in zip(golden_hues(len(big)), big):
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.85)
            for v, _fit in group:
                colors[id(v)] = rgb
                grouped.append(v)
        singles.extend(v for g in groups if len(g) < MIN_GROUP_SIZE for v, _fit in g)
    return colors, grouped, singles


def _render_geometry_layers(page_meta, clusters: "list[list[Vector]] | None") -> "list[DebugLayer]":
    """Collinear and parallel groups of straight vectors within each
    classification cluster (`line_geometry.group_collinear`/
    `group_parallel`), one hue per group; size-1 groups in their own gray
    layer. Debug-only -- these groups don't affect output."""
    from rastervec.commons.renderer import render_vectors_pdf
    from rastervec.P3_Vector_Parsing.VectorClassification.line_geometry import (
        group_collinear, group_parallel,
    )

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
    cls = debug_out.get("classification")
    out += _render_classification_layers(page_meta, cls)
    out += _render_geometry_layers(page_meta, [
        flat for flat in (
            [v for group in cluster for v in group] for cluster in (cls.text_clusters if cls else [])
        ) if flat
    ])
    out += _render_ocr_layers(
        page_meta, debug_out.get("texts"), debug_out.get("ocr_blank_quads"),
        debug_out.get("ocr_detect_quads"),
    )
    out += _render_rotation_layers(page_meta, debug_out.get("rotation"))
    out += _render_retry_layers(page_meta, debug_out.get("rotation"))
    if debug_out.get("fast") is not None:
        out += _render_fast_layers(page_meta, debug_out["fast"])
    out += _render_drawing_layers(page_meta, debug_out.get("drawing"))
    return out
