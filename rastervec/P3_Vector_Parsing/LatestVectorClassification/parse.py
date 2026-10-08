"""LatestVectorClassification's Phase3Backend entrypoint -- the merged
VectorClassification + CollinearVectorClass backend:

1. `classify_vectors` -- layer/color/width buckets; oversize removal (a
   Vector covering >= `MAX_VECTOR_PAGE_AREA_FRAC` of the page); collinear
   drawing removal (long, regular dashed/repeated lines); pattern-lattice
   removal (similar Vectors repeated on a regular lattice); seqno merge +
   spatial clustering (no cluster reaches `MAX_CLUSTER_PAGE_AREA_FRAC` of
   the page); per cluster, parallel-group length outliers and the
   crossed-grid removal -> drawing. Also the page's global potential
   angles (every >= 2-member collinear group's angle, page-wide, deduped).
2. Per cluster: render (unrotated), PaddleOCR detect.
3. Per detected quad: the quad's **long-edge angle**, snapped to the
   nearest global potential angle when within `QUAD_ANGLE_SNAP_TOL_DEG`
   (`line_geometry.snap_angle`; unchanged otherwise), sets the rotation;
   the image is rotated by it about the quad and the quad cropped out
   upright (`paddle_engine.upright_crop` -- a real rotation, never a
   re-boxed axis-aligned bbox). No Hough, no minAreaRect. A quad longer
   than `MAX_CROP_ASPECT` times its height is first split along its long
   side into pieces, cut in the gaps between the cluster's vectors where
   possible (`_split_positions`/`_sub_quads`); each piece is cropped,
   recognised and output as its own `Text`. The quad itself
   (`Text.quad_points`) stays as detected (or the piece's sub-quad).
4. Page-wide batched recognition: PaddleOCR's 0/180 angle classifier, then
   recognise; every crop whose score (confidence, halved for a single
   character, 0 when blank) is below `RETRY_CONFIDENCE_THRESHOLD` is also
   recognised at +90/180/270 (no classifier) and the best attempt wins --
   compared on score x `ENGLISH_WORD_MULTIPLIER` ** (dictionary English
   words in the read, `english_words.selection_score`). Text angle = (snapped)
   long-edge angle + classifier flip + retry rotation.
5. Text/drawing split (`_text_vectors_by_quad`): a vector of an OCR'd
   cluster is text when some non-blank quad (the rotated quad itself -- for
   a split quad, the whole detected quad, when any piece read non-blank)
   detected in its own cluster owns it -- tiered, cheapest first: bbox
   fully inside the quad -> owned; bbox area larger than the quad -> not;
   fewer than `TEXT_SEGMENT_OVERLAP_FRAC` of its pieces touching the quad
   -> not; else more than `TEXT_INK_INSIDE_FRAC` of its ink inside.
   Drawing = everything classification dropped + every other OCR'd-cluster
   vector. No FAST, nothing transitive.

Debug output: `on_debug_layer` streams each stage's PDF layers as it runs;
`on_debug_image` streams each debug image (`(folder_key, name,
make_image)`, folder keys `detect`/`rotation_quad`/`rotation_classifier`/
`recog_0`..`recog_3`/`recog_failed`) the moment its source array exists, so
no page-wide image arrays are kept. Without `on_debug_image`, a `debug_out`
with `keep_debug_arrays` keeps those arrays instead (batch/notebook path).

Fully self-contained (own paddle_engine.py/config.py/line geometry) --
imports nothing from a sibling P3 backend or P2.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from rastervec.commons.helpers.geometry import (
    bboxes_intersect, compute_origin, transform_direction, union_bbox,
)
from rastervec.commons.models import Page, Vector, Text
from rastervec.commons.renderer import ocr_prep
from rastervec.commons.step_timing import StepClock
from rastervec.P3_Vector_Parsing.LatestVectorClassification.classify_vectors import (
    CROSSED_CATEGORY,
    FLAGGED_KEPT_CATEGORY,
    PATTERN_BUCKET_CATEGORY,
    PATTERN_CATEGORY,
    PATTERN_REJECTED_CATEGORY,
    SINGLE_VECTOR_STEPS,
    classify_vectors,
)
from rastervec.P3_Vector_Parsing.LatestVectorClassification.config import (
    ANGLE_TOL_DEG,
    COLLINEAR_OFFSET_TOL_PT,
    DETECT_RENDER_CHUNK_SIZE,
    INK_CURVE_SAMPLES,
    MAX_CROP_ASPECT,
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
    english_word_count,
    selection_score,
)
from rastervec.P3_Vector_Parsing.LatestVectorClassification.line_geometry import (
    bbox_inside_quad,
    golden_hues,
    group_collinear,
    group_parallel,
    ink_fraction_in_quad,
    ink_segments,
    piece_overlap_fraction,
    piece_rows,
    quad_geom,
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
    quad_size,
    reorder_quad_reading,
    score,
    upright_crop,
)

STEP_NAMES = ["classify", "ocr", "drawing"]

DebugLayer = "tuple[str, str, str, bytes]"
OnDebugLayer = "Callable[[str, str, str, bytes], None]"
OnDebugImage = "Callable[[str, str, Callable[[], object]], None]"

_PDF_POINTS_PER_INCH = 72.0


def _cluster_render_padding(vectors: list[Vector]) -> float:
    """Page-space margin for a cluster's OCR render frame -- half its max
    stroke width plus `RENDER_PADDING_EXTRA_PT`."""
    return max((v.width or 0.0) for v in vectors) / 2.0 + RENDER_PADDING_EXTRA_PT


def _quad_envelope(quad) -> tuple[float, float, float, float]:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return (min(xs), min(ys), max(xs), max(ys))


def _quad_owns(v: Vector, quad, env=None, area: "float | None" = None, *, geom=None, cache=None) -> bool:
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

    `geom` (`line_geometry.quad_geom(quad)`) supplies the quad's precomputed
    edges/normals/envelope/area; `cache` (a dict per vector) keeps `v`'s
    pieces and ink segments across the quads it is tested against."""
    geom = geom if geom is not None else quad_geom(quad)
    env = geom.envelope if env is None else env
    area = geom.area if area is None else area
    if not bboxes_intersect(v.bbox, env):
        return False
    if bbox_inside_quad(v.bbox, quad, geom=geom):
        return True
    x0, y0, x1, y1 = v.bbox
    if (x1 - x0) * (y1 - y0) > area:
        return False
    cache = {} if cache is None else cache
    rows = cache.get("rows")
    if rows is None:
        rows = cache["rows"] = piece_rows(v)
    if piece_overlap_fraction(v, quad, geom=geom, rows=rows) < TEXT_SEGMENT_OVERLAP_FRAC:  # nan -> False
        return False
    ink = cache.get("ink")
    if ink is None:
        ink = cache["ink"] = ink_segments(v, INK_CURVE_SAMPLES)
    return ink_fraction_in_quad(
        v, quad, curve_samples=INK_CURVE_SAMPLES, geom=geom, ink=ink,
    ) > TEXT_INK_INSIDE_FRAC


def _text_vectors_by_quad(
    clusters: list[list[Vector]], quads_by_cluster: dict[int, list[tuple]],
) -> tuple[list[Vector], list[Vector]]:
    """`(text, drawing)` over every vector of `clusters`: a vector is text
    when some quad in `quads_by_cluster[its cluster index]` (only non-blank
    recognitions are passed in -- the rotated quad polygon, not its
    envelope) owns it per `_quad_owns`. Each quad's geometry is built once
    per cluster; a vector is only tested against the quads whose envelope
    its bbox meets (one numpy mask), in the original quad order -- the same
    first-owner-wins result as testing every quad."""
    text: list[Vector] = []
    drawing: list[Vector] = []
    for ci, cluster in enumerate(clusters):
        quads = quads_by_cluster.get(ci, [])
        if not quads:
            drawing.extend(cluster)
            continue
        geoms = [quad_geom(q) for q in quads]
        envs = np.asarray([g.envelope for g in geoms], dtype=float)
        for v in cluster:
            x0, y0, x1, y1 = v.bbox
            cand = np.flatnonzero(
                (envs[:, 0] <= x1) & (x0 <= envs[:, 2]) & (envs[:, 1] <= y1) & (y0 <= envs[:, 3])
            )
            cache: dict = {}
            owned = any(_quad_owns(v, quads[k], geom=geoms[k], cache=cache) for k in cand)
            (text if owned else drawing).append(v)
    return text, drawing


def _merge_intervals(intervals: "list[tuple[float, float]]") -> "list[tuple[float, float]]":
    out: list[list[float]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _split_positions(intervals: "list[tuple[float, float]]", length: float, n: int) -> list[float]:
    """Cut positions (`0 < t < length`, sorted) splitting a quad of `length`
    along its reading axis into `n` pieces. `intervals` are the cluster's
    vectors' extents along that axis; each target `k * length / n` takes the
    nearest unused **gap** midpoint between the merged intervals within half
    a piece length of it, else is cut there as-is (a hard cut)."""
    if n <= 1 or length <= 0.0:
        return []
    clipped = [(max(0.0, a), min(length, b)) for a, b in intervals if b > 0.0 and a < length]
    merged = _merge_intervals(clipped)
    gaps = [(merged[k][1] + merged[k + 1][0]) / 2.0 for k in range(len(merged) - 1)]
    step = length / n
    cuts: list[float] = []
    used: set[int] = set()
    for k in range(1, n):
        target = k * step
        best = None
        for gi, g in enumerate(gaps):
            if gi in used or abs(g - target) > 0.5 * step:
                continue
            if best is None or abs(g - target) < abs(gaps[best] - target):
                best = gi
        if best is None:
            cuts.append(target)
        else:
            used.add(best)
            cuts.append(gaps[best])
    return [c for c in sorted(set(cuts)) if 0.0 < c < length]


def _sub_quads(quad_reading, cuts: list[float], length: float) -> list[tuple]:
    """Pieces of a reading-ordered quad (`p0 -> p1` along the reading axis,
    `p3 -> p2` the opposite edge) cut at `cuts` (distances along `p0 -> p1`
    of total `length`), each in the same corner order."""
    p0, p1, p2, p3 = (np.asarray(p, dtype=float) for p in quad_reading)
    fs = [0.0] + [c / length for c in cuts] + [1.0]
    out = []
    for a, b in zip(fs, fs[1:]):
        corners = (p0 + a * (p1 - p0), p0 + b * (p1 - p0), p3 + b * (p2 - p3), p3 + a * (p2 - p3))
        out.append(tuple((float(c[0]), float(c[1])) for c in corners))
    return out


def _split_quad(qpts, angle: float, vectors: list[Vector]) -> "list[tuple]":
    """`[qpts]` when the page-space quad is at most `MAX_CROP_ASPECT` times
    as long as it is tall, else its pieces (`_split_positions`/`_sub_quads`)
    -- the reading axis is `angle`, and each vector of `vectors` touching
    the quad's envelope contributes its bbox's extent along it."""
    long_side, short_side = quad_size(qpts)
    if short_side <= 0.0 or long_side / short_side <= MAX_CROP_ASPECT:
        return [qpts]
    reading = reorder_quad_reading(qpts, angle)
    p0, p1 = np.asarray(reading[0]), np.asarray(reading[1])
    length = float(np.hypot(*(p1 - p0)))
    if length <= 0.0:
        return [qpts]
    u = (p1 - p0) / length
    env = _quad_envelope(qpts)
    intervals = []
    for v in vectors:
        if not bboxes_intersect(v.bbox, env):
            continue
        x0, y0, x1, y1 = v.bbox
        ts = [float((np.array(c) - p0) @ u) for c in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        intervals.append((min(ts), max(ts)))
    n = math.ceil(long_side / short_side / MAX_CROP_ASPECT)
    cuts = _split_positions(intervals, length, n)
    return _sub_quads(reading, cuts, length) if cuts else [qpts]


def _file_slug(text: str, limit: int = 40) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in (text or "")).strip("_")
    return keep[:limit] or "blank"


def _rgb_image(bgr: np.ndarray):
    from PIL import Image

    return Image.fromarray(np.ascontiguousarray(np.asarray(bgr)[..., ::-1]))


def parse(
    vectors_p1: list[Vector], vectors_p2: list[Vector], page: Page,
    *, verbose: bool = False, compute=None,
    debug_out: "dict | None" = None, on_debug_layer: "OnDebugLayer | None" = None,
    on_debug_image: "OnDebugImage | None" = None,
    step_durations: "dict | None" = None, keep_debug_arrays: bool = True,
) -> tuple[list[Vector], list[Text]]:
    """Phase 1 + Phase 2 vectors -> `(drawing, texts)`; see the module
    docstring for the stages. Render+detect run in chunks of
    `DETECT_RENDER_CHUNK_SIZE` clusters; recognition batches page-wide
    (`OCR_BATCH_SIZE`). `compute` (a Pool-2 proxy) dispatches detect and
    recognition jobs. `debug_out` stashes each stage's data for
    `render_debug`; `on_debug_layer` streams each stage's layers as soon as
    it runs; `on_debug_image` streams each debug image (see the module
    docstring) and keeps image arrays out of `debug_out`, as does
    `keep_debug_arrays=False`. `step_durations` receives per-step seconds
    (`classify_separate`/`_oversize`/`_collinear`/`_pattern`/`_seqno`/`_spatial`/
    `_outliers`/`_crossings`/`_collect`, `ocr_render`/`ocr_detect`/`ocr_crop`/
    `ocr_recognize`/`ocr_assemble`, `quad_ownership`, `drawing`,
    `debug_render`)."""
    all_vectors = list(vectors_p1) + list(vectors_p2)
    page_meta = page.meta
    clock = StepClock(step_durations)
    debugging = debug_out is not None or on_debug_layer is not None

    def _emit(render: "Callable[[], list[DebugLayer]]") -> None:
        if on_debug_layer is None:
            return
        with clock("debug_render"):
            for layer in render():
                on_debug_layer(*layer)

    # Debug images found inside a timed sub-step are queued and handed
    # over right after it (`_flush_images`): `StepClock` doesn't nest, so a
    # `debug_render` block inside `ocr_crop` would be counted twice. A
    # queue holds at most one chunk's crops/renders, which are alive then
    # anyway.
    pending_images: list = []

    def _image(folder: str, name: str, make) -> None:
        if on_debug_image is not None:
            pending_images.append((folder, name, make))

    def _flush_images() -> None:
        if not pending_images:
            return
        with clock("debug_render"):
            for folder, name, make in pending_images:
                on_debug_image(folder, name, make)
        pending_images.clear()

    # `classify_vectors` times its own steps (`classify_*`) on this clock;
    # no outer `classify` block, or `phase3.other` would subtract it twice.
    cls = classify_vectors(all_vectors, page, verbose=verbose, clock=clock, keep_steps=debugging)
    with clock("classify_collect"):
        flat_clusters = [
            [v for group in cluster for v in group] for cluster in cls.text_clusters
        ]
    ocr_clusters = [g for g in flat_clusters if g]
    _emit(lambda: _render_classification_layers(page_meta, cls))
    _emit(lambda: _render_geometry_layers(page_meta, ocr_clusters))

    rec_backend = PaddleRecBackend()
    det_backend = PaddleDetectBackend()

    keep_arrays = debug_out is not None and keep_debug_arrays and on_debug_image is None
    cluster_detections: list[tuple[np.ndarray, list]] = []
    detect_boxes: list[tuple] = []
    detect_quads: list[tuple] = []
    page_quads: list[dict] = []
    # The whole detected quads (in page space), one per detection; a split
    # quad's pieces point back at theirs via `parent` (ownership uses it).
    parent_quads: list[tuple] = []
    for chunk_start in range(0, len(ocr_clusters), DETECT_RENDER_CHUNK_SIZE):
        chunk = ocr_clusters[chunk_start:chunk_start + DETECT_RENDER_CHUNK_SIZE]

        # Stage 1: render each cluster (in-process -- needs fitz). The
        # page-space origin of the render, its zoom and the cluster's min
        # seqno are computed once here, not per quad.
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
                x0, y0, _x1, _y1 = union_bbox([v.bbox for v in group_vectors])
                clusters_render.append({
                    "cluster_idx": chunk_start + offset, "group_vectors": group_vectors,
                    "origin": (x0 - padding, y0 - padding),
                    "zoom": dpi_used / _PDF_POINTS_PER_INCH,
                    "min_seqno": min(v.seqno for v in group_vectors),
                    "bgr": _normalize_bgr(np.asarray(image)),
                })
                del image
        for c in clusters_render:
            _image("detect", f"cluster_{c['cluster_idx']:03d}.png", lambda b=c["bgr"]: _rgb_image(b))
        _flush_images()

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
        # unrotated, so the angles compare directly), the long-quad split,
        # and each piece's upright crop (image rotated by that angle about
        # the piece, piece cut out). Crops stay BGR, as the recognisers take.
        with clock("ocr_crop"):
            for c, quads in zip(clusters_render, quads_per_cluster):
                if keep_arrays:
                    cluster_detections.append((c["bgr"], quads))
                ox, oy = c["origin"]
                zoom = c["zoom"]
                for quad in quads:
                    pts = np.asarray(quad, dtype=np.float64).reshape(4, 2).tolist()
                    qpts = tuple((px / zoom + ox, py / zoom + oy) for px, py in pts)
                    detect_boxes.append(_quad_envelope(qpts))
                    detect_quads.append(qpts)
                    raw_angle = quad_long_edge_angle(quad)
                    snapped = snap_angle(raw_angle, cls.global_angles, QUAD_ANGLE_SNAP_TOL_DEG)
                    # Not re-wrapped to [-90, 90): a raw 89.9 snapping to 90
                    # must stay 90, not become -90 (a 180-degree crop flip).
                    angle = raw_angle if snapped is None else snapped
                    parent = len(parent_quads)
                    parent_quads.append(qpts)
                    pieces = _split_quad(qpts, angle, c["group_vectors"])
                    for piece in pieces:
                        if len(pieces) == 1:
                            pix_quad, size = quad, None
                        else:
                            pix_quad = np.asarray(
                                [((x - ox) * zoom, (y - oy) * zoom) for x, y in piece], dtype=np.float64,
                            )
                            # A piece can be shorter than it is tall: crop
                            # it along the parent's reading axis explicitly.
                            size = (
                                float(np.hypot(*(pix_quad[1] - pix_quad[0]))),
                                float(np.hypot(*(pix_quad[3] - pix_quad[0]))),
                            )
                        crop = upright_crop(c["bgr"], pix_quad, angle, size=size)
                        qi = len(page_quads)
                        _image(
                            "rotation_quad", f"det_{qi:03d}.png",
                            lambda b=c["bgr"], q=pix_quad: _rgb_image(quad_region(b, q)[0]),
                        )
                        _image("rotation_classifier", f"det_{qi:03d}.png", lambda cr=crop: _rgb_image(cr))
                        page_quads.append({
                            "cluster_idx": c["cluster_idx"], "min_seqno": c["min_seqno"],
                            "crop": crop, "bbox": _quad_envelope(piece), "quad_pts": tuple(piece),
                            "parent": parent,
                            "quad_angle": angle,
                            "quad_angle_raw": raw_angle,
                            "snapped": snapped is not None,
                            "region": (
                                quad_region(c["bgr"], pix_quad)[0][:, :, ::-1] if keep_arrays else None
                            ),
                        })
        _flush_images()
        del clusters_render, quads_per_cluster

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
        winning_k: list[int] = [0] * len(boxes)

        # Stage 5: every crop whose raw score is below
        # RETRY_CONFIDENCE_THRESHOLD is also read at +90, +180, +270 (no
        # classifier, batched page-wide per rotation). Attempts are compared
        # on `selection_score` (raw score x 1.2 per dictionary English word);
        # the highest wins, ties to the earlier attempt -- pass 1 is always a
        # candidate. `scores` keeps the winner's raw score. Only retried
        # crops are dictionary-scored.
        retry_idx = [i for i, s in enumerate(scores) if s < RETRY_CONFIDENCE_THRESHOLD]
        selected = {i: selection_score(boxes[i]) for i in retry_idx}
        for k in (1, 2, 3):
            if not retry_idx:
                break
            retry_crops = [np.ascontiguousarray(np.rot90(crops[i], k)) for i in retry_idx]
            retry_boxes = _recognize_batches(
                retry_crops, _recognize_crops_raw_job, rec_backend.recognize_crops_raw,
            )
            del retry_crops
            for i, rbox in zip(retry_idx, retry_boxes):
                rsel = selection_score(rbox)
                if rsel[0] > selected[i][0]:
                    boxes[i], scores[i], winning_k[i] = rbox, score(rbox), k
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
        owned_parents: set[int] = set()
        for idx, (pq, box, k) in enumerate(zip(page_quads, boxes, winning_k)):
            if keep_arrays or on_debug_image is not None:
                # The crop that decided this detection, exactly as the
                # recogniser read it: the winning rotation (a pass-1 win
                # includes the classifier's 180 flip), or (blank) the last
                # one tried -- +270 when retried, since every retried crop
                # runs k = 1, 2, 3.
                if box.text:
                    bucket = str(k)
                    kk = k if k else (2 if box.flip_deg == 180 else 0)
                else:
                    bucket, kk = "failed", (3 if idx in retried else 0)
                if keep_arrays:
                    crop = crops[idx] if kk == 0 else np.rot90(crops[idx], kk)
                    recog_bucket_crops[bucket].append((crop[:, :, ::-1], box.text))
                else:
                    _image(
                        f"recog_{bucket}", f"det_{idx:03d}__{_file_slug(box.text)}.png",
                        lambda cr=crops[idx], kk=kk: _rgb_image(cr if kk == 0 else np.rot90(cr, kk)),
                    )
            flip = box.flip_deg if k == 0 else 0
            final_angle = (pq["quad_angle"] + flip + 90.0 * k) % 360.0 if box.text else None
            if idx in retried:
                entry_score, english = selected[idx]
            else:
                entry_score = scores[idx]
                english = english_word_count(box.text) if debugging else None
            rotation_entries.append({
                "bbox": pq["bbox"],
                "quad_pts": pq["quad_pts"],
                "quad_angle_deg": pq["quad_angle"],
                "quad_angle_raw_deg": pq["quad_angle_raw"],
                "snapped": pq["snapped"],
                "cls_flip_deg": boxes[idx].flip_deg if k == 0 else None,
                "retry_count": k if box.text else None,
                "retried": idx in retried,
                "score": entry_score,
                "raw_score": scores[idx],
                "english_words": english,
                "final_angle_deg": final_angle,
            })
            if not box.text:
                blank_boxes.append(pq["bbox"])
                blank_quads.append(pq["quad_pts"])
                continue
            if pq["parent"] not in owned_parents:
                owned_parents.add(pq["parent"])
                owner_quads.setdefault(pq["cluster_idx"], []).append(parent_quads[pq["parent"]])
            direction = transform_direction((1.0, 0.0), final_angle)
            texts.append(Text(
                text=box.text, bbox=pq["bbox"], direction=direction,
                origin=compute_origin(pq["bbox"], direction),
                font="", font_size=0.0, color=None, flags=0,
                ascender=None, descender=None, wmode=0,
                block_no=0, line_no=0, word_no=0,
                page_index=page_meta.index, seqno=pq["min_seqno"],
                confidence=box.confidence, source="ocr", orientation_source="ocr",
                quad_points=reorder_quad_reading(pq["quad_pts"], final_angle),
            ))
        retry_stats: dict = {"0": 0, "1": 0, "2": 0, "3": 0, "failed": 0}
        for entry in rotation_entries:
            key = "failed" if entry["retry_count"] is None else str(entry["retry_count"])
            retry_stats[key] = retry_stats.get(key, 0) + 1
    _flush_images()
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
        debug_out["paddle_classifier_crops"] = (
            [c[:, :, ::-1] for c in crops] if keep_arrays else []
        )
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
_C_PATTERN_REJECTED = "#2563eb"
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
    differ from the previous step's (skipped for the oversize, collinear
    and pattern steps, whose kept entries are single Vectors the inspector
    already shows), plus one `dropped <category>` vector layer per dropped category
    with any content -- the pattern step's (`pattern` and `pattern_bucket`)
    colour each lattice group with its own hue, and it also emits `kept
    pattern, too much between (N groups)` for the groups that passed the
    size check but failed the in-between test -- except the crossings step's, which is its own
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
        pattern_rejected: list = []
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
                elif name == PATTERN_REJECTED_CATEGORY:
                    pattern_rejected.extend(cat.groups)
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
            if name in (PATTERN_CATEGORY, PATTERN_BUCKET_CATEGORY) and vectors:
                colors = _group_colors(entries)
                out.append((stage, f"dropped {name} ({len(entries)} groups)", _C_DROPPED, render_vectors_pdf(
                    page_meta, vectors, color_of=lambda v, c=colors: c[id(v)],
                )))
                continue
            if vectors:
                out.append((stage, f"dropped {name}", _C_DROPPED, render_vectors_pdf(
                    page_meta, vectors, color_of=lambda _v: _hex_rgb(_C_DROPPED),
                )))
        if pattern_rejected:
            colors = _group_colors(pattern_rejected)
            out.append((stage, f"kept pattern, too much between ({len(pattern_rejected)} groups)",
                        _C_PATTERN_REJECTED, render_vectors_pdf(
                            page_meta, _flatten_entries(pattern_rejected), color_of=lambda v, c=colors: c[id(v)],
                        )))
        if label in SINGLE_VECTOR_STEPS:
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
    return render_specs_pdf(page_meta, specs, deflate=True)


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
