"""Steps 2-5 of Junction's pre-tracing flow: find and read text in a raster.

1. `tile_grid` -- cut the image into `OCR_TILE_PX` tiles overlapping by
   `OCR_TILE_OVERLAP_FRAC` (960 px = PaddleOCR's own det_limit_side_len, so
   tiles are never resized by the detector).
2. `detect_tiles` -- PaddleOCR detect per tile; each quad becomes an
   axis-aligned box in image pixels, tagged with its tile.
3. `merge_cross_tile_boxes` -- boxes from *different* tiles that intersect
   (within `TILE_MERGE_GAP_PX`) are unioned transitively (union-find) into
   one box spanning all their extreme x/y. This re-joins a line split by a
   tile border, and dedupes a word seen whole in two overlapping tiles.
   Boxes from the same tile are never merged with each other -- PaddleOCR
   kept them apart deliberately.
4. `refine_and_recognize` -- per merged box: pad 1 (`RENDER_PADDING_EXTRA_PT`
   in page points, VectorClassification's render padding), upscale so the
   short side is at least `MIN_RENDER_SIDE_PX` (capped at `MAX_UPSCALE`),
   detect again on that small crop; each refined quad is then deskewed and
   cropped with pad 2 (`paddle_engine.hough_deskew`) and recognized in
   page-wide `OCR_BATCH_SIZE` batches, with a +90/180/270 blank-retry sweep
   -- the same staging as VectorClassification's `parse.py` stages 4-5.

Every detect/recognize call goes through Pool 2 when a `compute` proxy is
given (`paddle_engine._detect_job` / `_recognize_crops_job` /
`_recognize_crops_raw_job`), else runs in-process."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import cv2
import numpy as np
from tqdm import tqdm

from rastervec.commons.logging_setup import get_logger

from rastervec.P2_Raster_To_Vec.Junction.config import (
    MAX_UPSCALE,
    MIN_RENDER_SIDE_PX,
    OCR_BATCH_SIZE,
    OCR_LANG,
    OCR_TILE_OVERLAP_FRAC,
    OCR_TILE_PX,
    OCR_VERSION,
    RENDER_PADDING_EXTRA_PT,
    TILE_MERGE_GAP_PX,
)
from rastervec.P2_Raster_To_Vec.Junction.paddle_engine import (
    PaddleDetectBackend,
    PaddleRecBackend,
    _detect_job,
    _recognize_crops_job,
    _recognize_crops_raw_job,
    hough_deskew,
    normalize_rotation,
)

_LOG = get_logger("P2.Junction.ocr")

BBox = tuple[float, float, float, float]
# Images per detect dispatch -- bounds how many tile/crop copies are pickled
# to Pool 2 (or held) at once.
_DETECT_CHUNK = 16
# A second-pass quad must have at least this fraction of its own bbox area
# inside its merged (unpadded) box to belong to it -- pad 1 deliberately
# shows the detector neighbouring text, which its own merged box owns.
_REFINE_OWNERSHIP_FRAC = 0.5
# Refined quads (across all merged boxes) overlapping more than this IoU are
# duplicates; the first one wins.
_DEDUPE_IOU = 0.5


@dataclass
class TileBox:
    bbox: BBox
    tile: int
    # The detector's own quad (4x2, whole-image px) -- `bbox` is its
    # axis-aligned envelope; PaddleOCR's DB detector returns rotated quads.
    quad: "np.ndarray | None" = None


@dataclass
class OcrHit:
    """One refined detection: `quad` (4x2, image px), `padded_box` (the
    region text removal may erase in), recognized `text` ("" if blank after
    every retry), `confidence`, `angle_deg` (the reading-direction rotation,
    `None` if blank), and `retry_count` (0 = first pass, 1-3 = retry, `None`
    if blank)."""

    quad: np.ndarray
    padded_box: BBox
    text: str = ""
    confidence: float = 0.0
    angle_deg: "float | None" = None
    retry_count: "int | None" = None


@dataclass
class OcrResult:
    tiles: list[BBox] = field(default_factory=list)
    tile_boxes: list[TileBox] = field(default_factory=list)
    merged: list[BBox] = field(default_factory=list)
    hits: list[OcrHit] = field(default_factory=list)


# --------------------------------------------------------------------------
# Tiling + cross-tile merge (pure, unit-tested)
# --------------------------------------------------------------------------
def _starts(length: int, tile: int, step: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)  # snap the last tile to the edge
    return starts


def tile_grid(
    h: int, w: int, tile: int = OCR_TILE_PX, overlap_frac: float = OCR_TILE_OVERLAP_FRAC,
) -> list[tuple[int, int, int, int]]:
    """`(x0, y0, x1, y1)` tiles covering an `h`x`w` image, row-major."""
    step = max(1, tile - int(round(tile * overlap_frac)))
    return [
        (x0, y0, min(w, x0 + tile), min(h, y0 + tile))
        for y0 in _starts(h, tile, step) for x0 in _starts(w, tile, step)
    ]


def _quad_bbox(quad: np.ndarray) -> BBox:
    return (float(quad[:, 0].min()), float(quad[:, 1].min()),
            float(quad[:, 0].max()), float(quad[:, 1].max()))


def merge_cross_tile_boxes(boxes: list[TileBox], gap: float = TILE_MERGE_GAP_PX) -> list[BBox]:
    """Union-find over pairs from different tiles whose boxes intersect
    within `gap` px; each group -> the bbox of its extreme x/y. Sweep over
    boxes sorted by x0, so only x-overlapping candidates are compared."""
    n = len(boxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = sorted(range(n), key=lambda i: boxes[i].bbox[0])
    for a_pos, i in enumerate(order):
        ax0, ay0, ax1, ay1 = boxes[i].bbox
        for j in order[a_pos + 1:]:
            bx0, by0, bx1, by1 = boxes[j].bbox
            if bx0 > ax1 + gap:
                break
            if boxes[i].tile == boxes[j].tile:
                continue
            if by0 <= ay1 + gap and ay0 <= by1 + gap:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    merged: list[BBox] = []
    for members in groups.values():
        merged.append((
            min(boxes[i].bbox[0] for i in members), min(boxes[i].bbox[1] for i in members),
            max(boxes[i].bbox[2] for i in members), max(boxes[i].bbox[3] for i in members),
        ))
    merged.sort(key=lambda b: (b[1], b[0]))
    return merged


def _iou(a: BBox, b: BBox) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _inside_frac(inner: BBox, outer: BBox) -> float:
    area = max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return ix * iy / area


# --------------------------------------------------------------------------
# Paddle dispatch
# --------------------------------------------------------------------------
def _chunks(seq: list, n: int) -> Iterable[list]:
    for start in range(0, len(seq), n):
        yield seq[start:start + n]


def _make_detect_many(compute) -> Callable[[list[np.ndarray]], list[list[np.ndarray]]]:
    det = PaddleDetectBackend()

    def detect_many(images: list[np.ndarray]) -> list[list[np.ndarray]]:
        out: list[list[np.ndarray]] = []
        for chunk in _chunks(images, _DETECT_CHUNK):
            if compute is not None:
                out.extend(compute.starmap(_detect_job, [(im, OCR_VERSION, OCR_LANG) for im in chunk]))
            else:
                out.extend(det.detect(im) for im in chunk)
        return out

    return detect_many


def _make_recognize(compute):
    rec = PaddleRecBackend()

    def run(crops: list[np.ndarray], job, local_fn, desc: str) -> list:
        out: list = []
        batches = tqdm(
            _chunks(crops, OCR_BATCH_SIZE), total=math.ceil(len(crops) / OCR_BATCH_SIZE),
            desc=desc, unit="batch", leave=False,
        )
        for batch in batches:
            if compute is not None:
                out.extend(compute.apply(job, (batch, OCR_VERSION, OCR_LANG)))
            else:
                out.extend(local_fn(batch))
        return out

    def recognize(crops):
        return run(crops, _recognize_crops_job, rec.recognize_crops, "Junction OCR recognize")

    def recognize_raw(crops):
        return run(crops, _recognize_crops_raw_job, rec.recognize_crops_raw, "Junction OCR retry")

    return recognize, recognize_raw


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
def detect_tiles(bgr: np.ndarray, tiles: list[tuple[int, int, int, int]], detect_many) -> list[TileBox]:
    """Detect per tile (tiles are views into `bgr`, never copied locally)."""
    out: list[TileBox] = []
    bar = tqdm(total=len(tiles), desc="Junction OCR tile detect", unit="tile", leave=False)
    for chunk_start in range(0, len(tiles), _DETECT_CHUNK):
        chunk = tiles[chunk_start:chunk_start + _DETECT_CHUNK]
        quads_per_tile = detect_many([bgr[y0:y1, x0:x1] for x0, y0, x1, y1 in chunk])
        bar.update(len(chunk))
        for offset, (tile, quads) in enumerate(zip(chunk, quads_per_tile)):
            x0, y0 = tile[0], tile[1]
            for quad in quads:
                b = _quad_bbox(np.asarray(quad, dtype=np.float64))
                out.append(TileBox(
                    (b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0), chunk_start + offset,
                    np.asarray(quad, dtype=np.float64) + np.array([x0, y0], dtype=np.float64),
                ))
    bar.close()
    return out


def _padded_crop(bgr: np.ndarray, box: BBox, pad_px: float):
    h, w = bgr.shape[:2]
    x0 = max(0, int(np.floor(box[0] - pad_px)))
    y0 = max(0, int(np.floor(box[1] - pad_px)))
    x1 = min(w, int(np.ceil(box[2] + pad_px)))
    y1 = min(h, int(np.ceil(box[3] + pad_px)))
    if x1 <= x0 or y1 <= y0:
        return None
    crop = bgr[y0:y1, x0:x1]
    short = min(crop.shape[:2])
    scale = float(np.clip(MIN_RENDER_SIDE_PX / max(short, 1), 1.0, MAX_UPSCALE))
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop, (float(x0), float(y0), float(x1), float(y1)), scale


def refine_and_recognize(
    bgr: np.ndarray, merged: list[BBox], px_per_pt: float, bg_bgr: tuple[int, int, int],
    detect_many, recognize, recognize_raw,
) -> list[OcrHit]:
    pad_px = RENDER_PADDING_EXTRA_PT * px_per_pt
    hits: list[OcrHit] = []
    crops_for_rec: list[np.ndarray] = []
    angles: list[float] = []
    kept_bboxes: list[BBox] = []

    bar = tqdm(total=len(merged), desc="Junction OCR refine detect", unit="box", leave=False)
    for chunk in _chunks(merged, _DETECT_CHUNK):
        prepared = [(_padded_crop(bgr, box, pad_px), box) for box in chunk]
        prepared = [(p, box) for p, box in prepared if p is not None]
        quads_per_crop = detect_many([p[0] for p, _box in prepared])
        bar.update(len(chunk))
        for ((crop, padded, scale), box), quads in zip(prepared, quads_per_crop):
            for quad in quads:
                quad = np.asarray(quad, dtype=np.float64)
                quad_img = quad / scale + np.array([padded[0], padded[1]])
                qb = _quad_bbox(quad_img)
                if _inside_frac(qb, box) < _REFINE_OWNERSHIP_FRAC:
                    continue
                if any(_iou(qb, other) > _DEDUPE_IOU for other in kept_bboxes):
                    continue
                kept_bboxes.append(qb)
                rotated, angle = hough_deskew(crop, quad, bg_bgr)
                crops_for_rec.append(rotated)
                angles.append(angle)
                hits.append(OcrHit(quad=quad_img, padded_box=padded))

    bar.close()
    _LOG.debug("refine: %d merged boxes -> %d quads", len(merged), len(hits))

    boxes = recognize(crops_for_rec)
    _LOG.debug("recognize: %d/%d non-blank", sum(1 for b in boxes if b.text), len(boxes))
    flips = [float(b.flip_deg) for b in boxes]
    retry_counts: list = [0 if b.text else None for b in boxes]
    extras: list = [None] * len(boxes)
    for k, extra in ((1, 90.0), (2, 180.0), (3, 270.0)):
        blank = [i for i, b in enumerate(boxes) if not b.text]
        if not blank:
            break
        _LOG.debug("retry +%d deg: %d blank crops", int(extra), len(blank))
        retried = recognize_raw([np.ascontiguousarray(np.rot90(crops_for_rec[i], k)) for i in blank])
        for i, rbox in zip(blank, retried):
            if rbox.text:
                boxes[i] = rbox
                retry_counts[i] = k
                extras[i] = extra

    for i, (hit, box) in enumerate(zip(hits, boxes)):
        hit.text = box.text
        hit.confidence = box.confidence
        hit.retry_count = retry_counts[i]
        if box.text:
            effective = extras[i] if extras[i] is not None else flips[i]
            hit.angle_deg = normalize_rotation(angles[i] + effective)
    return hits


def run_ocr(
    bgr: np.ndarray, px_per_pt: float, bg_bgr: tuple[int, int, int], *, compute=None,
    detect_many=None, recognize=None, recognize_raw=None,
) -> OcrResult:
    """The whole tile -> merge -> refine -> recognize chain for one image.
    `detect_many`/`recognize`/`recognize_raw` default to PaddleOCR (local
    or via `compute`); tests inject fakes."""
    if detect_many is None:
        detect_many = _make_detect_many(compute)
    if recognize is None or recognize_raw is None:
        rec, rec_raw = _make_recognize(compute)
        recognize = recognize or rec
        recognize_raw = recognize_raw or rec_raw
    h, w = bgr.shape[:2]
    tiles = tile_grid(h, w)
    t0 = time.perf_counter()
    tile_boxes = detect_tiles(bgr, tiles, detect_many)
    _LOG.info("OCR tile detect: %d tiles -> %d boxes (%.1fs)", len(tiles), len(tile_boxes),
              time.perf_counter() - t0)
    merged = merge_cross_tile_boxes(tile_boxes)
    _LOG.debug("OCR cross-tile merge: %d -> %d boxes", len(tile_boxes), len(merged))
    t0 = time.perf_counter()
    hits = refine_and_recognize(bgr, merged, px_per_pt, bg_bgr, detect_many, recognize, recognize_raw)
    _LOG.info("OCR refine + recognize: %d merged -> %d hits, %d non-blank (%.1fs)", len(merged),
              len(hits), sum(1 for x in hits if x.text), time.perf_counter() - t0)
    return OcrResult(
        tiles=[tuple(float(v) for v in t) for t in tiles], tile_boxes=tile_boxes,
        merged=merged, hits=hits,
    )
