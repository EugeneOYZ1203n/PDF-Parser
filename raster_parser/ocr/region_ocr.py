"""Box recognition: recognize PaddleOCR's own detection boxes over the clean text raster.

Detection already ran (tiled + rotated) in ``OcrTextDrawingParser.parse`` and produced
``result.boxes`` -- line-level text detections. Rather than re-derive word groups by hand
(fragile on connected-component sizes), this recognizes each detection box directly:

  1. Crop the box's pixels from the clean text raster (``text_final`` -- drawings/lines
     already removed), so the recognizer sees black glyphs on white with no bleed.
  2. Orient it upright -- a tall-and-narrow box is a vertical label (e.g. MATCHLINE AREA),
     so it is rotated before recognition.
  3. Normalize the crop to the recognizer's preferred line height (reusing the native
     ``PdfRasterParser`` crop-normalization) and run PaddleOCR's *recognizer* on it.
  4. Keep only confident, non-empty results -- the confidence gate doubles as a final
     artefact filter (an isolated ``+``/arc/tick comes back low-confidence and is dropped).
  5. Map each box's pixel bbox to PDF points via ``to_points`` so it becomes a ``Text``
     ready for ``render_text_objects_to_pdf`` / native-vs-OCR evaluation.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing as mp
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz

from raster_parser.models import GlyphOrientation
from raster_parser.models import Text
from raster_parser.parsing.parser import PdfRasterParser
from raster_parser.parsing.text_drawing_parser import ocr_boxes, paddle_ocr_call

# Per-process OCR for ProcessPoolExecutor box-recog workers (spawn).
_REC_WORKER_OCR = None


def _rec_worker_init(lang: str, min_conf: float) -> None:
    """Limit BLAS threads and load one PaddleOCR per process (CPU)."""
    global _REC_WORKER_OCR
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    from raster_parser.parsing.text_drawing_parser import make_ocr

    _REC_WORKER_OCR = make_ocr(lang, min_conf, use_gpu=False)


def _box_rec_job(job: dict) -> dict:
    """Process-pool recognition job: load spilled crop(s) → word/conf."""
    with np.load(job["crop_path"]) as data:
        if job["vertical"]:
            w_cw, c_cw = _recognize(_REC_WORKER_OCR, data["cw"])
            w_ccw, c_ccw = _recognize(_REC_WORKER_OCR, data["ccw"])
            cw_won = c_cw >= c_ccw
            word, conf = (w_cw, c_cw) if cw_won else (w_ccw, c_ccw)
            rotate = 90 if cw_won else 270
        else:
            word, conf = _recognize(_REC_WORKER_OCR, data["norm"])
            rotate = 0
    return {
        "job_id": job["job_id"],
        "bidx": job["bidx"],
        "x0": job["x0"],
        "y0": job["y0"],
        "x1": job["x1"],
        "y1": job["y1"],
        "word": word,
        "conf": float(conf),
        "rotate": rotate,
        "vertical": job["vertical"],
    }


@dataclass
class RegionOcrResult:
    texts: list[Text]  # recognized words, bbox in PDF points
    kept_mask: np.ndarray  # uint8 mask: ink of boxes that passed the confidence gate
    n_boxes: int  # detection boxes fed in
    n_text: int  # boxes that recognized as confident text


def _extract_text_conf(res) -> tuple[str, float]:
    """Pull the first (text, confidence) pair out of a PaddleOCR result of any shape."""
    stack = [res]
    while stack:
        cur = stack.pop()
        if isinstance(cur, (list, tuple)):
            if (
                len(cur) >= 2
                and isinstance(cur[0], str)
                and isinstance(cur[1], (int, float, np.floating))
            ):
                return cur[0], float(cur[1])
            stack.extend(cur)
    return "", 0.0


def _recognize(ocr, img_gray) -> tuple[str, float]:
    """Recognition-only PaddleOCR on a single box crop, robust to version differences."""
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR) if img_gray.ndim == 2 else img_gray
    for kwargs in (
        {"det": False, "rec": True, "cls": True},
        {"det": False, "cls": True},
        {"det": False},
    ):
        try:
            res = paddle_ocr_call(ocr, img_bgr, **kwargs)
        except Exception:
            continue
        return _extract_text_conf(res)
    return "", 0.0


def _bbox_to_points(x0, y0, x1, y1, to_points, scale) -> tuple[float, float, float, float]:
    """Map a pixel bbox to an axis-aligned bbox in PDF points.

    With a ``to_points`` matrix (the inverse of the page render transform) all four corners
    are mapped so 90/180/270 rotations -- which swap width/height -- come out correct.
    Without it, fall back to a plain pixel*72/dpi scale.
    """
    if to_points is None:
        return (x0 * scale, y0 * scale, x1 * scale, y1 * scale)
    corners = [
        fitz.Point(x0, y0) * to_points,
        fitz.Point(x1, y0) * to_points,
        fitz.Point(x1, y1) * to_points,
        fitz.Point(x0, y1) * to_points,
    ]
    xs = [p.x for p in corners]
    ys = [p.y for p in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def _mode_height(boxes) -> float:
    """Most common line height (min side) across boxes -- the page's dominant text height."""
    heights = [min(abs(x1 - x0), abs(y1 - y0)) for x0, y0, x1, y1 in boxes]
    if not heights:
        return 0.0
    return float(Counter(heights).most_common(1)[0][0])


def split_words(
    text_gray: np.ndarray,
    box,
    *,
    gap_factor: float = 1.9,
    box_pad: int = 1,
) -> list[tuple[int, int, int, int]]:
    """Split a horizontal box into per-word sub-boxes by inter-column whitespace.

    Project the crop onto the x-axis: ink columns form runs (glyph strokes) separated by
    whitespace gaps. Traversing left->right, gaps inside a word (between letters) are small
    and roughly uniform, while the gap between two words is noticeably larger -- so any gap
    that jumps above ``gap_factor * (median gap)`` marks a word boundary. Runs are grouped
    into words at those boundaries and each word's x-extent (grown by ``box_pad``, clamped to
    the parent box) becomes a sub-box. A box with no ink, a single run, or only one word is
    returned unchanged.
    """
    x0, y0, x1, y1 = box
    crop = text_gray[y0:y1, x0:x1]
    if crop.size == 0:
        return [(x0, y0, x1, y1)]

    col_has_ink = (crop < 250).any(axis=0)  # columns that contain dark glyph ink
    cols = np.where(col_has_ink)[0]
    if cols.size == 0:
        return [(x0, y0, x1, y1)]

    # contiguous ink-column runs (a run is one stroke/letter-part)
    runs: list[tuple[int, int]] = []
    start = prev = int(cols[0])
    for c in cols[1:]:
        c = int(c)
        if c > prev + 1:
            runs.append((start, prev))
            start = c
        prev = c
    runs.append((start, prev))
    if len(runs) < 2:
        return [(x0, y0, x1, y1)]

    gaps = [runs[i + 1][0] - runs[i][1] - 1 for i in range(len(runs) - 1)]
    thresh = max(2.0, gap_factor * float(np.median(gaps)))  # a bigger-than-usual gap = new word

    words: list[list[tuple[int, int]]] = [[runs[0]]]
    for gap, run in zip(gaps, runs[1:]):
        if gap > thresh:
            words.append([run])  # gap grew -> start a new word
        else:
            words[-1].append(run)
    if len(words) < 2:
        return [(x0, y0, x1, y1)]

    out: list[tuple[int, int, int, int]] = []
    for grp in words:
        wx0 = max(x0, x0 + grp[0][0] - box_pad)
        wx1 = min(x1, x0 + grp[-1][1] + 1 + box_pad)
        if wx1 > wx0:
            out.append((wx0, y0, wx1, y1))
    return out or [(x0, y0, x1, y1)]


def refine_boxes(
    text_gray: np.ndarray,
    boxes,
    *,
    ocr,
    lang: str = "en",
    min_conf: float = 0.0,
    tall_factor: float = 1.5,
    pad: int = 8,
    box_pad: int = 2,
) -> list[tuple[int, int, int, int]]:
    """Second detection pass: split multi-line boxes into finer per-line/word boxes.

    A box that is much taller (``min(w,h)``) than the page's mode line height usually holds
    two+ stacked lines, which wrecks recognition. Such boxes are re-detected on their own
    crop (padded, white background) and replaced by the sub-boxes found; every other box is
    passed through unchanged. Coordinates are mapped back to the full-page frame.

    ``pad`` is the white border added around a crop before re-detection; ``box_pad`` grows
    each returned sub-box outward (clamped to the parent box so it can't spill into a
    neighbouring line) to give the recognizer a little margin.
    """
    if not boxes:
        return list(boxes)
    mode_h = _mode_height(boxes)
    if mode_h <= 0:
        return list(boxes)
    H, W = text_gray.shape[:2]
    thresh = tall_factor * mode_h
    out: list[tuple[int, int, int, int]] = []
    for x0, y0, x1, y1 in boxes:
        if min(x1 - x0, y1 - y0) <= thresh:
            out.append((x0, y0, x1, y1))
            continue
        crop = text_gray[y0:y1, x0:x1]
        if crop.size == 0:
            out.append((x0, y0, x1, y1))
            continue
        padded = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        sub = ocr_boxes(cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB), min_conf, lang, 0, 0, ocr=ocr)
        if not sub:
            out.append((x0, y0, x1, y1))
            continue
        for sx0, sy0, sx1, sy1 in sub:
            # map crop coords back to the page, then grow by box_pad (clamped to the parent
            # box so padding never bleeds into an adjacent line, and to image bounds).
            nx0 = max(x0, min(W, x0 + sx0 - pad - box_pad))
            ny0 = max(y0, min(H, y0 + sy0 - pad - box_pad))
            nx1 = min(x1, max(0, x0 + sx1 - pad + box_pad))
            ny1 = min(y1, max(0, y0 + sy1 - pad + box_pad))
            if nx1 > nx0 and ny1 > ny0:
                out.append((nx0, ny0, nx1, ny1))
    return out


def _batch_recognize(ocr, crops, batch: int = 256) -> list[tuple[str, float]]:
    """Recognize many crops via PaddleOCR's internal batch recognizer."""
    rec_fn = getattr(ocr, "text_recognizer", None)
    out: list[tuple[str, float]] = []
    for i in range(0, len(crops), batch):
        chunk = crops[i : i + batch]
        if rec_fn is not None:
            try:
                res, _ = rec_fn(chunk)
                out.extend((str(t), float(c)) for t, c in res)
                continue
            except Exception:
                pass
        out.extend(_recognize(ocr, im) for im in chunk)
    return out


@dataclass
class _GlyphCc:
    label: int
    x0: int
    y0: int
    x1: int
    y1: int
    area: int
    cx: float
    cy: float

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0


def _find_glyph_ccs(
    gray: np.ndarray,
    cov: np.ndarray,
    *,
    ink_thresh: int = 240,
    min_w: int = 12,
    max_w: int = 90,
    min_h: int = 15,
    max_h: int = 95,
    min_area: int = 40,
    max_area: int = 3500,
    min_fill: float = 0.12,
) -> list[_GlyphCc]:
    """Glyph-sized ink components whose centroid falls outside existing det boxes."""
    ink = (gray < ink_thresh).astype(np.uint8)
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(ink, 8)
    out: list[_GlyphCc] = []
    for i in range(1, n):
        L, T, w, h, area = (int(v) for v in stats[i])
        if not (min_w <= w <= max_w and min_h <= h <= max_h):
            continue
        if not (min_area <= area <= max_area):
            continue
        if area / float(w * h) < min_fill:
            continue
        cx, cy = cent[i]
        if cov[int(cy), int(cx)]:
            continue
        out.append(_GlyphCc(i, L, T, L + w, T + h, area, float(cx), float(cy)))
    return out


def _cc_size_similar(
    a: _GlyphCc, b: _GlyphCc, *, height_ratio: float = 0.5, area_ratio: float = 0.3
) -> bool:
    ha, hb = a.h, b.h
    if min(ha, hb) / max(ha, hb) < height_ratio:
        return False
    aa, ab = a.area, b.area
    return min(aa, ab) / max(aa, ab) >= area_ratio


def _cc_close(
    a: _GlyphCc, b: _GlyphCc, *, gap_factor: float = 2.5, align_factor: float = 0.6
) -> bool:
    """True when two glyph CCs sit on the same text line/column (horizontal or vertical)."""
    ref = max(a.h, b.h, 1)
    gx = max(0, max(a.x0, b.x0) - min(a.x1, b.x1))
    gy = max(0, max(a.y0, b.y0) - min(a.y1, b.y1))
    horiz = gy <= ref * align_factor and gx <= ref * gap_factor
    vert = gx <= ref * align_factor and gy <= ref * gap_factor
    return horiz or vert


def _cluster_glyph_ccs(
    ccs: list[_GlyphCc],
    *,
    height_ratio: float = 0.5,
    area_ratio: float = 0.3,
    gap_factor: float = 2.5,
    align_factor: float = 0.6,
    max_members: int = 6,
) -> list[list[_GlyphCc]]:
    """Union-find clusters of similar-sized, nearby glyph CCs (multi-char labels)."""
    n = len(ccs)
    if n == 0:
        return []
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if _cc_size_similar(
                ccs[i], ccs[j], height_ratio=height_ratio, area_ratio=area_ratio
            ) and _cc_close(ccs[i], ccs[j], gap_factor=gap_factor, align_factor=align_factor):
                union(i, j)

    buckets: dict[int, list[_GlyphCc]] = {}
    for i, cc in enumerate(ccs):
        buckets.setdefault(find(i), []).append(cc)

    clusters: list[list[_GlyphCc]] = []
    for grp in buckets.values():
        if len(grp) > max_members:
            clusters.extend([[cc] for cc in grp])
        else:
            clusters.append(grp)
    return clusters


def _cluster_bbox(grp: list[_GlyphCc]) -> tuple[int, int, int, int]:
    return (
        min(c.x0 for c in grp),
        min(c.y0 for c in grp),
        max(c.x1 for c in grp),
        max(c.y1 for c in grp),
    )


def recover_missed_glyphs(
    gray: np.ndarray,
    boxes,
    *,
    ocr,
    to_points: fitz.Matrix | None = None,
    dpi: int = 300,
    min_conf: float = 0.9,
    ink_thresh: int = 240,
    box_margin: int = 4,
    pad: int = 6,
    upscale: int = 4,
    max_len: int = 6,
    batch: int = 256,
    max_group_w: int = 140,
    max_group_h: int = 100,
) -> RegionOcrResult:
    """Recover isolated glyphs the detector never boxed.

    Like the earlier single-CC pass, but also groups nearby glyph-sized connected components
    that match in height/area (e.g. ``150``, ``A2``, ``CAT5`` shattered into separate strokes)
    and recognizes each cluster as one candidate.
    """
    H, W = gray.shape[:2]
    scale = 72.0 / dpi
    cov = np.zeros((H, W), np.uint8)
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(
            cov,
            (int(x0) - box_margin, int(y0) - box_margin),
            (int(x1) + box_margin, int(y1) + box_margin),
            255,
            -1,
        )

    ccs = _find_glyph_ccs(gray, cov, ink_thresh=ink_thresh)
    clusters = _cluster_glyph_ccs(ccs)
    cand: list[tuple[int, int, int, int, list[_GlyphCc]]] = []
    for grp in clusters:
        x0, y0, x1, y1 = _cluster_bbox(grp)
        if len(grp) > 1 and (x1 - x0 > max_group_w or y1 - y0 > max_group_h):
            for cc in grp:
                cand.append((cc.x0, cc.y0, cc.x1, cc.y1, [cc]))
        else:
            cand.append((x0, y0, x1, y1, grp))

    empty = np.zeros((H, W), np.uint8)
    if not cand:
        return RegionOcrResult(texts=[], kept_mask=empty, n_boxes=0, n_text=0)

    crops = []
    for x0, y0, x1, y1, _grp in cand:
        crop = gray[max(0, y0 - pad) : min(H, y1 + pad), max(0, x0 - pad) : min(W, x1 + pad)]
        big = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        norm = PdfRasterParser.normalise_crop_for_ocr(big)
        crops.append(cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR) if norm.ndim == 2 else norm)

    results = _batch_recognize(ocr, crops, batch=batch)
    texts: list[Text] = []
    kept = empty
    n_text = 0
    for (x0, y0, x1, y1, grp), (word, conf) in zip(cand, results):
        word = word.strip()
        if not word or conf < min_conf or len(word) > max_len or not word.isalnum():
            continue
        n_text += 1
        for cc in grp:
            sub = gray[cc.y0 : cc.y1, cc.x0 : cc.x1]
            kept[cc.y0 : cc.y1, cc.x0 : cc.x1][sub < ink_thresh] = 255
        bx0, by0, bx1, by1 = _bbox_to_points(x0, y0, x1, y1, to_points, scale)
        texts.append(
            Text(
                x0=bx0,
                y0=by0,
                x1=bx1,
                y1=by1,
                word=word,
                block_no=-1,
                line_no=0,
                word_no=0,
                rotate=0,
                glyph_orientation=GlyphOrientation.HORIZONTAL,
            )
        )
    return RegionOcrResult(texts=texts, kept_mask=kept, n_boxes=len(cand), n_text=n_text)


def recognize_boxes(
    text_gray: np.ndarray,
    boxes,
    *,
    ocr,
    to_points: fitz.Matrix | None = None,
    dpi: int = 300,
    min_conf: float = 0.0,
    vertical_aspect: float = 1.5,
    pad: int = 5,
    min_len: int = 1,
    word_split: bool = True,
    gap_factor: float = 1.8,
    workers: int = 1,
    lang: str = "en",
    scratch_dir: str | Path | None = None,
) -> RegionOcrResult:
    """Recognize each detection box's pixels from the clean text raster.

    ``text_gray`` is the grayscale text layer (original ink where text, white elsewhere).
    ``boxes`` is a list of ``(x0, y0, x1, y1)`` pixel detection boxes. For each box the
    pixels inside it are cropped into a fresh image, oriented upright, normalized and
    recognized. Confident boxes are returned as ``Text`` (bbox in PDF points) plus a mask of
    the ink that survived the confidence gate (use it to drop glyph-shaped non-text).

    With ``word_split`` (default), each HORIZONTAL box is first split into per-word sub-boxes
    (:func:`split_words`) so each word is recognized at true aspect -- a whole note line
    otherwise overflows the recognizer's width ceiling and is squished into gibberish.
    Vertical labels are never word-split.

    ``workers > 1`` uses ``ProcessPoolExecutor`` (spawn): spill normalized crops to disk,
    one PaddleOCR per worker process (CPU). ``workers == 1`` recognizes on this process
    with the provided ``ocr`` handle.
    """
    H, W = text_gray.shape[:2]
    scale = 72.0 / dpi
    kept = np.zeros((H, W), np.uint8)

    # Expand horizontal boxes into per-word sub-boxes (verticals stay whole).
    work: list[tuple[int, int, int, int, int, bool, bool]] = []
    for bidx, (bx0, by0, bx1, by1) in enumerate(boxes):
        bx0, by0, bx1, by1 = int(bx0), int(by0), int(bx1), int(by1)
        vertical = (by1 - by0) > vertical_aspect * (bx1 - bx0)
        if word_split and not vertical:
            parts = split_words(text_gray, (bx0, by0, bx1, by1), gap_factor=gap_factor)
            is_split = len(parts) > 1
            for wx0, wy0, wx1, wy1 in parts:
                work.append((bidx, wx0, wy0, wx1, wy1, False, is_split))
        else:
            work.append((bidx, bx0, by0, bx1, by1, vertical, False))

    def _prep(item: tuple[int, int, int, int, int, bool, bool]) -> dict | None:
        bidx, bx0, by0, bx1, by1, vertical, is_split = item
        x0, y0 = max(0, int(bx0) - pad), max(0, int(by0) - pad)
        x1, y1 = min(W, int(bx1) + pad), min(H, int(by1) + pad)
        if x1 <= x0 or y1 <= y0:
            return None
        crop = text_gray[y0:y1, x0:x1]
        if crop.size == 0 or (crop < 255).sum() == 0:
            return None

        if is_split:
            crop = crop.copy()
            crop[:, : bx0 - x0] = 255
            crop[:, bx1 - x0 :] = 255
            if (crop < 255).sum() == 0:
                return None

        if vertical:
            cw = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            ccw = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return {
                "bidx": bidx,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "vertical": True,
                "cw": PdfRasterParser.normalise_crop_for_ocr(cw),
                "ccw": PdfRasterParser.normalise_crop_for_ocr(ccw),
            }
        return {
            "bidx": bidx,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "vertical": False,
            "norm": PdfRasterParser.normalise_crop_for_ocr(crop),
        }

    prepared = [p for p in (_prep(item) for item in work) if p is not None]
    n_workers = max(1, int(workers))

    raw_hits: list[dict] = []
    if n_workers == 1 or len(prepared) <= 1:
        for p in prepared:
            if p["vertical"]:
                w_cw, c_cw = _recognize(ocr, p["cw"])
                w_ccw, c_ccw = _recognize(ocr, p["ccw"])
                cw_won = c_cw >= c_ccw
                word, conf = (w_cw, c_cw) if cw_won else (w_ccw, c_ccw)
                rotate = 90 if cw_won else 270
            else:
                word, conf = _recognize(ocr, p["norm"])
                rotate = 0
            raw_hits.append(
                {
                    "bidx": p["bidx"],
                    "x0": p["x0"],
                    "y0": p["y0"],
                    "x1": p["x1"],
                    "y1": p["y1"],
                    "word": word,
                    "conf": conf,
                    "rotate": rotate,
                    "vertical": p["vertical"],
                }
            )
    else:
        own_scratch = scratch_dir is None
        root = (
            Path(scratch_dir)
            if scratch_dir is not None
            else Path(tempfile.mkdtemp(prefix="rec_boxes_"))
        )
        root.mkdir(parents=True, exist_ok=True)
        crop_dir = root / "box_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        jobs: list[dict] = []
        try:
            for i, p in enumerate(prepared):
                path = crop_dir / f"b{i:05d}.npz"
                if p["vertical"]:
                    np.savez_compressed(path, cw=p["cw"], ccw=p["ccw"])
                else:
                    np.savez_compressed(path, norm=p["norm"])
                jobs.append(
                    {
                        "job_id": i,
                        "bidx": p["bidx"],
                        "x0": p["x0"],
                        "y0": p["y0"],
                        "x1": p["x1"],
                        "y1": p["y1"],
                        "vertical": p["vertical"],
                        "crop_path": str(path),
                    }
                )

            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=_rec_worker_init,
                initargs=(lang, float(min_conf)),
            ) as pool:
                futures = [pool.submit(_box_rec_job, job) for job in jobs]
                for fut in as_completed(futures):
                    raw_hits.append(fut.result())
            raw_hits.sort(key=lambda h: h["job_id"])
        finally:
            if own_scratch:
                for p in crop_dir.glob("*.npz"):
                    p.unlink(missing_ok=True)
                try:
                    crop_dir.rmdir()
                    root.rmdir()
                except OSError:
                    pass

    hits: list[tuple] = []
    for h in raw_hits:
        word = (h["word"] or "").strip()
        if not word or len(word) < min_len or h["conf"] < min_conf:
            continue
        x0, y0, x1, y1 = h["x0"], h["y0"], h["x1"], h["y1"]
        orient = GlyphOrientation.VERTICAL if h["vertical"] else GlyphOrientation.HORIZONTAL
        bpx0, bpy0, bpx1, bpy1 = _bbox_to_points(x0, y0, x1, y1, to_points, scale)
        text = Text(
            x0=bpx0,
            y0=bpy0,
            x1=bpx1,
            y1=bpy1,
            word=word,
            block_no=h["bidx"],
            line_no=0,
            word_no=0,
            rotate=h["rotate"],
            glyph_orientation=orient,
        )
        hits.append((x0, y0, x1, y1, text))

    texts: list[Text] = []
    for x0, y0, x1, y1, text in hits:
        sub = text_gray[y0:y1, x0:x1]
        kept[y0:y1, x0:x1][sub < 255] = 255
        texts.append(text)

    return RegionOcrResult(texts=texts, kept_mask=kept, n_boxes=len(boxes), n_text=len(texts))
